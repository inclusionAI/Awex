# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os

import torch
import torch.distributed as dist

from awex import logging

logger = logging.getLogger(__name__)


# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")
    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


def create_pair_subgroups_from_parent(parent_group, world_size):
    """
    Create pairwise 2-rank NCCL subgroups for all pairs of ranks in a parent process group.

    This function creates true subgroups from the parent_group by using _new_process_group_helper
    with a derived store from the parent group, rather than using dist.new_group() which always
    creates subgroups from the default process group.

    Args:
        parent_group: The parent process group to create subgroups from
        world_size: The number of ranks in the parent group

    Returns:
        dict: Dictionary mapping (i, j) tuples to ProcessGroup objects, where i < j
              are ranks within the parent group. Returns None if creation fails.

    Example:
        # Create pair subgroups from weights_update_group
        pair_subgroups = create_pair_subgroups_from_parent(weights_update_group, 128)
        # Use subgroup for barrier between ranks 5 and 10
        pg = pair_subgroups[(5, 10)]
        dist.barrier(group=pg)
    """
    try:
        from torch.distributed.distributed_c10d import (
            PrefixStore,
            _new_process_group_helper,
            _world,
        )

        # Get my rank within the parent group
        my_rank = dist.get_rank(group=parent_group)

        # Get the backend and store from the parent group
        backend, parent_store = _world.pg_map[parent_group]

        logger.info(
            f"Creating pair subgroups from parent group with backend={backend}, my_rank={my_rank}, world_size={world_size}"
        )

        # Create subgroups for all pairs
        pair_subgroups = {}
        for i in range(world_size):
            for j in range(i + 1, world_size):
                # Only processes that are part of this pair create the group
                if my_rank != i and my_rank != j:
                    # This process is not part of this pair subgroup
                    continue

                # Create a unique prefix store for this pair
                pair_name = f"pair_{i}_{j}"
                pair_store = PrefixStore(pair_name, parent_store)

                # Determine this process's rank within the pair (0 or 1)
                pair_rank = 0 if my_rank == i else 1
                pair_world_size = 2

                # Get timeout from parent group
                timeout = parent_group._timeout

                # Determine the appropriate parameter name based on PyTorch version
                pg_options_param_name = (
                    "backend_options"
                    if str(torch.__version__) >= "2.6"
                    else "pg_options"
                )

                # Create the pair subgroup using _new_process_group_helper
                pg, _ = _new_process_group_helper(
                    pair_world_size,
                    pair_rank,
                    [],
                    backend,
                    pair_store,
                    group_name=pair_name,
                    **{pg_options_param_name: None},
                    timeout=timeout,
                )

                # Set up the rank mapping for this pair subgroup
                # Map pair ranks (0, 1) to parent group ranks (i, j)
                _world.pg_group_ranks[pg] = {0: i, 1: j}

                pair_subgroups[(i, j)] = pg
                logger.info(
                    f"Rank {my_rank} created pair subgroup ({i}, {j}) with pair_rank={pair_rank}"
                )

        logger.info(
            f"Rank {my_rank} built {len(pair_subgroups)} pair subgroups from parent group"
        )
        return pair_subgroups if pair_subgroups else None

    except Exception as e:
        logger.exception(f"Failed to build pair subgroups: {e}")
        return None


def init_weights_update_group(
    master_address,
    master_port,
    rank,
    world_size,
    group_name,
    backend="nccl",
    role="",
):
    """Initialize the Torch process group for model parameter updates."""
    assert torch.distributed.is_initialized(), (
        "Default torch process group must be initialized"
    )
    assert group_name != "", "Group name cannot be empty"

    logger.info(
        f"init custom process group for {role}: master_address={master_address}, master_port={master_port}, "
        f"rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}, "
        f"current device id {torch.cuda.current_device()} "
        f"CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"Local rank env {os.environ.get('LOCAL_RANK')} DEVICE env {os.environ.get('DEVICE')} "
        f"Global rank env {os.environ.get('RANK')}"
    )

    try:
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(f"Initialized custom process group: {group}")
        return group
    except Exception as e:
        raise RuntimeError(f"Failed to initialize custom process group: {e}.") from e


def setup_batch_isend_irecv(
    process_group, rank, world_size, tensor_size=10 * 10, dtype=torch.float32
):
    """
    Perform a simple communication using batch_isend_irecv to avoid the hang for later sub-ranks.

    Args:
    process_group (ProcessGroup): The process group to work on.
    tensor_size (int): Size of the tensor to send/receive.
    dtype (torch.dtype): Data type of the tensor.
    """
    assert process_group is not None, "Process group cannot be None"
    device = torch.cuda.current_device()
    logger.info(
        f"Setup batch isend irecv for rank {rank} world size {world_size} device {device}"
    )

    # Create tensors for sending and receiving
    send_tensor = torch.full(
        (tensor_size,), rank, dtype=dtype, device=device, requires_grad=False
    )
    recv_tensor = torch.zeros(
        (tensor_size,), dtype=dtype, device=device, requires_grad=False
    )

    # Prepare the ops for batch_isend_irecv
    ops = []

    # First half of ranks receive from rank + half
    mid_point = world_size // 2
    if rank < mid_point:
        # First half: receive from rank + half
        target_rank = rank + mid_point
        if target_rank < world_size:
            ops.append(
                dist.P2POp(dist.irecv, recv_tensor, target_rank, group=process_group)
            )
    else:
        # Second half: send to rank - half
        target_rank = rank - mid_point
        if target_rank >= 0:
            ops.append(
                dist.P2POp(dist.isend, send_tensor, target_rank, group=process_group)
            )

    # Execute batch_isend_irecv
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        # Wait for all communications to complete
        for req in reqs:
            req.wait()

    # Synchronize
    torch.cuda.synchronize(device=torch.cuda.current_device())
    dist.barrier(group=process_group, device_ids=[torch.cuda.current_device()])

    logger.info(
        f"Simple communication completed for process group of size {world_size}"
    )

    # Verify the results
    if rank < mid_point and rank + mid_point < world_size:
        expected_value = rank + mid_point
        assert torch.all(recv_tensor == expected_value), (
            f"Rank {rank} received incorrect data from rank {rank + mid_point}"
        )

    logger.info("Simple communication verification successful")
