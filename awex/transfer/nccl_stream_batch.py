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

from awex import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager

import torch
import torch.distributed as dist

from awex.transfer.nccl_comm import (
    execute_tensors_to_copy,
    validate_rank_mappings,
    detect_hang,
)
from awex.transfer.transfer_plan import slice_tensor

logger = logging.getLogger(__name__)
hang_detector = ThreadPoolExecutor(max_workers=1)


class NcclColocateStreamBatchTransport:
    def __init__(self, transfer_rank, world_size):
        self.transfer_rank = transfer_rank
        self.world_size = world_size
        self._streams = []

    def update_weights_in_colocate_mode(
        self,
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_transfer_plan,
        recv_transfer_plan,
        weights_update_group,
        send_parameters,
        recv_parameters,
        *,
        stream=True,
        **kwargs,
    ):
        logger.info(f"train_to_infer_device_mapping {train_to_infer_device_mapping}")
        logger.info(f"infer_to_train_device_mapping {infer_to_train_device_mapping}")
        logger.info("Using RECURSIVE PARTITION batch_isend_irecv with O(log N) rounds")
        validate_rank_mappings(
            train_to_infer_device_mapping, infer_to_train_device_mapping
        )
        start_time = time.time()

        # Get send/recv operations dict
        send_ops = dict(send_transfer_plan.operations)
        recv_ops = dict(recv_transfer_plan.operations)
        num_sends = sum(len(ops) for ops in send_ops.values())
        num_recvs = sum(len(ops) for ops in recv_ops.values())
        logger.info(
            f"Start to execute weights update for {rank_coordinate}, "
            f"num_sends {num_sends}, num_recvs {num_recvs}"
        )

        # Build P2P operations with sliced tensors
        all_send_p2p_ops = {}  # peer_rank -> List[(plan_op, p2p_op)]
        all_recv_p2p_ops = {}  # peer_rank -> List[(plan_op, p2p_op)]
        tensors_to_copy = []
        train_slice_context = {}

        # Process send operations
        for peer_rank, ops in send_ops.items():
            # Map training rank to inference rank in colocate mode
            mapped_peer_rank = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer_rank == transfer_rank:
                # Self-copy operations
                for op in ops:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    tensors_to_copy.append(tensor_sliced)
            else:
                # P2P send operations
                p2p_ops = []
                for op in ops:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    # Use mapped inference rank for P2P operation
                    recv_rank = train_to_infer_device_mapping.get(
                        op.recv_rank, op.recv_rank
                    )
                    p2p_op = dist.P2POp(
                        dist.isend,
                        tensor_sliced.clone(),
                        recv_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((op, p2p_op))
                all_send_p2p_ops[mapped_peer_rank] = p2p_ops

        # Process recv operations
        for send_rank, ops in recv_ops.items():
            recv_from_rank = train_to_infer_device_mapping[send_rank]
            if recv_from_rank == transfer_rank:
                # Skip self-recv (handled by tensors_to_copy)
                continue
            p2p_ops = []
            for op in ops:
                recv_tensor = recv_parameters[op.recv_shard_meta.name]
                tensor_sliced = slice_tensor(recv_tensor, op, False)
                p2p_op = dist.P2POp(
                    dist.irecv,
                    tensor_sliced,
                    recv_from_rank,
                    group=weights_update_group,
                )
                p2p_ops.append((op, p2p_op))
            all_recv_p2p_ops[recv_from_rank] = p2p_ops

        # Execute self-copy operations
        if len(tensors_to_copy) > 0:
            send_rank = infer_to_train_device_mapping[transfer_rank]
            execute_tensors_to_copy(
                tensors_to_copy,
                recv_transfer_plan.operations[send_rank],
                recv_parameters,
                f"tensor copy for {rank_coordinate}",
            )
        else:
            logger.info(f"No tensors to copy for {rank_coordinate}")

        future = Future()
        total_send_ops = sum(len(ops) for ops in all_send_p2p_ops.values())
        total_recv_ops = sum(len(ops) for ops in all_recv_p2p_ops.values())
        msg = f"[{os.getpid()}] execute {total_send_ops} sends {total_recv_ops} recvs with recursive partition for {rank_coordinate}"
        hang_detector.submit(detect_hang, future, msg, [], timeout=60)

        # Execute recursive partition transfer
        func = (
            self.execute_recursive_partition_stream_transfer
            if stream
            else execute_recursive_partition_transfer
        )
        func(
            transfer_rank,
            world_size,
            all_send_p2p_ops,
            all_recv_p2p_ops,
            weights_update_group,
            rank_coordinate,
        )

        future.set_result(True)
        torch.cuda.synchronize()
        duration = time.time() - start_time
        logger.info(
            f"Finished executing weights update for {rank_coordinate}, took {duration:.4f} seconds"
        )

    def execute_recursive_partition_stream_transfer(
        self,
        transfer_rank,
        world_size,
        all_send_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
        all_recv_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
        weights_update_group,
        rank_coordinate,
    ):
        """
        Execute P2P transfer using recursive partition algorithm.

        Algorithm:
        - Round 1: partition_size=world_size, split into [0, world_size/2) and [world_size/2, world_size)
          - First half sends to second half
          - Second half recvs from first half
          - First half recvs from second half
          - Second half sends to first half

        - Round 2: partition_size=world_size/2, operate on each half independently
        - ...
        - Continue until partition_size=2

        Total rounds: log2(world_size)
        Each rank sends/recvs to/from ALL ranks in the other half of its partition.
        """
        num_rounds = int(math.log2(world_size))
        dist.barrier(
            group=weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}]  Starting recursive partition transfer with {num_rounds} rounds"
        )
        for round_idx in range(num_rounds):
            partition_size = world_size // (2**round_idx)
            half = partition_size // 2

            # Determine my partition base (which partition I'm in)
            partition_base = (transfer_rank // partition_size) * partition_size
            partition_end = partition_base + partition_size
            offset_in_partition = transfer_rank - partition_base

            # Determine if I'm in first half or second half of my partition
            in_first_half = offset_in_partition < half

            # Determine the range of ranks in the other half
            if in_first_half:
                other_half_start = partition_base + half
                other_half_end = partition_end
            else:
                other_half_start = partition_base
                other_half_end = partition_base + half
            logger.info(
                f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx}: partition_size={partition_size}, "
                f"partition=[{partition_base}, {partition_end}), half={half}, "
                f"in_first_half={in_first_half}, other_half=[{other_half_start}, {other_half_end})"
            )

            round_start = time.time()
            num_ops = 0
            # === PHASE 1: First half sends to second half, second half receives from first half ===
            if in_first_half:
                # Collect all send operations to ranks in the other half
                for peer_rank in range(other_half_start, other_half_end):
                    if peer_rank in all_send_p2p_ops:
                        num_ops += self._execute_ops(all_send_p2p_ops[peer_rank])
            else:
                # Collect all recv operations from ranks in the other half
                for peer_rank in range(other_half_start, other_half_end):
                    if peer_rank in all_recv_p2p_ops:
                        num_ops += self._execute_ops(all_recv_p2p_ops[peer_rank])
            logger.info(
                f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 1: enqueued {num_ops} "
                f"{'sends' if in_first_half else 'recvs'}"
            )
            torch.cuda.synchronize()
            logger.info(
                f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 1: executed {num_ops} "
                f"{'sends' if in_first_half else 'recvs'}"
            )
            num_ops2 = 0
            # === PHASE 2: First half receives from second half, second half sends to first half ===
            if in_first_half:
                # Collect all recv operations from ranks in the other half
                for peer_rank in range(other_half_start, other_half_end):
                    if peer_rank in all_recv_p2p_ops:
                        num_ops2 += self._execute_ops(all_recv_p2p_ops[peer_rank])
            else:
                # Collect all send operations to ranks in the other half
                for peer_rank in range(other_half_start, other_half_end):
                    if peer_rank in all_send_p2p_ops:
                        num_ops2 += self._execute_ops(all_send_p2p_ops[peer_rank])
            logger.info(
                f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 2: enqueued {num_ops2} "
                f"{'recvs' if in_first_half else 'sends'}"
            )
            torch.cuda.synchronize()
            logger.info(
                f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 2: executed {num_ops2} "
                f"{'recvs' if in_first_half else 'sends'}"
            )
            round_duration = time.time() - round_start
            logger.info(
                f"[{os.getpid()}] Round {round_idx} completed: "
                f"phase1={num_ops} ops, phase2={num_ops2} ops, "
                f"took {round_duration:.4f}s"
            )
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] All {num_rounds} rounds completed"
        )
        # Final barrier to ensure all ranks complete before proceeding
        dist.barrier(
            group=weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(f"[{os.getpid()}] [{rank_coordinate}] Final barrier passed")

    def _execute_ops(self, ops):
        num_ops = 0
        if not ops:
            return num_ops
        with self._get_stream():
            for plan_op, p2p_op in ops:
                if p2p_op.op is dist.isend:
                    dist.send(p2p_op.tensor, p2p_op.peer, group=p2p_op.group)
                else:
                    dist.recv(p2p_op.tensor, p2p_op.peer, group=p2p_op.group)
                num_ops += 1
        return num_ops

    @contextmanager
    def _get_stream(self):
        if not self._streams:
            self._streams.append(torch.cuda.Stream())
        stream = self._streams.pop()
        try:
            with torch.cuda.stream(stream):
                yield  # Now the body of "with _get_stream()" executes under stream context
        finally:
            self._streams.append(stream)


def execute_recursive_partition_transfer(
    transfer_rank,
    world_size,
    all_send_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
    all_recv_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
    weights_update_group,
    rank_coordinate,
):
    """
    Execute P2P transfer using recursive partition algorithm with `batch_isend_irecv`.
    FIXME: It hang sometimes, seems `batch_isend_irecv` can't handle asymmetric p2p communication.
    """
    num_rounds = int(math.log2(world_size))
    dist.barrier(group=weights_update_group, device_ids=[torch.cuda.current_device()])
    logger.info(
        f"[{os.getpid()}] Starting recursive partition transfer with {num_rounds} rounds for {rank_coordinate}"
    )

    for round_idx in range(num_rounds):
        partition_size = world_size // (2**round_idx)
        half = partition_size // 2

        # Determine my partition base (which partition I'm in)
        partition_base = (transfer_rank // partition_size) * partition_size
        partition_end = partition_base + partition_size
        offset_in_partition = transfer_rank - partition_base

        # Determine if I'm in first half or second half of my partition
        in_first_half = offset_in_partition < half

        # Determine the range of ranks in the other half
        if in_first_half:
            other_half_start = partition_base + half
            other_half_end = partition_end
        else:
            other_half_start = partition_base
            other_half_end = partition_base + half

        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx}: partition_size={partition_size}, "
            f"partition=[{partition_base}, {partition_end}), half={half}, "
            f"in_first_half={in_first_half}, other_half=[{other_half_start}, {other_half_end})"
        )

        round_start = time.time()

        # === PHASE 1: First half sends to second half, second half receives from first half ===
        # In colocate mode, all ranks are in the SAME NCCL group, so we need to call
        # batch_isend_irecv with BOTH sends and recvs together (not separately)
        phase1_ops = []
        if in_first_half:
            # Collect all send operations to ranks in the other half
            for peer_rank in range(other_half_start, other_half_end):
                if peer_rank in all_send_p2p_ops:
                    for plan_op, p2p_op in all_send_p2p_ops[peer_rank]:
                        phase1_ops.append(p2p_op)
        else:
            # Collect all recv operations from ranks in the other half
            for peer_rank in range(other_half_start, other_half_end):
                if peer_rank in all_recv_p2p_ops:
                    for plan_op, p2p_op in all_recv_p2p_ops[peer_rank]:
                        phase1_ops.append(p2p_op)

        # All ranks call batch_isend_irecv together (some with sends, some with recvs)
        # IMPORTANT: ALL ranks must call batch_isend_irecv, even if they have no operations
        # Otherwise, sends from ranks with operations won't match with receives from ranks without
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 1: executing {len(phase1_ops)} "
            f"{'sends' if in_first_half else 'recvs'}"
        )
        reqs = dist.batch_isend_irecv(phase1_ops)
        for req in reqs:
            req.wait()
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 1: completed {len(phase1_ops)} ops"
        )

        # MUST synchronize before barrier to ensure CUDA operations complete
        torch.cuda.synchronize()
        # Barrier to ensure all ranks complete Phase 1 before starting Phase 2
        dist.barrier(
            group=weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 1: barrier passed"
        )

        # === PHASE 2: First half receives from second half, second half sends to first half ===
        phase2_ops = []
        if in_first_half:
            # Collect all recv operations from ranks in the other half
            for peer_rank in range(other_half_start, other_half_end):
                if peer_rank in all_recv_p2p_ops:
                    for plan_op, p2p_op in all_recv_p2p_ops[peer_rank]:
                        phase2_ops.append(p2p_op)
        else:
            # Collect all send operations to ranks in the other half
            for peer_rank in range(other_half_start, other_half_end):
                if peer_rank in all_send_p2p_ops:
                    for plan_op, p2p_op in all_send_p2p_ops[peer_rank]:
                        phase2_ops.append(p2p_op)

        # All ranks call batch_isend_irecv together (some with recvs, some with sends)
        # IMPORTANT: ALL ranks must call batch_isend_irecv, even if they have no operations
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 2: executing {len(phase2_ops)} "
            f"{'recvs' if in_first_half else 'sends'}"
        )
        reqs = dist.batch_isend_irecv(phase2_ops)
        for req in reqs:
            req.wait()
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 2: completed {len(phase2_ops)} ops"
        )

        # MUST synchronize before barrier to ensure CUDA operations complete
        torch.cuda.synchronize()
        # Barrier to ensure all ranks complete Phase 2 before next round
        dist.barrier(
            group=weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} Phase 2: barrier passed"
        )

        round_duration = time.time() - round_start
        logger.info(
            f"[{os.getpid()}] [{rank_coordinate}] Round {round_idx} completed: "
            f"phase1={len(phase1_ops)} ops, phase2={len(phase2_ops)} ops, "
            f"took {round_duration:.4f}s"
        )

    logger.info(
        f"[{os.getpid()}] [{rank_coordinate}] All {num_rounds} rounds completed for {rank_coordinate}"
    )

    # Final barrier to ensure all ranks complete before proceeding
    dist.barrier(group=weights_update_group, device_ids=[torch.cuda.current_device()])
    logger.info(
        f"[{os.getpid()}] [{rank_coordinate}] Final barrier passed for {rank_coordinate}"
    )
