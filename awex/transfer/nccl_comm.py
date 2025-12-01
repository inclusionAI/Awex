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
import math
import os
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

from awex import logging
from awex.transfer.transfer_plan import slice_tensor

logger = logging.getLogger(__name__)


NUM_COMM_STREAMS = 64
_COMM_STREAMS_PER_DEVICE: Dict[int, List[torch.cuda.Stream]] = {}


def _get_comm_streams() -> List[torch.cuda.Stream]:
    """Get (and lazily create) CUDA streams for the current device.

    We create a pool of NUM_COMM_STREAMS streams per CUDA device to allow
    concurrent execution of P2P send/recv operations. If CUDA is not
    available, an empty list is returned and the default stream is used.
    """

    if not torch.cuda.is_available():
        return []

    device_index = torch.cuda.current_device()
    streams = _COMM_STREAMS_PER_DEVICE.get(device_index)
    if streams is None:
        streams = [
            torch.cuda.Stream(device=device_index) for _ in range(NUM_COMM_STREAMS)
        ]
        _COMM_STREAMS_PER_DEVICE[device_index] = streams
    return streams


class NcclColocateTransport:
    def __init__(self, transfer_rank, world_size):
        self.transfer_rank = transfer_rank
        self.world_size = world_size

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
    ):
        logger.info(f"train_to_infer_device_mapping {train_to_infer_device_mapping}")
        logger.info(f"infer_to_train_device_mapping {infer_to_train_device_mapping}")
        validate_rank_mappings(
            train_to_infer_device_mapping, infer_to_train_device_mapping
        )
        start_time = time.time()
        send_ops = dict(send_transfer_plan.operations)
        recv_ops = dict(recv_transfer_plan.operations)
        num_sends = sum(len(ops) for ops in send_ops.values())
        num_recvs = sum(len(ops) for ops in recv_ops.values())
        logger.info(
            f"Start to execute weights update for {rank_coordinate},"
            f"num_sends {num_sends}, num_recvs {num_recvs}"
        )
        p2p_send_op_groups, p2p_recv_op_groups = [], []
        tensors_to_copy = []
        stage = 0
        stage_offsets = []
        train_slice_context = {}
        while stage < world_size:
            p2p_send_ops, p2p_recv_ops = [], []
            stage_offset = stage % world_size
            send_to_rank = transfer_rank + stage
            if send_to_rank >= world_size:
                send_to_rank -= world_size
            recv_from_rank = transfer_rank - stage
            if recv_from_rank < 0:
                recv_from_rank += world_size
            had_send_entries = send_to_rank in send_ops
            if had_send_entries:
                for op in send_ops[send_to_rank]:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    if send_to_rank == transfer_rank:
                        tensors_to_copy.append(tensor_sliced)
                        continue
                    assert send_to_rank == op.recv_rank, (
                        f"rank unmatched: {send_to_rank} {op.recv_rank}"
                    )
                    p2p_op = dist.P2POp(
                        dist.isend,
                        tensor_sliced.clone(),
                        op.recv_rank,
                        group=weights_update_group,
                    )
                    p2p_send_ops.append((op, p2p_op))
            send_rank = infer_to_train_device_mapping[recv_from_rank]
            op_send_rank = train_to_infer_device_mapping[send_rank]
            had_recv_entries = send_rank in recv_ops
            if had_recv_entries:
                for op in recv_ops[send_rank]:
                    if recv_from_rank == transfer_rank:
                        continue
                    recv_tensor = recv_parameters[op.recv_shard_meta.name]
                    tensor_sliced = slice_tensor(recv_tensor, op, False)
                    assert send_rank == op.send_rank, (
                        f"rank unmatched: {send_rank} {op.send_rank}"
                    )
                    p2p_op = dist.P2POp(
                        dist.irecv,
                        tensor_sliced,
                        op_send_rank,
                        group=weights_update_group,
                    )
                    p2p_recv_ops.append((op, p2p_op))
            if had_send_entries and send_to_rank != transfer_rank and not p2p_send_ops:
                raise RuntimeError(
                    f"Inconsistent send plan for rank {rank_coordinate}: "
                    f"stage_offset {stage_offset}, transfer_rank {transfer_rank} "
                    f"expected sends to rank {send_to_rank} but built zero P2P ops."
                )
            if (
                had_recv_entries
                and recv_from_rank != transfer_rank
                and not p2p_recv_ops
            ):
                raise RuntimeError(
                    f"Inconsistent recv plan for rank {rank_coordinate}: "
                    f"stage_offset {stage_offset}, transfer_rank {transfer_rank} "
                    f"expected recvs from rank {op_send_rank} but built zero P2P ops."
                )
            p2p_send_op_groups.append(p2p_send_ops)
            p2p_recv_op_groups.append(p2p_recv_ops)
            stage_offsets.append(stage_offset)
            stage += 1
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

        # Execute p2p operations in two phases per stage so each batch_isend_irecv only
        # contains sends OR receives (not both), and peer ranks do complementary ops.
        for stage_idx, stage_offset in enumerate(stage_offsets):
            send_ops = p2p_send_op_groups[stage_idx]
            recv_ops = p2p_recv_op_groups[stage_idx]
            stage_name = f"{rank_coordinate} stage {stage_offset}"

            if stage_offset == 0:
                assert not send_ops
                assert not recv_ops
            else:
                # Compute partition so every sender and its peer receiver pick opposite phases
                partition = compute_two_phase_partition(
                    transfer_rank, stage_offset, world_size
                )
                # Phase 1: Partition 0 sends, partition 1 receives
                if partition == 0:
                    if send_ops:
                        execute_p2p_op_list(
                            send_ops, f"p2p send for {stage_name}", weights_update_group
                        )
                else:
                    if recv_ops:
                        execute_p2p_op_list(
                            recv_ops, f"p2p recv for {stage_name}", weights_update_group
                        )

                # Global barrier after Phase 1 to ensure all ranks complete before Phase 2
                # This is necessary because NCCL might have internal state that requires synchronization
                dist.barrier(group=weights_update_group)

                # Phase 2: Partition 0 receives, partition 1 sends
                if partition == 0:
                    if recv_ops:
                        execute_p2p_op_list(
                            recv_ops, f"p2p recv for {stage_name}", weights_update_group
                        )
                else:
                    if send_ops:
                        execute_p2p_op_list(
                            send_ops, f"p2p send for {stage_name}", weights_update_group
                        )

                # Global barrier after Phase 2 to ensure all ranks complete before next stage
                dist.barrier(group=weights_update_group)
            # No additional barrier here; Phase 2 per-pair barrier above prevents early advance

        # No final global barrier here
        logger.info(f"[{os.getpid()}] All p2p stages completed for {rank_coordinate}")

        duration = time.time() - start_time
        logger.info(
            f"Finished executing weights update for {rank_coordinate}, took {duration:.4f} seconds"
        )


def execute_tensors_to_copy(tensors_to_copy, copy_ops, recv_parameters, stage: str):
    start_time = time.time()
    num_ops = len(copy_ops)
    logger.info(f"Start to execute {num_ops} copy operations for {stage}")
    assert len(copy_ops) == len(tensors_to_copy), (
        f"Number of copy operations mismatch: {len(copy_ops)} != {len(tensors_to_copy)}"
    )
    for send_tensor, recv_op in zip(tensors_to_copy, copy_ops):
        recv_tensor = recv_parameters[recv_op.recv_shard_meta.name]
        recv_tensor_sliced = slice_tensor(recv_tensor, recv_op, False)
        recv_tensor_sliced.copy_(send_tensor)
    duration = time.time() - start_time
    torch.cuda.synchronize(device=torch.cuda.current_device())
    logger.info(
        f"Finished executing {num_ops} copy operations for {stage}, took {duration:.4f} seconds"
    )


def validate_rank_mappings(
    train_to_infer_device_mapping, infer_to_train_device_mapping
):
    for train_rank, infer_rank in train_to_infer_device_mapping.items():
        mapped_back = infer_to_train_device_mapping.get(infer_rank)
        if mapped_back != train_rank:
            raise ValueError(
                f"Inconsistent rank mapping: train_rank {train_rank} -> infer_rank {infer_rank} "
                f"but inverse mapping returns {mapped_back}"
            )
    for infer_rank, train_rank in infer_to_train_device_mapping.items():
        mapped_back = train_to_infer_device_mapping.get(train_rank)
        if mapped_back != infer_rank:
            raise ValueError(
                f"Inconsistent rank mapping: infer_rank {infer_rank} -> train_rank {train_rank} "
                f"but forward mapping returns {mapped_back}"
            )


def compute_two_phase_partition(rank, stage_idx, world_size):
    """
    Determine whether a rank should run sends or recvs in phase 1 for a stage.
    We color the graph formed by edges (r, r+stage_idx) so adjacent ranks have opposite colors.
    """
    if stage_idx == 0:
        return 0

    gcd_val = math.gcd(world_size, stage_idx)
    reduced_world = world_size // gcd_val
    reduced_stage = stage_idx // gcd_val
    stage_inv = pow(reduced_stage, -1, reduced_world)

    cycle_base = rank % gcd_val
    cycle_index = (rank - cycle_base) // gcd_val
    cycle_pos = (cycle_index * stage_inv) % reduced_world
    return cycle_pos % 2


hang_detector = ThreadPoolExecutor(max_workers=1)


def summary_meta(p2p_op_list):
    return [
        (
            plan_op.send_shard_meta.name,
            plan_op.recv_shard_meta.name,
            op.tensor.dtype,
            list(op.tensor.shape),
            op.tensor.is_contiguous(),
            op.peer,
        )
        for plan_op, op in p2p_op_list
    ]


def detect_hang(future, msg, p2p_op_list, timeout=30):
    try:
        future.result(timeout=timeout)
    except Exception:
        meta_summary = summary_meta(p2p_op_list)
        logger.exception(
            f"Exception while detecting hang at [{msg}], meta: {meta_summary}"
        )
        stack = get_stack_trace(os.getpid())
        logger.info(f"Stacktrace [{msg}]:\n{stack}")
        pass


def get_stack_trace(pid):
    """Get stack trace for a process using py-spy"""
    try:
        result = subprocess.run(
            ["py-spy", "dump", "--pid", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Timeout getting stack for PID {pid}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error getting stack for PID {pid}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for PID {pid}: {e}")
        return None


def execute_p2p_op_list(p2p_op_list, stage: str, weights_update_group):
    start_time = time.time()
    num_ops = len(p2p_op_list)

    num_sends = sum(1 for _, op in p2p_op_list if op.op == dist.isend)
    num_recvs = sum(1 for _, op in p2p_op_list if op.op == dist.irecv)

    logger.info(
        f"[{os.getpid()}] Start to execute {num_ops} p2p operations for {stage}, {num_sends} sends, {num_recvs} recvs"
    )
    if p2p_op_list:
        # Use synchronous send/recv
        msg = f"[{os.getpid()}] Using synchronous send/recv for {num_ops} operations for stage {stage}"
        logger.info(msg)
        future = Future()
        hang_detector.submit(detect_hang, future, msg, p2p_op_list)

        # Execute synchronous operations
        for _, p2p_op in p2p_op_list:
            if p2p_op.op == dist.isend:
                logger.debug(
                    f"[{os.getpid()}] send to rank {p2p_op.peer}, tensor shape {p2p_op.tensor.shape}"
                )
                dist.send(p2p_op.tensor, p2p_op.peer, group=p2p_op.group)
            elif p2p_op.op == dist.irecv:
                logger.debug(
                    f"[{os.getpid()}] recv from rank {p2p_op.peer}, tensor shape {p2p_op.tensor.shape}"
                )
                dist.recv(p2p_op.tensor, p2p_op.peer, group=p2p_op.group)
            else:
                raise ValueError(f"Unknown p2p op: {p2p_op.op}")

        torch.cuda.synchronize()
        future.set_result(True)
        logger.info(
            f"[{os.getpid()}] All sync operations completed for {num_ops} operations for {stage}"
        )
    else:
        logger.info(f"[{os.getpid()}] No p2p operations for {stage}")
    duration = time.time() - start_time
    logger.info(
        f"[{os.getpid()}] Finished executing {num_ops} p2p operations for {stage}, took {duration:.4f} seconds"
    )


@torch.no_grad()
def nccl_build_send_ops(parameters, transfer_plan, weights_update_group, copy_rank):
    send_progress = dict.fromkeys(transfer_plan.operations.keys(), 0)
    unfinished_ranks = set(transfer_plan.operations.keys())
    p2p_op_list = []
    copy_op_list = []
    train_slice_context = {}
    while len(unfinished_ranks) > 0:
        finished_ranks = set()
        for recv_rank in sorted(unfinished_ranks):
            operations = transfer_plan.operations[recv_rank]
            progress = send_progress[recv_rank]
            num_operations = len(operations)
            if progress < num_operations:
                op = operations[progress]
                send_tensor = parameters[op.send_shard_meta.name]
                tensor_sliced = slice_tensor(
                    send_tensor, op, True, slice_context=train_slice_context
                )
                if recv_rank == copy_rank:
                    copy_op_list.append(tensor_sliced)
                else:
                    p2p_op_list.append(
                        dist.P2POp(
                            dist.isend,
                            tensor_sliced,
                            recv_rank,
                            group=weights_update_group,
                        )
                    )
                send_progress[recv_rank] = progress + 1
            else:
                finished_ranks.add(recv_rank)
        for rank in finished_ranks:
            unfinished_ranks.remove(rank)
    return p2p_op_list, copy_op_list


def nccl_build_recv_ops(
    parameters: Dict[str, torch.Tensor], transfer_plan, weights_update_group
):
    p2p_op_list = []
    recv_progress = dict.fromkeys(transfer_plan.operations.keys(), 0)
    unfinished_ranks = set(transfer_plan.operations.keys())
    while len(unfinished_ranks) > 0:
        finished_ranks = set()
        for send_rank in sorted(unfinished_ranks):
            operations = transfer_plan.operations[send_rank]
            progress = recv_progress[send_rank]
            num_operations = len(operations)
            if progress < num_operations:
                op = operations[progress]
                recv_tensor = parameters[op.recv_shard_meta.name]
                tensor_sliced = slice_tensor(recv_tensor, op, False)
                p2p_op_list.append(
                    dist.P2POp(
                        dist.irecv,
                        tensor_sliced,
                        send_rank,
                        group=weights_update_group,
                    )
                )
                recv_progress[send_rank] = progress + 1
            else:
                finished_ranks.add(send_rank)
        for rank in finished_ranks:
            unfinished_ranks.remove(rank)
    return p2p_op_list


def _interleave_p2p_ops_by_peer(ops: Sequence[dist.P2POp]) -> List[dist.P2POp]:
    """Return a new list of P2P ops interleaved by peer rank.

    This performs a simple round-robin across peers so that operations to
    different ranks are interleaved, which tends to give better overlap for
    multi-rank communication patterns.
    """

    if not ops:
        return []

    by_peer: Dict[int, List[dist.P2POp]] = {}
    for op in ops:
        by_peer.setdefault(op.peer, []).append(op)

    peers = sorted(by_peer.keys())
    progress = dict.fromkeys(peers, 0)
    remaining = sum(len(v) for v in by_peer.values())
    interleaved: List[dist.P2POp] = []

    while remaining > 0:
        for peer in peers:
            idx = progress[peer]
            bucket = by_peer[peer]
            if idx >= len(bucket):
                continue
            interleaved.append(bucket[idx])
            progress[peer] = idx + 1
            remaining -= 1

    return interleaved


def _run_p2p_op(op: dist.P2POp, async_op: bool) -> Optional[dist.Work]:
    """Run a single P2P op, returning Work for async operations.

    The direction (send vs recv) is determined by ``op.op``; ``async_op``
    selects between blocking and non-blocking variants.
    """

    if op.op is dist.isend or op.op is dist.send:
        if async_op:
            return dist.isend(op.tensor, op.peer, group=op.group)
        dist.send(op.tensor, op.peer, group=op.group)
        return None
    if op.op is dist.irecv or op.op is dist.recv:
        if async_op:
            return dist.irecv(op.tensor, op.peer, group=op.group)
        dist.recv(op.tensor, op.peer, group=op.group)
        return None
    raise ValueError(f"Unsupported P2P op type: {op.op}")


def batch_send_recv(
    send_ops: Optional[Sequence[dist.P2POp]],
    recv_ops: Optional[Sequence[dist.P2POp]],
    blocking: bool = True,
    use_group: bool = True,
    use_stream: bool = True,
):
    """Execute send and recv P2P operations with optional grouping.

    Args:
        send_ops: Sequence of P2POp objects representing sends.
        recv_ops: Sequence of P2POp objects representing recvs.
        blocking: If False, use non-blocking isend/irecv and return Work
            handles. If True, use blocking send/recv and wait for
            completion before returning.
        use_group: If True, use ``torch.distributed.batch_isend_irecv`` to
            launch the operations as a group. Otherwise, execute them
            explicitly, interleaving across ranks and using a pool of CUDA
            streams for concurrency.
        use_stream: If True, use multiple cuda streams for send/recv
    Returns:
        List of ``torch.distributed.Work`` objects if ``async_op`` is True,
        otherwise an empty list.
    """

    send_ops = list(send_ops) if send_ops is not None else []
    recv_ops = list(recv_ops) if recv_ops is not None else []

    if not send_ops and not recv_ops:
        return []

    # Grouped execution path using batch_isend_irecv.
    if use_group:
        all_ops = _interleave_p2p_ops_by_peer(send_ops + recv_ops)
        works = dist.batch_isend_irecv(all_ops)
        if not blocking:
            return works
        for work in works:
            work.wait()
        torch.cuda.synchronize()
        return []

    # Manual execution path with explicit interleaving and CUDA streams.
    streams = _get_comm_streams() if use_stream else []
    num_streams = len(streams)

    # Interleave sends and recvs across peers for better overlap.
    all_ops = _interleave_p2p_ops_by_peer(send_ops + recv_ops)
    works = []

    # Assign a fixed stream index per peer so that all operations to the
    # same peer execute on the same CUDA stream, preserving ordering within
    # that peer while allowing different peers to run concurrently.
    peers = sorted({op.peer for op in all_ops})
    peer_to_stream_idx: Dict[int, int] = {}
    if use_stream > 0:
        for idx, peer in enumerate(peers):
            peer_to_stream_idx[peer] = idx % num_streams

    for op in all_ops:
        current_stream = None
        if use_stream:
            stream_idx = peer_to_stream_idx.get(op.peer, 0)
            current_stream = streams[stream_idx]

        if use_stream and current_stream is not None:
            # Execute on a dedicated CUDA stream for this peer.
            with torch.cuda.stream(current_stream):
                work = _run_p2p_op(op, not blocking)
        else:
            work = _run_p2p_op(op, not blocking)

        if work is not None:
            works.append(work)

    if not blocking:
        return works

    # For blocking mode, make sure all CUDA work is completed.
    torch.cuda.synchronize()
    return []
