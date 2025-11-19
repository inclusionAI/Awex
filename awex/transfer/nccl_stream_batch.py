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
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch
import torch.distributed as dist

from awex import logging
from awex.transfer.nccl_comm import (
    detect_hang,
    execute_tensors_to_copy,
    validate_rank_mappings,
)
from awex.transfer.transfer_plan import slice_tensor

logger = logging.getLogger(__name__)
hang_detector = ThreadPoolExecutor(max_workers=1)


class NcclColocateStreamBatchTransport:
    MAX_STREAMS = 64

    def __init__(self, transfer_rank, world_size):
        self.transfer_rank = transfer_rank
        self.world_size = world_size
        # Initialize a fixed pool of CUDA streams
        self._stream_pool = [
            torch.cuda.Stream() for _ in range(min(self.MAX_STREAMS, world_size))
        ]

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
        step_id=-1,
        async_op=True,
        **kwargs,
    ):
        logger.info("Using RECURSIVE PARTITION batch_isend_irecv with O(log N) rounds")
        task_id = f"{rank_coordinate}-{step_id}"
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
            f"Start to execute weights update for {task_id}, "
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
                        dist.isend if async_op else dist.send,
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
                    dist.irecv if async_op else dist.recv,
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
                f"tensor copy for {task_id}",
            )
        else:
            logger.info(f"No tensors to copy for {task_id}")

        future = Future()
        total_send_ops = sum(len(ops) for ops in all_send_p2p_ops.values())
        total_recv_ops = sum(len(ops) for ops in all_recv_p2p_ops.values())
        msg = f"[{os.getpid()}] execute {total_send_ops} sends {total_recv_ops} recvs with recursive partition for {task_id}"
        hang_detector.submit(detect_hang, future, msg, [], timeout=60)

        # Execute recursive partition transfer
        # FIXME: batch_isend_irecv hang sometimes, seems `batch_isend_irecv` can't handle asymmetric p2p communication.
        # so we use send/recv directly
        self.execute_recursive_partition_stream_transfer(
            transfer_rank,
            world_size,
            all_send_p2p_ops,
            all_recv_p2p_ops,
            weights_update_group,
            rank_coordinate,
            step_id,
        )

        torch.cuda.synchronize()
        future.set_result(True)
        duration = time.time() - start_time
        logger.info(
            f"Finished executing weights update for {task_id}, took {duration:.4f} seconds"
        )

    def execute_recursive_partition_stream_transfer(
        self,
        transfer_rank,
        world_size,
        all_send_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
        all_recv_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
        weights_update_group,
        rank_coordinate,
        step_id,
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
        prefix = f"[{os.getpid()}] [{rank_coordinate}] [step {step_id}]"
        start_time = time.time()
        logger.info(
            f"{prefix} Starting recursive partition transfer with {num_rounds} rounds"
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
                f"{prefix} Round {round_idx}: partition_size={partition_size}, "
                f"partition=[{partition_base}, {partition_end}), half={half}, "
                f"in_first_half={in_first_half}, other_half=[{other_half_start}, {other_half_end})"
            )

            round_start = time.time()
            # === PHASE 1: First half sends to second half, second half receives from first half ===
            if in_first_half:
                # Execute all send operations to ranks in the other half with concurrent execution
                num_ops = self._execute_ops_concurrent(
                    all_send_p2p_ops, range(other_half_start, other_half_end)
                )
            else:
                # Execute all recv operations from ranks in the other half with concurrent execution
                num_ops = self._execute_ops_concurrent(
                    all_recv_p2p_ops, range(other_half_start, other_half_end)
                )
            logger.info(
                f"{prefix} Round {round_idx} Phase 1: enqueued {num_ops} "
                f"{'sends' if in_first_half else 'recvs'}"
            )
            # === PHASE 2: First half receives from second half, second half sends to first half ===
            if in_first_half:
                # Execute all recv operations from ranks in the other half with concurrent execution
                num_ops2 = self._execute_ops_concurrent(
                    all_recv_p2p_ops, range(other_half_start, other_half_end)
                )
            else:
                # Execute all send operations to ranks in the other half with concurrent execution
                num_ops2 = self._execute_ops_concurrent(
                    all_send_p2p_ops, range(other_half_start, other_half_end)
                )
            logger.info(
                f"{prefix} Round {round_idx} Phase 2: enqueued {num_ops2} "
                f"{'recvs' if in_first_half else 'sends'}"
            )
            round_duration = time.time() - round_start
            logger.info(
                f"[{os.getpid()}] Round {round_idx} completed: "
                f"phase1={num_ops} ops, phase2={num_ops2} ops, "
                f"took {round_duration:.4f}s"
            )
        torch.cuda.synchronize()
        duration = time.time() - start_time
        logger.info(f"{prefix} All {num_rounds} rounds completed in {duration:.4f}s")

    def _execute_ops_concurrent(self, ops_dict, peer_ranks):
        """
        Execute ops from multiple peers with interleaved execution for better concurrency.

        Instead of executing all ops for one peer sequentially (peer1_all_ops, peer2_all_ops, ...),
        this method interleaves operations in a round-robin fashion (peer1_op1, peer2_op1, ...,
        peer1_op2, peer2_op2, ...). This allows operations from different peers to overlap and
        execute concurrently on the GPU.

        Each peer rank consistently uses the same CUDA stream to maintain ordering within
        that peer's operations, while different peers use different streams (up to max)
        for concurrent execution.

        Args:
            ops_dict: Dictionary mapping peer_rank to list of (plan_op, p2p_op) tuples
            peer_ranks: Range or iterable of peer ranks to process

        Returns:
            Total number of ops executed
        """
        # Collect ops from all peers that have operations, along with their peer_rank
        peer_ops_with_rank = []
        active_peer_ranks = []
        for peer_rank in peer_ranks:
            if peer_rank in ops_dict:
                peer_ops_with_rank.append((peer_rank, ops_dict[peer_rank]))
                active_peer_ranks.append(peer_rank)

        if not peer_ops_with_rank:
            return 0

        # Allocate stream indices sequentially to active peer ranks for even distribution
        # This ensures ranks are evenly distributed across available streams
        peer_to_stream_idx = {}
        for idx, peer_rank in enumerate(active_peer_ranks):
            stream_idx = idx % len(self._stream_pool)
            peer_to_stream_idx[peer_rank] = stream_idx

        # Find the maximum number of ops across all peers
        max_ops = max(len(ops) for _, ops in peer_ops_with_rank)
        total_ops = 0

        # Execute ops in round-robin fashion: one op from each peer per iteration
        # This allows concurrent execution across multiple peers
        work_handles = []
        for op_idx in range(max_ops):
            for peer_rank, ops in peer_ops_with_rank:
                if op_idx < len(ops):
                    _, p2p_op = ops[op_idx]
                    # Use the stream allocated to this peer to maintain ordering
                    stream_idx = peer_to_stream_idx[peer_rank]
                    stream = self._stream_pool[stream_idx]
                    with torch.cuda.stream(stream):
                        result = p2p_op.op(
                            p2p_op.tensor, p2p_op.peer, group=p2p_op.group
                        )
                        if p2p_op.op is dist.isend or p2p_op.op is dist.irecv:
                            work_handles.append(result)
                    total_ops += 1

        # Wait for all async operations to complete
        for work in work_handles:
            work.wait()

        return total_ops
