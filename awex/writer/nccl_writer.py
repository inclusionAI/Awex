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

import gc
import os
import time

import torch
import torch.distributed as dist

from awex import logging
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from awex.util import device as device_util
from awex.util.common import compute_statistics, get_ip_address
from awex.util.gpu import print_current_gpu_status
from awex.util.process_group import init_weights_update_group, setup_batch_isend_irecv
from awex.util.system_util import count_open_fds
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
    ipc_serialize,
    release_tensors,
)
from awex.writer.weights_writer import WeightsExchangeShardingWriter

logger = logging.getLogger(__name__)


class NCCLWeightsWriter(WeightsExchangeShardingWriter):
    def _initialize(self):
        super()._initialize()
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )
        if self.enable_colocate_mode:
            self._init_writer_in_colocate_mode()
            return
        logger.info(f"Start to build transfer plan for rank {self.transfer_rank}")
        self.transfer_plan = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_infer_engines,
            self.enable_debug_mode,
        ).build_local_transfer_plan(
            self.infer_params_meta,
            self.parameters_meta,
            self.transfer_rank,
        )
        inter_hash = compute_transfer_plan_hash(self.transfer_plan)
        logger.info(
            "Writer rank %s inter plan hash: %s",
            self.transfer_rank,
            inter_hash,
        )
        logger.info(
            "Writer rank %s transfer plan stats: %s",
            self.transfer_rank,
            compute_transfer_plan_stats(self.transfer_plan),
        )
        self.recv_ranks = list(self.transfer_plan.operations.keys())
        self.required_param_names = {
            op.send_shard_meta.name
            for ops in self.transfer_plan.operations.values()
            for op in ops
        }
        logger.info(
            f"Writer rank {self.transfer_rank}: Built transfer plan to send to ranks: {self.recv_ranks}"
        )
        # FIXME (Derek): this might print too many params?
        logger.info(
            "Writer rank %s: required converted params for this plan: %s",
            self.transfer_rank,
            len(self.required_param_names),
        )
        logger.info(
            f"Writer rank {self.transfer_rank}: Operations per rank: {[(rank, len(ops)) for rank, ops in self.transfer_plan.operations.items()]}"
        )
        self.recv_ranks_sample = (
            self.recv_ranks[:8] + ["..."] + self.recv_ranks[-8:]
            if len(self.recv_ranks) > 16
            else self.recv_ranks
        )
        self.num_to_sends = sum(
            len(operations) for operations in self.transfer_plan.operations.values()
        )
        logger.info(f"Finished building transfer plan for rank {self.transfer_rank}")
        logger.info(
            f"Start to get master info from meta server for rank {self.transfer_rank}"
        )
        master_info = self.meta_server_client.get_object(
            "master_info", timeout=self.timeout
        )
        master_address, master_port = master_info
        logger.info(
            f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
        )
        self._set_device()
        self.weights_update_group = init_weights_update_group(
            master_address,
            master_port,
            self.transfer_rank,
            self.transfer_world_size,
            "weights_exchange",
            backend=self.comm_backend,
            role="train",
        )
        logger.info(f"Initialized NCCL weights writer for rank {self.transfer_rank}")
        # Add a barrier to ensure all processes are ready
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        logger.info(f"Barrier passed for weights writer with rank {self.transfer_rank}")
        if self.transfer_rank == self.transfer_world_size - 1:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.send(
                torch.tensor(1, device=device_util.get_torch_device()),
                dst=0,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: send tensor to rank {self.transfer_world_size - 1} from rank {self.transfer_rank}"
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.transfer_world_size
        )
        logger.info(
            f"Finished initializing NCCL weights writer for rank {self.transfer_rank}"
        )

    def _set_device(self):
        device = device_util.current_device()
        count = device_util.device_count() or 1
        gpu_id = int(os.environ.get("DEVICE", device)) % count
        logger.info(
            f"[NCCLWeightsWriter] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, "
            f"previous device is {device}, device_count is {count}, "
            f"{'/'.join(device_util.visible_devices_env_names())} env is {device_util.visible_devices_env_value() or '(unset)'}"
        )
        device_util.set_device(gpu_id)

    def _init_writer_in_colocate_mode(self):
        self.ipc_backend = self.asystem_train_config.get(
            "weights_exchange_ipc_backend", "cuda"
        )
        # Don't get IPC tensors here since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        ip_address = get_ip_address()
        self._set_device()
        device_id = device_util.current_device()
        self.meta_server_client.add_object_to_set(
            "training_device_rank_entries", (ip_address, device_id, self.transfer_rank)
        )
        logger.info(
            f"Initialized NCCL weights writer for rank {self.transfer_rank} in colocate mode"
        )

    @torch.no_grad()
    def _write_weights(self, step_id, **kwargs):
        """
        Asynchronously send weights to inference ranks using torch.distributed.isend.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to send
        2. We start isend operations in parallel to all sender ranks
        3. When a send completes, we immediately start the next send to that rank
        4. We continue until all operations to all sender ranks are completed

        Args:
            step_id: The training step ID used as communication tag
            **kwargs: Additional keyword arguments (unused)
        """
        rank_coordinate = self.transfer_rank
        logger.info(
            f"Start to send weights using NCCL to {len(self.transfer_plan.operations)} "
            f"ranks({self.recv_ranks_sample}) from rank {rank_coordinate} "
            f"with {self.num_to_sends} sends"
        )
        start_time = time.time()
        parameters = None
        p2p_op_list = None
        try:
            if self.enable_mem_debug:
                print_current_gpu_status(f"writer-{self.transfer_rank} before convert")
            parameters = self.convert_parameters(
                required_names=self.required_param_names
            )
            logger.info("Writer: Converting parameters completed, building send ops")
            if self.enable_mem_debug:
                print_current_gpu_status(f"writer-{self.transfer_rank} after convert")
            p2p_op_list, _ = nccl_build_send_ops(
                parameters, self.transfer_plan, self.weights_update_group, -1
            )
            logger.info(
                f"Writer: Built {len(p2p_op_list)} send operations to "
                f"{len(self.transfer_plan.operations)} ranks"
            )
            if self.enable_mem_debug:
                total_send_bytes = sum(
                    int(op.tensor.numel()) * int(op.tensor.element_size())
                    for op in p2p_op_list
                )
                logger.info(
                    "[Writer %s][MEM] built send ops: count=%s total_send_bytes=%s",
                    self.transfer_rank,
                    len(p2p_op_list),
                    total_send_bytes,
                )
                print_current_gpu_status(
                    f"writer-{self.transfer_rank} after build_send_ops"
                )

            # Execute all sends via batch_send_recv to get consistent interleaving
            # and per-peer stream assignment without relying directly on
            # batch_isend_irecv.
            logger.info(
                f"Writer: Executing {len(p2p_op_list)} send ops via batch_send_recv"
            )
            batch_send_recv(
                send_ops=p2p_op_list, recv_ops=[], blocking=True, use_group=True
            )
            device_util.synchronize(device_id=device_util.current_device())
            if self.enable_mem_debug:
                print_current_gpu_status(f"writer-{self.transfer_rank} after send")
            duration = time.time() - start_time
            logger.info(
                f"Finished sending weights for step {step_id} using NCCL to {len(self.transfer_plan.operations)} ranks({self.recv_ranks_sample}) "
                f"from rank {rank_coordinate} with {self.num_to_sends} sends, took {duration:.4f} seconds"
            )
            compute_statistics(
                self._history_write_weights_time,
                step_id,
                duration,
                "Send weights using NCCL",
            )
            dist.barrier(
                group=self.weights_update_group,
                device_ids=[device_util.current_device()],
            )
            logger.info(
                f"Barrier passed for writer step {step_id} with rank {self.transfer_rank}"
            )
        finally:
            # Explicitly release temporary converted tensors after each step
            # to avoid carrying peak memory into the next training iteration.
            if p2p_op_list is not None:
                p2p_op_list.clear()
            if parameters is not None:
                parameters.clear()
            gc.collect()
            if self.enable_mem_debug:
                if device_util.get_device_type() == "cuda":
                    torch.cuda.empty_cache()
                elif device_util.get_device_type() == "npu" and hasattr(torch, "npu"):
                    torch.npu.empty_cache()
                print_current_gpu_status(f"writer-{self.transfer_rank} after cleanup")

    @torch.no_grad()
    def _prepare_params_for_colocate(self):
        logger.info(
            f"Start to write weights in colocate mode for rank {self.transfer_rank}"
        )
        self.train_engine.release_grad_memory()
        converted = self.convert_parameters()
        tensors, names = [], []
        for name, tensor in converted.items():
            assert not tensor.requires_grad
            tensors.append(tensor)
            names.append(name)
        return tensors, names

    @torch.no_grad()
    def _write_weights_in_colocate_mode(self, step_id, **kwargs):
        start_time = time.time()
        tensors, names = self._prepare_params_for_colocate()
        num_tensors = len(tensors)
        if self.ipc_backend in ("cpu", "npu"):
            tensors = [t.cpu() for t in tensors]
        logger.info(
            f"Start to group tensors by shape and dtype for rank {self.transfer_rank}"
        )
        # this will copy tensor by concatenate
        group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        device_util.synchronize(device_id=device_util.current_device())
        logger.info(
            f"Finished grouping tensors by shape and dtype for rank {self.transfer_rank}"
        )
        print_current_gpu_status(
            f"after group_tensors_by_shape_and_dtype for rank {self.transfer_rank}"
        )
        logger.info(f"Open fds before serialize: {count_open_fds()}")

        release_tensors(tensors)
        del tensors
        self.train_engine.release_memory_occupation("weights")
        self.meta_server_client.add_object_to_set(
            "all_training_offloaded_weights", self.transfer_rank
        )
        print_current_gpu_status(
            f"after offloaded weights for rank {self.transfer_rank}"
        )

        if self.ipc_backend in ("cpu", "npu"):
            group_shared = [tensor.cpu().share_memory_() for tensor in group_tensors]
            serialized_weights = ipc_serialize((group_shared, metadata, names))
        else:
            group_shared = [
                tensor.to(device_util.get_torch_device()).share_memory_()
                for tensor in group_tensors
            ]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        device_util.synchronize(device_id=device_util.current_device())
        logger.info(
            f"Finished serializing ipc weights with {num_tensors} params, and {len(group_shared)} groups "
            f"for rank {self.transfer_rank}"
        )
        logger.info(f"Open fds after serialize: {count_open_fds()}")

        # Put serialized weights to meta server
        ip_address = get_ip_address()
        device_id = device_util.current_device()
        key_suffix = f"_{ip_address}_{device_id}_{step_id}"
        serialized_weights_key = f"training_serialized_weights{key_suffix}"
        self.meta_server_client.put_object(
            serialized_weights_key,
            (self.transfer_rank, self.rank_info, serialized_weights),
        )
        logger.info(
            f"Put {len(group_shared)} serialized training weights to meta server "
            f"with key {serialized_weights_key} for step {step_id}"
        )
        # Wait for inference engines to finish processing
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.get_object(update_finished_key, timeout=self.timeout)
        self.meta_server_client.delete_if_exists(update_finished_key)
        release_tensors(group_tensors)
        release_tensors(group_shared)
        del group_tensors
        del group_shared
        device_util.synchronize(device_id=device_util.current_device())
        gc.collect()
        if device_util.get_device_type() == "cuda":
            torch.cuda.empty_cache()
        print_current_gpu_status(
            f"after clear group_shared for rank {self.transfer_rank}"
        )
        write_finished_key = f"write_finished{key_suffix}"
        self.meta_server_client.put_object(write_finished_key, True)
        duration = time.time() - start_time
        compute_statistics(
            self._history_write_weights_time,
            step_id,
            duration,
            "Send weights using NCCL in colocate mode",
        )
        logger.info(
            f"Finished writing weights in colocate mode for rank {self.transfer_rank}"
        )
