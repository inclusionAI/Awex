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
from awex.reader.weights_reader import WorkerWeightsReader
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.util.common import (
    compute_statistics,
    get_free_port,
    get_ip_address,
)
from awex.util.gpu import get_gpu_status, print_current_gpu_status
from awex.util.system_util import count_open_fds
from awex.util.tensor_util import (
    cuda_ipc_deserialize,
    ipc_deserialize,
    reconstruct_tensors_from_groups,
)

logger = logging.getLogger(__name__)


class NCCLWorkerWeightsReader(WorkerWeightsReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_plan = None
        self.weights_update_group = None
        self.send_ranks = None
        self.send_ranks_sample = None
        self.num_to_recvs = None
        self.rank_coordinate = None

    def initialize(self):
        super().initialize()
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.transfer_rank,
        )
        if self.transfer_rank == 0:
            master_address = get_ip_address()
            master_port = get_free_port()
            master_info = (master_address, master_port)
            self.meta_server_client.put_object("master_info", master_info)
            logger.info(
                f"Put master info to meta server for rank {self.transfer_rank}: {master_info}"
            )
        else:
            master_info = self.meta_server_client.get_object(
                "master_info", timeout=self.timeout
            )
            master_address, master_port = master_info
            logger.info(
                f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
            )
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )

        from awex.util.process_group import (
            init_weights_update_group,
            setup_batch_isend_irecv,
        )

        gpu_id = self.scheduler.gpu_id
        logger.info(
            f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, "
            f"device env is {os.environ.get('DEVICE')}, "
            f"previous device is {torch.cuda.current_device()}, "
            f"device_count is {torch.cuda.device_count()}, "
            f"CUDA_VISIBLE_DEVICES env is {os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        torch.cuda.set_device(gpu_id)
        world_size = (
            self.infer_world_size
            if self.enable_colocate_mode
            else self.transfer_world_size
        )
        self.weights_update_group = init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank=self.transfer_rank,
            world_size=world_size,
            group_name="weights_exchange",
            role="inference",
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank}, engine rank {self.engine_rank}"
        )
        # Add a barrier to ensure all processes are ready
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(f"Barrier passed for weights reader with rank {self.transfer_rank}")
        if self.transfer_rank == 0:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.recv(
                torch.tensor(1).cuda(),
                src=world_size - 1,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: recv tensor from rank 0 for rank {self.transfer_rank}"
            )
        if (
            self.enable_colocate_mode
            and self.transfer_rank == self.infer_world_size - 1
        ):
            dist.send(
                torch.tensor(1).cuda(),
                dst=0,
                group=self.weights_update_group,
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, world_size
        )
        self.send_ranks = list(self.transfer_plan.operations.keys())
        self.send_ranks_sample = (
            self.send_ranks[:8] + ["..."] + self.send_ranks[-8:]
            if len(self.send_ranks) > 16
            else self.send_ranks
        )
        self.num_to_recvs = sum(
            len(operations) for operations in self.transfer_plan.operations.values()
        )
        self.rank_coordinate = (
            f"{self.engine_rank}-{self.rank_info.global_rank}-{self.transfer_rank}"
        )
        if self.enable_colocate_mode:
            self._init_reader_in_colocate_mode()
        self.deserialized_weights = {}
        logger.info(
            f"Created NCCL weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def _init_reader_in_colocate_mode(self):
        self.meta_server_client.add_object_to_set(
            "inference_device_rank_entries",
            (get_ip_address(), torch.cuda.current_device(), self.transfer_rank),
        )
        self.meta_server_client.wait_set_until_size(
            "inference_device_rank_entries", self.infer_world_size, timeout=self.timeout
        )
        self.inference_device_mapping = self.meta_server_client.get_set(
            "inference_device_rank_entries"
        )
        self.inference_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in self.inference_device_mapping
        }

        self.meta_server_client.wait_set_until_size(
            "training_device_rank_entries",
            self.training_world_size,
            timeout=self.timeout,
        )
        device_rank_entries = self.meta_server_client.get_set(
            "training_device_rank_entries"
        )
        self.training_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in device_rank_entries
        }
        self.train_to_infer_device_mapping = {}
        self.infer_to_train_device_mapping = {}
        for ip_address, device_id, transfer_rank in device_rank_entries:
            infer_rank = self.inference_device_mapping[(ip_address, device_id)]
            self.train_to_infer_device_mapping[transfer_rank] = infer_rank
            self.infer_to_train_device_mapping[infer_rank] = transfer_rank
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.send_transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.infer_to_train_device_mapping[self.transfer_rank],
        )
        from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport

        self.colocate_transport = NcclColocateStreamBatchTransport(
            self.transfer_rank, self.infer_world_size
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank} in colocate mode"
        )

    def pre_update_weights(self, step_id, **kwargs):
        pass

    def collect_training_weights(self, step_id, **kwargs):
        if not self.enable_colocate_mode:
            return
        # Can't serialize IPC tensors at initialization since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        # We'll get serialized weights from meta server each step instead
        # Get serialized weights from meta server
        ip_address = get_ip_address()
        device_id = torch.cuda.current_device()
        key = f"training_serialized_weights_{ip_address}_{device_id}_{step_id}"
        logger.info(
            f"Start to get serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        self.send_rank, self.send_rank_info, serialized_weights = (
            self.meta_server_client.get_object(key, timeout=self.timeout)
        )
        logger.info(
            f"Finished getting serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        logger.info(
            f"GPU status before deserialization:\n{get_gpu_status()} for rank {self.rank_coordinate}"
        )
        logger.info(f"Open fds before deserialization: {count_open_fds()}")
        # Deserialize weights into tensors
        if self.ipc_backend == "cpu":
            group_shared, metadata, names = ipc_deserialize(serialized_weights)
            group_shared = [t.to(device_id) for t in group_shared]
        else:
            group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        torch.cuda.synchronize(device=torch.cuda.current_device())
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        torch.cuda.synchronize(device=torch.cuda.current_device())
        self.deserialized_weights = dict(zip(names, tensors))
        logger.info(
            f"Deserialized {len(self.deserialized_weights)} parameters and {len(group_shared)} groups"
        )
        logger.info(
            f"GPU status after deserialization for rank {self.rank_coordinate}:\n{get_gpu_status()}"
        )
        logger.info(f"Open fds after deserialization: {count_open_fds()}")

    def _update_weights(self, step_id, **kwargs):
        """
        Asynchronously receive weights from training ranks using torch.distributed.irecv.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to receive
        2. We start irecv operations in parallel from all sender ranks
        3. When a receive completes, we immediately start the next receive from that rank
        4. We continue until all operations from all sender ranks are completed

        Args:
            step_id: The training step ID
            **kwargs: Additional keyword arguments (unused)
        """
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from "
            f"{len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) "
            f"for rank {self.rank_coordinate}."
        )
        start_time = time.time()

        # Build receive ops once for logging, then execute them via
        # batch_send_recv to keep scheduling consistent with the writer.
        p2p_op_list = nccl_build_recv_ops(
            self.parameters, self.transfer_plan, self.weights_update_group
        )
        logger.info(
            f"Reader: Built {len(p2p_op_list)} recv operations from "
            f"{len(self.transfer_plan.operations)} training ranks"
        )

        logger.info(
            f"Reader: Executing {len(p2p_op_list)} recv ops via batch_send_recv"
        )
        batch_send_recv(
            send_ops=[], recv_ops=p2p_op_list, blocking=True, use_group=True
        )
        torch.cuda.synchronize(device=torch.cuda.current_device())
        duration = time.time() - start_time
        logger.info(
            f"Finished receiving weights for step {step_id} using NCCL "
            f"from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) "
            f"to rank {self.rank_coordinate} with {self.num_to_recvs} receives, took {duration:.4f} seconds"
        )
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )

    def _update_weights_in_colocate_mode(self, step_id, **kwargs):
        assert self.enable_colocate_mode, "Colocate mode is not enabled"
        self.collect_training_weights(step_id, **kwargs)
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from {len(self.transfer_plan.operations)} "
            f"ranks({self.send_ranks_sample}) for rank {self.rank_coordinate}."
        )
        start_time = time.time()
        self.colocate_transport.update_weights_in_colocate_mode(
            self.train_to_infer_device_mapping,
            self.infer_to_train_device_mapping,
            self.transfer_rank,
            self.rank_coordinate,
            self.infer_world_size,
            self.send_transfer_plan,
            self.transfer_plan,
            self.weights_update_group,
            self.deserialized_weights,
            self.parameters,
            step_id=step_id,
        )
        print_current_gpu_status(
            f"after weights update using NCCL for rank {self.rank_coordinate}"
        )
        self.deserialized_weights = None
        duration = time.time() - start_time
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        ip_address = get_ip_address()
        device_id = torch.cuda.current_device()
        key_suffix = f"_{ip_address}_{device_id}_{step_id}"
        # Signal completion to training process
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.put_object(update_finished_key, True)
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )
        gc.collect()
        torch.cuda.empty_cache()
        write_finished_key = f"write_finished{key_suffix}"
        self.meta_server_client.get_object_then_delete(write_finished_key)
        logger.info(
            f"Finished updating weights in colocate mode for rank {self.transfer_rank}"
        )
