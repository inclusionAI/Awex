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
import threading
import time
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributed as dist

from awex import logging
from awex.converter.mcore_converter import get_mcore_model_parameters
from awex.meta.meta_resolver import (
    ParameterMeta,
)
from awex.meta.meta_server import MetaServerClient
from awex.meta.train_meta_resolver import McoreParamMetaResolver
from awex.models.registry import get_train_weights_converter
from awex.sharding.param_sharding import (
    get_rank_info_extractor,
)
from awex.util.common import (
    check_train_infer_params_meta,
    compute_statistics,
    from_binary,
    stripped_env_vars,
)
from awex.util.gpu import get_gpu_status
from awex.util.tensor_util import check_and_log_nan_values

logger = logging.getLogger(__name__)


class WeightExchangeWriter(ABC):
    def __init__(self, train_engine):
        self.train_engine = train_engine
        self.enable_debug_mode = train_engine.enable_debug_mode
        self.enable_colocate_mode = train_engine.enable_colocate_mode

    @abstractmethod
    def initialize(self, **kwargs):
        """Initialize the weight exchange writer."""
        pass

    @abstractmethod
    def write_weights(self, step_id, **kwargs):
        pass


class FileWeightExchangeWriter(WeightExchangeWriter):
    def __init__(self, train_engine):
        super().__init__(train_engine)

    def initialize(self, **kwargs):
        """Initialize file-based weight exchange writer (no-op)."""
        pass

    def write_weights(self, **kwargs):
        if self.enable_colocate_mode:
            self.train_engine.resume_memory_occupation(tags=["weights"])
        self.train_engine.save_hf_checkpoint(kwargs["path"])
        if self.enable_colocate_mode:
            self.train_engine.release_memory_occupation(tags=["weights"])


class WeightsExchangeShardingWriter(WeightExchangeWriter):
    def __init__(self, train_engine):
        super().__init__(train_engine)
        self.meta_server_addr = train_engine.meta_server_addr
        logger.info(f"Meta server address: {self.meta_server_addr}")
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.infer_conf = None
        self.infer_engine_config = None
        self.model = self.train_engine.model
        self.hf_config = self.train_engine.hf_config
        self.model_arch_name = self.hf_config.architectures[0]
        self.validated_steps = 0
        self.start_step = -1
        self.config = self.train_engine.config
        self.weights_validation_steps = self.config.get("weights_validation_steps", 0)
        self.validate_weights_every_n_steps = self.config.get(
            "validate_weights_every_n_steps", 1
        )
        self.dump_weights_for_validation = self.config.get(
            "dump_weights_for_validation", False
        )
        self.disable_pipeline = self.config.get(
            "disable_weights_exchange_pipeline", False
        )
        self.enable_nccl_debug_mode = self.config.get("debug_mode_config", {}).get(
            "enable_nccl_debug_mode", False
        )
        self.dump_weights_list_for_validation = self.config.get(
            "dump_weights_list_for_validation", []
        )
        self.dump_weights_dir_for_validation = self.config.get(
            "dump_weights_dir_for_validation", os.getcwd()
        )
        logger.info(f"Disable pipeline for weights writer: {self.disable_pipeline}")
        logger.info(f"Env variables for weights writer: {stripped_env_vars()}")
        self.lock = threading.Lock()
        self.timeout = 10000
        self.initialized = False
        self.num_infer_engines = None
        self.engine_name = train_engine.engine_name

    def initialize(self, **kwargs):
        pass

    def _initialize(self):
        rank = dist.get_rank()
        logger.info(f"Initializing weights exchange sharding writer for rank {rank}")
        # inference engine load model may take a long time, so we set a long timeout
        self.infer_conf = self.meta_server_client.get_object(
            "infer_conf", timeout=self.timeout
        )
        logger.info(f"Got inference config from meta server: {self.infer_conf}")
        self.infer_engine_config = self.infer_conf["infer_engine_config"]
        self.infer_world_size = self.infer_conf["infer_world_size"]
        self.rank_info = get_rank_info_extractor(self.engine_name)()
        logger.info(f"Writer rank info: {self.rank_info}")
        self.training_world_size = self.rank_info.world_size
        self.transfer_world_size = self.infer_world_size + self.training_world_size
        self.transfer_rank = self.infer_world_size + self.rank_info.global_rank
        logger.info(
            f"Writer transfer rank: {self.transfer_rank}, transfer world size: {self.transfer_world_size}"
        )
        self.parameter_meta_resolver = McoreParamMetaResolver(
            self.train_engine, self.train_engine.hf_config, self.infer_conf
        )
        self.parameters_meta = self.parameter_meta_resolver.get_parameters_meta()
        logger.info(
            "Finished querying and building parameters meta from all training workers"
        )
        if rank == 0:
            self.meta_server_client.put_object(
                "training_params_meta", self.parameters_meta
            )
            logger.info("Put training parameters meta to meta server")
        new_meta = [
            p.to_local_parameter_meta(self.rank_info.global_rank)
            for p in self.parameters_meta
        ]
        self.total_local_num_elements = sum(
            shard.numel for p in new_meta for shard in p.shards
        )
        self.total_local_param_size = sum(
            shard.numel * shard.dtype.itemsize for p in new_meta for shard in p.shards
        )
        logger.info(
            f"[Writer {self.transfer_rank}] Total local number of elements: {self.total_local_num_elements}, "
            f"total local parameter size: {self.total_local_param_size}"
        )
        self.current_worker_parameters_meta = new_meta
        self.current_worker_parameters_map = {
            parameter_meta.name: parameter_meta for parameter_meta in new_meta
        }
        self._history_write_weights_time = {}
        logger.info(
            f"Start to get inference parameters meta from meta server for rank {dist.get_rank()}"
        )
        if rank == 0:
            infer_params_meta_binary = self.meta_server_client.get_binary(
                "infer_params_meta", timeout=self.timeout
            )
            dist.broadcast_object_list([infer_params_meta_binary], src=0)
        else:
            result = [None]
            dist.broadcast_object_list(result, src=0)
            infer_params_meta_binary = result[0]
        self.infer_params_meta: List[ParameterMeta] = from_binary(
            infer_params_meta_binary
        )
        logger.info("Finished getting inference parameters meta from meta server")
        check_train_infer_params_meta(
            self.parameters_meta,
            self.infer_params_meta,
            raise_exception=not self.enable_debug_mode,
        )
        self.weight_converter = get_train_weights_converter(
            self.train_engine.engine_name,
            self.model_arch_name,
            self.hf_config,
            self.rank_info,
            self.infer_conf,
        )
        logger.info("Start to get number of inference engines from meta server")
        self.num_infer_engines = self.meta_server_client.get_object(
            "num_infer_engines", timeout=self.timeout
        )
        logger.info("Finished getting number of inference engines from meta server")
        self.infer_instance_world_size = self.infer_params_meta[0].shards[0].world_size
        logger.info(
            f"Finished building parameters for weights writer for rank {dist.get_rank()}"
        )

    @torch.no_grad()
    def convert_parameters(self):
        # for megatron vpp, model is a list of modules
        parameters = [
            (name, param.detach())
            for model in self.model
            for name, param in get_mcore_model_parameters(model).items()
        ]
        logger.info(f"[Writer {self.transfer_rank}] Start to convert parameters")
        converted = {}
        for name, param in parameters:
            for hf_name, hf_param in self.weight_converter.convert_param(name, param):
                converted[hf_name] = hf_param
        logger.info(f"[Writer {self.transfer_rank}] Finished converting parameters")
        return converted

    @torch.no_grad()
    def write_weights(self, step_id, **kwargs):
        with self.lock:
            logger.info(
                f"Start to write weights for step {step_id}, current thread {threading.current_thread()}"
            )
            try:
                if self.enable_colocate_mode:
                    self._release_memory_for_weights_exchange()
                if not self.initialized:
                    logger.info("Start to initialize weights exchange sharding writer")
                    self._initialize()
                    self.initialized = True
                    logger.info(
                        "Finished initializing weights exchange sharding writer"
                    )
                self._validate_weights(step_id, **kwargs)
                start_time = time.time()
                if self.enable_colocate_mode:
                    self.train_engine.resume_memory_occupation(tags=["weights"])
                    self._write_weights_in_colocate_mode(step_id, **kwargs)
                else:
                    self._write_weights(step_id, **kwargs)
                self._finish_weights_update()
                duration = time.time() - start_time
                compute_statistics(
                    self._history_write_weights_time, step_id, duration, "Write weights"
                )
            except Exception as e:
                logger.exception(f"Error in write_weights: {e}")
                raise e

    def _release_memory_for_weights_exchange(self):
        if self.num_infer_engines is None:
            logger.info("Start to get number of inference engines from meta server")
            # first time, the inference engine is not initialized, so we need to wait for it to be initialized
            self.num_infer_engines = self.meta_server_client.get_object(
                "num_infer_engines", timeout=self.timeout
            )
            logger.info("Start to wait for all inference engines to be initialized")
            self.meta_server_client.wait_set_until_size(
                "num_inited_inference_engines",
                self.num_infer_engines,
                timeout=self.timeout,
            )
            logger.info(
                "All inference engines have been initialized, start to resume weights memory occupation"
            )
        self.train_engine.release_memory_occupation(tags=["optimizer"])
        self.train_engine.resume_memory_occupation(tags=["weights"])
        dist.barrier()
        if dist.get_rank() == 0:
            self.meta_server_client.add_object_to_set(
                "all_training_offloaded_optimizers", dist.get_rank()
            )

    def _finish_weights_update(self):
        if not self.enable_colocate_mode:
            return
        logger.info("Waiting for all inference engines to finish weights update")
        self.meta_server_client.wait_set_until_size(
            "finished_weights_update_engines",
            self.num_infer_engines,
            timeout=self.timeout,
        )
        logger.info(
            "All inference engines have finished weights update, start to release weights memory occupation"
        )
        dist.barrier()
        if dist.get_rank() == 0:
            self.meta_server_client.delete_if_exists("finished_weights_update_engines")
            self.meta_server_client.delete_if_exists(
                "all_training_offloaded_optimizers"
            )
            self.meta_server_client.delete_if_exists("all_training_offloaded_weights")
        logger.info("Finished releasing weights memory occupation")

    def _write_weights(self, step_id, **kwargs):
        logger.info(f"Writing weights for step {step_id}")
        parameters = [
            (name, param)
            for model in self.model
            for name, param in get_mcore_model_parameters(model).items()
        ]
        logger.info(f"GPU status before write weights:\n{get_gpu_status()}")
        for name, param in parameters:
            temp_parameters = self.weight_converter.convert_param(name, param)
            temp_parameters = dict(temp_parameters)
            tensor_pairs = []
            for name, parameter in temp_parameters.items():
                if name not in self.current_worker_parameters_map:
                    raise ValueError(
                        f"Parameter {name} not found in current worker parameters map"
                    )
                param_meta = self.current_worker_parameters_map[name]
                assert len(param_meta.shards) == 1
                tensor_pairs.append(
                    (
                        name,
                        parameter,
                        param_meta.shards[0],
                        param_meta,
                    )
                )
            self.write_tensors(step_id, tensor_pairs, **kwargs)
            temp_parameters.clear()
        del parameters
        gc.collect()
        logger.info(f"GPU status after write weights:\n{get_gpu_status()}")
        if self.enable_colocate_mode:
            self.train_engine.release_memory_occupation("weights")
            self.meta_server_client.add_object_to_set(
                "all_training_offloaded_weights", self.transfer_rank
            )
        self.finish_step(step_id)
        logger.info(f"Finished writing weights for step {step_id}")

    def _write_weights_in_colocate_mode(self, step_id, **kwargs):
        logger.info(
            f"Start to write weights in colocate mode for rank {self.transfer_rank}"
        )
        self._write_weights(step_id, **kwargs)
        logger.info(
            f"Finished writing weights in colocate mode for rank {self.transfer_rank}"
        )

    def _validate_weights(self, step_id, **kwargs):
        if self.validated_steps == 0:
            self.start_step = step_id
        if self.validated_steps >= self.weights_validation_steps:
            return
        if (step_id - self.start_step) % self.validate_weights_every_n_steps != 0:
            return
        self.validated_steps += 1
        for model in self.model:
            for name, parameter in model.named_parameters():
                if name in self.dump_weights_list_for_validation:
                    # save to file
                    abs_path = os.path.join(
                        self.dump_weights_dir_for_validation,
                        f"writer_{os.getpid()}_native_{name}.{step_id}.pt",
                    )
                    torch.save(parameter.detach().cpu(), abs_path)
                    logger.info(
                        f"[Writer] Saved parameter(native) {name} to {abs_path}"
                    )
        parameters = self.convert_parameters()
        for name, parameter in parameters.items():
            if name in self.dump_weights_list_for_validation:
                # save to file
                abs_path = os.path.join(
                    self.dump_weights_dir_for_validation,
                    f"writer_{os.getpid()}_converted_{name}.{step_id}.pt",
                )
                torch.save(parameter.detach().cpu(), abs_path)
                logger.info(f"[Writer] Saved parameter(converted) {name} to {abs_path}")
        model_path = kwargs["path"]
        logger.info(
            f"Start to write weights for step {step_id} to {model_path} for weights validation"
        )
        for name, parameter in parameters.items():
            check_and_log_nan_values(parameter, name, stage_info=" before validation")
            if self.dump_weights_for_validation:
                logger.info(
                    f"Parameter {name} with shape {parameter.shape}: {parameter}"
                )
        self.train_engine.save_hf_checkpoint(model_path)
        dist.barrier()
        logger.info(
            f"Validation weights Barrier passed for weights writer for step {step_id}, all weights are written to disk"
        )
        if dist.get_rank() == 0:
            load_key = "weights_ready_for_load"
            self.meta_server_client.put_object(load_key, True)
            start_time = time.time()
            last_log_time = time.time()
            send_key = "weights_ready_for_send"
            ready_engines = self.meta_server_client.get_object(
                send_key, default_value=set()
            )
            while len(ready_engines) != self.num_infer_engines:
                current_time = time.time()
                if current_time - last_log_time >= 10:
                    logger.info(
                        f"Waiting {current_time - start_time} seconds for {self.num_infer_engines} inference engine instances "
                        f"to read weights for step {step_id}, ready_engines: {ready_engines}"
                    )
                    last_log_time = current_time
                ready_engines = self.meta_server_client.get_object(
                    send_key, default_value=set()
                )
                time.sleep(0.5)
            self.meta_server_client.delete_if_exists(send_key)
            self.meta_server_client.delete_if_exists(load_key)
            dist.barrier()
        else:
            dist.barrier()
        logger.info(
            f"All inference engine instances has read weights for step {step_id} from {model_path}"
        )

    def finish_step(self, step_id):
        pass

    def write_tensors(self, step_id, tensor_pairs: List, **kwargs):
        pass


def get_weights_exchange_writer(train_engine) -> WeightExchangeWriter:
    if train_engine.comm_backend == "file":
        return FileWeightExchangeWriter(train_engine)
    elif train_engine.comm_backend == "nccl":
        from awex.writer.nccl_writer import NCCLWeightsWriter

        return NCCLWeightsWriter(train_engine)
    elif train_engine.comm_backend == "astate":
        from awex.writer.astate_writer import AStateWeightsWriter

        return AStateWeightsWriter(train_engine)
    raise ValueError(
        f"Unsupported weights exchange comm backend: {train_engine.comm_backend}"
    )
