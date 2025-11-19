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

import argparse
import copy
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch
import torch.distributed as dist

from awex import logging
from awex.engine.mcore import MegatronEngine
from awex.engine.sglang import SGLangEngine
from awex.meta.meta_server import start_meta_server
from awex.tests.test_utils import megatron_model_from_hf

logger = logging.getLogger(__name__)


enable_debug_mode = False

tp_size = 1
lite_inference_config = {
    "model_path": "Qwen/Qwen2-1.5B",
    "tokenizer_path": ".models/moe_lite",
    "tokenizer_mode": "auto",
    "skip_tokenizer_init": True,
    "load_format": "auto",
    "trust_remote_code": True,
    "kv_cache_dtype": "auto",
    "device": "cuda",
    "served_model_name": "moe_lite",
    "mem_fraction_static": 0.5,
    "tp_size": tp_size,
    "pp_size": 1,
    "random_seed": 47,
    "log_level": "info",
    "dp_size": 1,
    "ep_size": 1,
    "attention_backend": "torch_native",
    "num_engines": 1,
    "engine_rank": 0,
    "comm_backend": "nccl",
    "enable_debug_mode": enable_debug_mode,
    "disable_shared_experts_fusion": True,
}


# Define parameter verification configurations
params_sharding_config = [
    # Word embeddings - no tensor parallelism
    {
        "name": "model.word_embeddings.weight",
        "tp_sharding": False,
        "sharding_dim": None,
    },
    # Attention weights - tensor parallelism on specific dimensions
    {
        "name": "model.layers.0.attention.query_key_value.weight",
        "tp_sharding": True,
        "sharding_dim": 0,
    },
    {
        "name": "model.layers.0.attention.dense.weight",
        "tp_sharding": True,
        "sharding_dim": 1,
    },
    # MLP expert weights - tensor parallelism on first dimension
    {
        "name": "model.layers.0.mlp.experts.0.gate_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 0,
    },
    {
        "name": "model.layers.0.mlp.experts.0.up_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 0,
    },
    {
        "name": "model.layers.0.mlp.experts.0.down_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 1,
    },
    {
        "name": "model.layers.1.mlp.experts.63.gate_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 0,
    },
    {
        "name": "model.layers.1.mlp.experts.63.down_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 1,
    },
    # Shared expert weights - tensor parallelism on second dimension
    {
        "name": "model.layers.3.mlp.shared_experts.up_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 0,
    },
    {
        "name": "model.layers.3.mlp.shared_experts.down_proj.weight",
        "tp_sharding": True,
        "sharding_dim": 1,
    },
    # Layer norm weights - tensor parallelism on first dimension
    {
        "name": "model.layers.1.input_layernorm.weight",
        "tp_sharding": False,
        "sharding_dim": None,
    },
    {
        "name": "model.layers.3.post_attention_layernorm.weight",
        "tp_sharding": False,
        "sharding_dim": None,
    },
    {"name": "model.norm.weight", "tp_sharding": False, "sharding_dim": None},
    # Output layer weights - tensor parallelism on second dimension
    {"name": "lm_head.weight", "tp_sharding": True, "sharding_dim": 0},
]


class WeightsExchangeIT:
    def __init__(
        self,
        inference_config=None,
        comm_backend=None,
    ):
        self.comm_backend = comm_backend
        ip, port = start_meta_server()
        self.meta_server_addr = f"{ip}:{port}"
        self.inference_config = inference_config or copy.deepcopy(lite_inference_config)
        self.inference_config["comm_backend"] = comm_backend
        self.inference_config["meta_server_addr"] = self.meta_server_addr
        self.train_config = {
            "meta_server_addr": self.meta_server_addr,
            "comm_backend": comm_backend,
            "enable_debug_mode": enable_debug_mode,
        }
        self.sgl_engine = None
        self.sglang_engine = None
        self.megatron_engine = None
        self.executor = ThreadPoolExecutor(max_workers=4)

    def initialize(self):
        self._init_sglang_engine()
        self._init_megatron_engine()

    def destroy(self):
        self.sgl_engine.shutdown()

    def _init_sglang_engine(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(tp_size)))
        import sglang as sgl

        from awex.engine.sglang import extract_sgl_config

        sgl_engine = sgl.Engine(
            **extract_sgl_config(self.inference_config), random_seed=42
        )
        self.sgl_engine = sgl_engine
        self.sglang_engine = SGLangEngine(self.inference_config, sgl_engine)
        self.sglang_engine.initialize()
        os.environ.pop("CUDA_VISIBLE_DEVICES")
        logger.info("SGLang backend initialized")

    def _init_megatron_engine(self):
        self.train_config["tensor_model_parallel_size"] = 1
        self.train_config["pipeline_model_parallel_size"] = 1
        self.train_config["expert_model_parallel_size"] = 1
        self.train_config["num_layers"] = 4

        os.environ["RANK"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(tp_size + 1)))
        os.environ["DEVICE"] = f"{tp_size}"
        os.environ["LOCAL_RANK"] = f"{tp_size}"
        os.environ["MASTER_PORT"] = "17443"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["WORLD_SIZE"] = "1"

        self.mcore_model, self.mcore_hf_config = self.setup_megatron()
        self.megatron_engine = MegatronEngine(
            self.train_config, self.mcore_hf_config, self.mcore_model
        )
        self.megatron_engine.initialize()
        logger.info("Megatron backend initialized")

    def setup_megatron(self):
        """Setup megatron"""
        # TODO mpu.initialize_model_parallel
        # TODO initialize_rerun_state_machine
        # TODO load model and create optimizer
        model, hf_config = megatron_model_from_hf()
        return model, hf_config

    def exchange_weights(self):
        if self.megatron_engine is None:
            raise RuntimeError("Megatron backend not initialized")
        if self.sglang_engine is None:
            raise RuntimeError("SGLang backend not initialized")

        if self.comm_backend == "file":
            temp_ctx = tempfile.TemporaryDirectory()
            path = os.path.join(temp_ctx.name, "checkpoint")
        else:
            temp_ctx = nullcontext()
            path = None

        with temp_ctx:
            future = self.executor.submit(
                self.sglang_engine.update_weights,
                **({} if path is None else {"path": path}),
            )
            time.sleep(1)
            self.megatron_engine.write_weights(
                **({} if path is None else {"path": path})
            )
            future.result()
        logger.info("Update weights finished")

    def verify_updated_parameters(self):
        """Verify that parameters have been correctly updated across all tensor parallel workers."""
        assert self.megatron_engine is not None, "Megatron backend not initialized"
        assert self.sglang_engine is not None, "SGLang backend not initialized"
        assert self.megatron_engine.weights_exchange_writer is not None, (
            "Weights exchange writer not initialized"
        )
        parameters = self.megatron_engine.weights_exchange_writer.convert_parameters()
        length = 10000
        tp_size = self.inference_config["tp_size"]
        # Verify each parameter
        for config in params_sharding_config:
            self._verify_single_parameter(parameters, config, length, tp_size)

    def _verify_single_parameter(self, parameters, config, length, tp_size):
        """Verify a single parameter across all tensor parallel workers."""
        param_name = config["name"]
        tp_sharding = config["tp_sharding"]
        sharding_dim = config["sharding_dim"]
        assert param_name in parameters, (
            f"Parameter {param_name} not found in training parameters {parameters.keys()}"
        )

        param = parameters[param_name]
        logger.info(f"Start to verify {param_name}")

        # Create slices for verification
        slices = tuple(slice(0, min(length, dim)) for dim in param.shape)

        # Get weights from inference backend
        weights = self.sglang_engine.execute_task_in_model_worker(
            get_weights_from_tp_worker, param_name=param_name, slices=slices
        )

        if not tp_sharding:
            # For non-tensor-parallel parameters, verify directly
            assert weights is not None and len(weights) > 0, (
                f"No weights returned for {param_name}"
            )
            train_weights = param[slices]
            for i in range(tp_size):
                inference_weights = weights[i] if isinstance(weights, list) else weights
                # Move both tensors to the same device (cuda:0) for comparison
                target_device = torch.device("cuda:0")
                inference_weights_gpu = inference_weights.to(target_device)
                train_weights_gpu = train_weights.to(target_device)
                if not torch.allclose(
                    inference_weights_gpu, train_weights_gpu, atol=1e-5
                ):
                    logger.error(f"Inference weights: \n{inference_weights_gpu}")
                    logger.error(f"Train weights: \n{train_weights_gpu}")
                    raise RuntimeError(
                        f"Parameter {param_name} verification failed for TP worker {i}"
                    )
        else:
            # For tensor-parallel parameters, verify each TP worker
            assert len(weights) == tp_size, (
                f"Expected {tp_size} weights for {param_name}, "
                f"got {len(weights) if isinstance(weights, list) else 'non-list'}"
            )
            for i in range(tp_size):
                tp_worker_weights = weights[i]
                assert tp_worker_weights is not None, (
                    f"No weights for TP worker {i} for {param_name}"
                )
                # Calculate the corresponding slice for this TP worker
                train_slices = self._calculate_tp_slices(
                    param.shape, i, tp_size, sharding_dim, length
                )
                train_weights = param[train_slices]
                # Move both tensors to the same device (cuda:0) for comparison
                target_device = torch.device("cuda:0")
                tp_worker_weights_gpu = tp_worker_weights.to(target_device)
                train_weights_gpu = train_weights.to(target_device)
                assert torch.allclose(
                    tp_worker_weights_gpu, train_weights_gpu, atol=1e-5
                ), f"Parameter {param_name} verification failed for TP worker {i}"

        logger.info(f"Verify {param_name} finished")

    def _calculate_tp_slices(self, param_shape, tp_rank, tp_size, sharding_dim, length):
        """Calculate the appropriate slices for tensor parallel verification."""
        slices = [slice(0, min(length, dim)) for dim in param_shape]
        assert sharding_dim is not None, (
            f"Sharding dimension is not set for parameter of shape {param_shape}"
        )
        # Calculate the local size for this dimension
        local_size = param_shape[sharding_dim] // tp_size
        start_idx = tp_rank * local_size
        end_idx = min(start_idx + length, start_idx + local_size)
        # Update the slice for the sharding dimension
        slices[sharding_dim] = slice(start_idx, end_idx)
        return tuple(slices)


def get_weights_from_tp_worker(**kwargs):
    model_context = kwargs["model_context"]
    scheduler = model_context["scheduler"]
    slices = kwargs["slices"]
    param_name = kwargs["param_name"]
    weights_reader = scheduler.awes_weights_reader
    if weights_reader is None:
        return None
    parameters = weights_reader.parameters
    if param_name in parameters:
        return parameters[param_name][slices]
    else:
        return None


def main(args):
    os.environ["NCCL_DEBUG"] = "WARNING"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_MAX_NCHANNELS"] = "8"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ALLOC"
    os.environ["GLOO_USE_LIBUV"] = "0"
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    comm_backend = args.comm_backend
    weights_exchange_it = WeightsExchangeIT(
        comm_backend=comm_backend,
    )
    weights_exchange_it.initialize()
    # warm up and init nccl
    logger.info("========== Warm up and init ==========")
    weights_exchange_it.exchange_weights()
    if comm_backend != "file":
        weights_exchange_it.verify_updated_parameters()
    # test weights exchange
    for i in range(10):
        logger.info(f"========== Test weights exchange {i} ==========")
        weights_exchange_it.exchange_weights()
    dist.destroy_process_group()
    weights_exchange_it.destroy()


if __name__ == "__main__":
    # export PYTHONPATH=$MegatronLM:$PYTHONPATH
    parser = argparse.ArgumentParser(description="Run awex integration test cases")
    parser.add_argument("-b", "--comm_backend", required=True, help="file|nccl|astate")
    args = parser.parse_args()
    main(args)
