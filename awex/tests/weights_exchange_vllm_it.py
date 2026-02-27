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
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import requests
import torch
import torch.distributed as dist

from awex import logging
from awex.meta.meta_server import start_meta_server, stop_meta_server
from awex.util import device as device_util

logger = logging.getLogger(__name__)

# NOTE: vLLM plugin discovery requires awex to be installed (e.g., editable install)
# so that the `vllm.general_plugins` entry point is discoverable.

enable_debug_mode = False

tp_size = 1
vllm_inference_config = {
    "model_path": "/home/model/Qwen3-0.6B",
    "tp_size": tp_size,
    "pp_size": 1,
    "dp_size": 1,
    "ep_size": 1,
    "num_engines": 1,
    "engine_rank": 0,
    "comm_backend": "file",
    "enable_debug_mode": enable_debug_mode,
}


class VLLMWeightsExchangeIT:
    """
    Goal: Megatron (main process) uses the FIRST visible GPU.
          vLLM (child process) uses the SECOND visible GPU (or next tp_size GPUs).

    Key rule: Do NOT rely on changing os.environ["CUDA_VISIBLE_DEVICES"] in the same process
              after torch.cuda has been touched. Instead:
      - Main process: torch.cuda.set_device(0) to pin Megatron to GPU-0 (logical index).
      - Child process: pass env["CUDA_VISIBLE_DEVICES"]="1" (or list) to vLLM subprocess.
    """

    def __init__(
        self,
        inference_config=None,
        comm_backend=None,
        use_mbridge=False,
        host="127.0.0.1",
        port=8000,
        validate=False,
        dump_weights_list_for_validation=None,
        dump_weights_dir_for_validation=None,
    ):
        self.comm_backend = comm_backend
        self.device_backend = device_util.get_device_type()
        ip, port_meta = start_meta_server()
        self.meta_server_addr = f"{ip}:{port_meta}"
        self.inference_config = inference_config or copy.deepcopy(vllm_inference_config)
        self.inference_config["comm_backend"] = comm_backend
        self.inference_config["meta_server_addr"] = self.meta_server_addr
        self.train_config = {
            "meta_server_addr": self.meta_server_addr,
            "comm_backend": comm_backend,
            "enable_debug_mode": enable_debug_mode,
        }
        self.host = host
        self.port = port
        self.use_mbridge = use_mbridge
        self.validate = validate
        self.dump_weights_list_for_validation = dump_weights_list_for_validation or []
        self.dump_weights_dir_for_validation = dump_weights_dir_for_validation

        # Select devices so that:
        #   - Megatron uses first visible GPU
        #   - vLLM uses second (and onward for tp_size)
        self.vllm_visible_devices, self.megatron_device = self._select_devices()

        self.megatron_engine = None
        self.vllm_process = None

    def _select_devices(self):
        tp = self.inference_config["tp_size"]

        visible_env = device_util.visible_devices_env_value().strip()
        if visible_env:
            # Example: "0,1,2" -> [0,1,2] (physical ids as provided by user)
            visible_devices = [
                int(x) for x in visible_env.split(",") if x.strip() != ""
            ]
        else:
            # Fallback: use torch to detect. (May touch CUDA, but that's OK with set_device below.)
            visible_devices = list(range(device_util.device_count()))

        need = 1 + tp  # 1 for Megatron + tp for vLLM
        if len(visible_devices) < need:
            raise RuntimeError(
                f"Need at least {need} visible devices (1 for Megatron + {tp} for vLLM). "
                f"Found {len(visible_devices)} via visible devices env='{visible_env or '(unset)'}'."
            )

        # Pin Megatron to the FIRST visible GPU
        megatron_device = visible_devices[0]
        # Give vLLM the NEXT tp GPUs
        vllm_devices = visible_devices[1 : 1 + tp]
        return vllm_devices, megatron_device

    def initialize(self):
        # 1) Init Megatron first on GPU-0 (main process)
        self._init_megatron_engine()
        # 2) Start vLLM server in a subprocess restricted to GPU-1 (or more)
        self._start_vllm_server()
        # 3) Awex init handshake
        self._awex_init()

    def destroy(self):
        if self.vllm_process is not None:
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
        stop_meta_server()

    def _start_vllm_server(self):
        env = os.environ.copy()
        # IMPORTANT: restrict the vLLM subprocess to its own devices
        visible_env = device_util.visible_devices_env_names()[0]
        env[visible_env] = ",".join(map(str, self.vllm_visible_devices))
        env.setdefault("AWEX_DEVICE_TYPE", device_util.get_device_type())

        cmd = [
            "python",
            "-m",
            "awex.awex_vllm_server",
            "--model",
            self.inference_config["model_path"],
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.inference_config["tp_size"]),
            "--pipeline-parallel-size",
            str(self.inference_config["pp_size"]),
            "--disable-log-requests",
            "--enforce-eager",
        ]
        logger.info("Starting vLLM server: %s", " ".join(cmd))
        logger.info("vLLM subprocess %s=%s", visible_env, env.get(visible_env, ""))

        self.vllm_process = subprocess.Popen(cmd, env=env)
        self._wait_for_health()

    def _wait_for_health(self, timeout=120):
        url = f"http://{self.host}:{self.port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(1)
        raise RuntimeError("vLLM server failed to start within timeout.")

    def _awex_init(self):
        url = f"http://{self.host}:{self.port}/areal_awex_init"
        payload = {
            "meta_server_addr": self.meta_server_addr,
            "engine_rank": self.inference_config["engine_rank"],
            "num_engines": self.inference_config["num_engines"],
            "comm_backend": self.inference_config["comm_backend"],
            "enable_debug_mode": enable_debug_mode,
            "nnodes": 1,
            "node_rank": 0,
        }
        if self.device_backend == "npu":
            payload["weights_exchange_ipc_backend"] = "cpu"
        if self.validate:
            payload["weights_validation_steps"] = 1
            payload["validate_weights_every_n_steps"] = 1
            if self.dump_weights_list_for_validation:
                payload["dump_weights_list_for_validation"] = (
                    self.dump_weights_list_for_validation
                )
            if self.dump_weights_dir_for_validation:
                payload["dump_weights_dir_for_validation"] = (
                    self.dump_weights_dir_for_validation
                )
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Awex init failed: {resp.text}")

    def _init_megatron_engine(self):
        self.train_config["tensor_model_parallel_size"] = 1
        self.train_config["pipeline_model_parallel_size"] = 1
        self.train_config["expert_model_parallel_size"] = 1

        # Single-process Megatron
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_PORT"] = "17443"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"

        # IMPORTANT:
        # Do NOT try to isolate Megatron by changing CUDA_VISIBLE_DEVICES in this same process.
        # Instead, explicitly pin Megatron to the first visible GPU (logical index 0).
        #
        # If your parent process sees multiple GPUs, logical 0 corresponds to the first one.
        # We'll also log the mapping expectation:
        logger.info(
            "Megatron intended physical device id=%s (first visible)",
            self.megatron_device,
        )
        logger.info(
            "Main-process visible devices env=%s",
            device_util.visible_devices_env_value() or "(unset)",
        )

        # Pin to logical GPU 0 for allocations + NCCL.
        # (Even if physical ids differ, logical 0 is the first visible device in this process.)
        device_util.set_device(0)

        # Optional sanity log:
        try:
            logger.info(
                "Megatron pinned device=%s:%s name=%s",
                device_util.get_device_type(),
                device_util.current_device(),
                device_util.get_device_name(device_util.current_device()),
            )
        except Exception:
            pass

        backend = "hccl" if device_util.get_device_type() == "npu" else "nccl"
        torch.distributed.init_process_group(backend)

        self.mcore_model, self.mcore_hf_config = self.setup_megatron()
        from awex.engine.mcore import MegatronEngine

        self.megatron_engine = MegatronEngine(
            self.train_config, self.mcore_hf_config, self.mcore_model
        )
        self.megatron_engine.initialize()
        logger.info("Megatron backend initialized")

    def setup_megatron(self):
        # Ensure MindSpeed patches are applied before importing Megatron when on NPU.
        from awex.util.mindspeed import ensure_mindspeed_patched

        ensure_mindspeed_patched("weights_exchange_vllm_it")

        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        from awex.tests.test_utils import megatron_model_from_hf

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )

        try:
            model_parallel_cuda_manual_seed(0)
        except Exception as exc:
            if device_util.get_device_type() != "npu":
                raise
            logger.warning(
                "model_parallel_cuda_manual_seed failed on NPU (%s); "
                "falling back to torch.npu.manual_seed.",
                exc,
            )
            if getattr(torch, "npu", None) is not None:
                torch.npu.manual_seed(0)

        model, hf_config = megatron_model_from_hf(
            model_path=self.inference_config["model_path"],
            use_mbridge=self.use_mbridge,
        )
        return model[0], hf_config

    def exchange_weights(self):
        if self.megatron_engine is None:
            raise RuntimeError("Megatron backend not initialized")

        if self.comm_backend == "file":
            temp_ctx = tempfile.TemporaryDirectory()
            path = os.path.join(temp_ctx.name, "checkpoint")
        else:
            temp_ctx = nullcontext()
            path = None

        with temp_ctx:
            if self.comm_backend == "file":
                self.megatron_engine.write_weights(path=path)
                self._awex_update(path=path)
            else:
                # For NCCL, start reader first so it can receive as soon as writer sends.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._awex_update, path=None)
                    time.sleep(1)
                    self.megatron_engine.write_weights()
                    future.result()
        logger.info("Update weights finished")

    def _awex_update(self, path: str | None):
        url = f"http://{self.host}:{self.port}/areal_awex_update"
        payload = {"step_id": self.megatron_engine.global_step, "kwargs": {}}
        if path is not None:
            payload["kwargs"]["path"] = path
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"Awex update failed: {resp.text}")


def main(args):
    os.environ["NCCL_DEBUG"] = "WARNING"
    comm_backend = args.comm_backend
    inference_config = copy.deepcopy(vllm_inference_config)
    if args.model_path:
        inference_config["model_path"] = args.model_path

    weights_exchange_it = VLLMWeightsExchangeIT(
        inference_config=inference_config,
        comm_backend=comm_backend,
        use_mbridge=args.use_mbridge,
        host=args.host,
        port=args.port,
        validate=args.validate,
        dump_weights_list_for_validation=args.dump_weights_list_for_validation,
        dump_weights_dir_for_validation=args.dump_weights_dir_for_validation,
    )

    try:
        weights_exchange_it.initialize()
        logger.info("========== Test weights exchange ==========")
        weights_exchange_it.exchange_weights()
    finally:
        weights_exchange_it.destroy()
        _destroy_process_group(timeout=5)


def _destroy_process_group(timeout: float = 5.0) -> None:
    if not dist.is_initialized():
        return

    error = {}

    def _destroy():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            error["exc"] = exc

    thread = threading.Thread(target=_destroy, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning("Timed out destroying process group; continuing.")
    elif error:
        logger.warning("Failed to destroy process group: %s", error["exc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run awex vLLM integration test")
    parser.add_argument(
        "-b",
        "--comm_backend",
        default="file",
        help="Weight exchange communication backend (file/nccl/hccl).",
    )
    parser.add_argument(
        "--model-path",
        default=vllm_inference_config["model_path"],
        help="HF model path used by Megatron and the vLLM server.",
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )
    parser.add_argument(
        "--use-mbridge",
        action="store_true",
        help="Load HF weights into Megatron via mbridge (skip DCP conversion).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable weights validation (NCCL or file backend).",
    )
    parser.add_argument(
        "--dump-weights-list-for-validation",
        default="",
        help="Comma-separated parameter names to dump during validation.",
    )
    parser.add_argument(
        "--dump-weights-dir-for-validation",
        default="",
        help="Directory to dump validation tensors.",
    )
    args = parser.parse_args()
    if args.device_backend and args.device_backend != "auto":
        os.environ["AWEX_DEVICE_TYPE"] = args.device_backend
    if device_util.get_device_type() == "npu" and args.comm_backend == "nccl":
        logger.warning("Switching comm_backend from nccl to hccl for NPU backend.")
        args.comm_backend = "hccl"
    if args.dump_weights_list_for_validation:
        args.dump_weights_list_for_validation = [
            name.strip()
            for name in args.dump_weights_list_for_validation.split(",")
            if name.strip()
        ]
    else:
        args.dump_weights_list_for_validation = []
    args.dump_weights_dir_for_validation = args.dump_weights_dir_for_validation or None
    main(args)
