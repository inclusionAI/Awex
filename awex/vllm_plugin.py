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

import asyncio
import logging
import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.api_server import router
from vllm.entrypoints.openai.protocol import OpenAIBaseModel

from awex.config import InferenceConfig
from awex.vllm_awex_adapter import AwexVLLMServerAdapter

logger = logging.getLogger(__name__)

_awex_plugin_registered = False
_AWEX_WORKER_METHODS = {
    "_get_model_param_info": (
        "awex.meta.infer_meta_resolver",
        "InferParamMetaResolver._get_model_param_info",
    ),
    "_init_in_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._init_in_tp_worker",
    ),
    "_update_parameters_in_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._update_parameters_in_tp_worker",
    ),
    "_pre_update_weights_in_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._pre_update_weights_in_tp_worker",
    ),
    "_pre_validate_weights_on_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._pre_validate_weights_on_tp_worker",
    ),
    "_verify_weights_on_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._verify_weights_on_tp_worker",
    ),
    # Optional test helper (kept as a template):
    # "get_weights_from_tp_worker": (
    #     "awex.tests.weights_exchange_it",
    #     "get_weights_from_tp_worker",
    # ),
}
_AWEX_WORKER_SIGNATURES = {
    "_get_model_param_info": {
        "required": ["engine_name", "infer_engine_config"],
        "optional": ["convert_params", "engine_rank"],
    },
    "_init_in_tp_worker": {
        "required": [
            "infer_conf_bytes",
            "parameters_meta_bytes",
            "training_params_meta_bytes",
            "engine_rank",
            "num_engines",
            "meta_server_addr",
            "weights_comm_backend",
            "debug_mode_config",
            "disable_pipeline",
            "enable_colocate_mode",
            "ipc_backend",
        ],
        "optional": ["enable_debug_mode", "weights_comm_nccl_group_size"],
    },
    "_update_parameters_in_tp_worker": {"required": ["step_id"], "optional": []},
    "_pre_update_weights_in_tp_worker": {"required": ["step_id"], "optional": []},
    "_pre_validate_weights_on_tp_worker": {"required": ["step_id"], "optional": []},
    "_verify_weights_on_tp_worker": {
        "required": ["step_id"],
        "optional": [
            "dump_weights_list_for_validation",
            "dump_weights_dir_for_validation",
        ],
    },
}


class AwexInitRequest(OpenAIBaseModel):
    meta_server_addr: str
    engine_rank: int = 0
    num_engines: int = 1
    comm_backend: str = "file"
    enable_debug_mode: bool = False
    debug_mode_config: dict[str, Any] | None = None
    disable_weights_exchange_pipeline: bool = False
    enable_colocate_mode: bool = False
    weights_exchange_ipc_backend: str = "cuda"
    weights_comm_nccl_group_size: int = 1
    nnodes: int | None = None
    node_rank: int | None = None
    weights_validation_steps: int = 0
    validate_weights_every_n_steps: int = 1
    dump_weights_list_for_validation: list[str] | None = None
    dump_weights_dir_for_validation: str | None = None


class AwexUpdateRequest(OpenAIBaseModel):
    step_id: int
    kwargs: dict[str, Any] | None = None


def _to_json_response(success: bool, message: str):
    content = {"success": success, "message": message}
    status_code = 200 if success else 400
    return JSONResponse(content, status_code=status_code)


def _to_json_error(message: str, status_code: int = 500):
    content = {"success": False, "message": message}
    return JSONResponse(content, status_code=status_code)


def _sanitize_for_ipc(obj):
    # Ensure objects are msgpack-serializable for vLLM EngineCore IPC.
    try:
        import torch

        if isinstance(obj, torch.dtype):
            return str(obj).replace("torch.", "")
        if isinstance(obj, torch.device):
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize_for_ipc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_ipc(v) for v in obj]
    return obj


def _get_awex_adapter(raw_request):
    adapter = getattr(raw_request.app.state, "awex_adapter", None)
    if adapter is None:
        raise RuntimeError("Awex adapter not initialized. Call /areal_awex_init first.")
    return adapter


def _patch_awex_worker() -> None:
    try:
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            get_pp_group,
            get_tp_group,
        )
        from vllm.v1.worker.worker_base import WorkerBase
    except Exception as exc:
        logger.warning("Failed to patch vLLM worker for Awex: %s", exc)
        return

    def _awex_rank_info(self, infer_engine_config: InferenceConfig | None = None):
        parallel_config = self.model_runner.vllm_config.parallel_config
        external_engine_rank = 0
        if infer_engine_config is not None:
            external_engine_rank = int(
                getattr(infer_engine_config, "engine_rank", 0) or 0
            )
        try:
            tp_group = get_tp_group()
            tp_rank = tp_group.rank_in_group
            tp_size = tp_group.world_size
        except AssertionError:
            tp_rank, tp_size = 0, 1
        try:
            pp_group = get_pp_group()
            pp_rank = pp_group.rank_in_group
            pp_size = pp_group.world_size
        except AssertionError:
            pp_rank, pp_size = 0, 1
        try:
            dp_group = get_dp_group()
            dp_rank = dp_group.rank_in_group
            dp_size = dp_group.world_size
        except AssertionError:
            dp_rank = parallel_config.data_parallel_rank
            dp_size = parallel_config.data_parallel_size
        try:
            ep_group = get_ep_group()
            ep_rank = ep_group.rank_in_group
            ep_size = ep_group.world_size
        except AssertionError:
            ep_rank, ep_size = 0, 1

        local_world_size = int(getattr(parallel_config, "world_size", 1) or 1)
        local_rank = int(getattr(parallel_config, "rank", 0) or 0)
        cp_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
        cp_rank = 0
        if cp_size > 1:
            try:
                from vllm.distributed import parallel_state as _ps

                get_pcp_group = getattr(_ps, "get_pcp_group", None)
                if callable(get_pcp_group):
                    pcp_group = get_pcp_group()
                    cp_rank = int(getattr(pcp_group, "rank_in_group", 0))
                    cp_size = int(getattr(pcp_group, "world_size", cp_size))
                else:
                    cp_rank = local_rank % cp_size
            except Exception:
                cp_rank = local_rank % cp_size
        cp_mode = os.environ.get("AWEX_CP_MODE")
        if not cp_mode:
            cp_mode = "ring" if cp_size > 1 else "none"
        # In internal DP mode, each core process commonly uses rank in [0, TP*PP*CP),
        # so compose a world-size-across-dp global rank with dp_rank.
        if 0 <= local_rank < local_world_size:
            global_rank = dp_rank * local_world_size + local_rank
        else:
            # Fallback for launchers that already expose a fully global rank.
            global_rank = local_rank

        reported_local_rank = getattr(self, "local_rank", local_rank)
        return {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "ep_tp_rank": 0,
            "ep_tp_size": 1,
            "local_rank": reported_local_rank,
            "global_rank": global_rank,
            "world_size": parallel_config.world_size_across_dp,
            # engine_rank is AWEX external instance index, not vLLM internal DP rank.
            "engine_rank": external_engine_rank,
            "is_infer": True,
            "attn_tp_rank": tp_rank,
            "attn_tp_size": tp_size,
            "attn_dp_rank": 0,
            "cp_rank": cp_rank,
            "cp_size": cp_size,
            "cp_mode": cp_mode,
        }

    def _awex_model_context(self, infer_engine_config: InferenceConfig | None = None):
        if not hasattr(self, "_awex_infer_engine_config"):
            parallel_config = self.model_runner.vllm_config.parallel_config
            nnodes = getattr(parallel_config, "nnodes", 1)
            node_rank = getattr(parallel_config, "node_rank", 0)
            enable_expert_parallel = bool(
                getattr(parallel_config, "enable_expert_parallel", False)
            )
            pcp_size = int(
                getattr(parallel_config, "prefill_context_parallel_size", 1) or 1
            )
            inferred_ep_size = (
                parallel_config.tensor_parallel_size
                * parallel_config.data_parallel_size
                * pcp_size
                if enable_expert_parallel
                else 1
            )
            external_num_engines = 1
            external_engine_rank = 0
            if infer_engine_config is not None:
                external_num_engines = int(
                    getattr(infer_engine_config, "num_engines", 1) or 1
                )
                external_engine_rank = int(
                    getattr(infer_engine_config, "engine_rank", 0) or 0
                )
            comm_backend = (
                getattr(infer_engine_config, "comm_backend", None)
                if infer_engine_config is not None
                else None
            )
            if not comm_backend:
                comm_backend = os.environ.get("AWEX_COMM_BACKEND", "file")
            self._awex_infer_engine_config = InferenceConfig(
                tp_size=parallel_config.tensor_parallel_size,
                pp_size=parallel_config.pipeline_parallel_size,
                dp_size=parallel_config.data_parallel_size,
                ep_size=inferred_ep_size,
                enable_dp_attention=False,
                enable_dp_lm_head=False,
                moe_dense_tp_size=None,
                nnodes=nnodes,
                node_rank=node_rank,
                # AWEX num_engines/engine_rank describe external inference instances.
                # Do not derive them from vLLM internal DP.
                num_engines=external_num_engines,
                engine_rank=external_engine_rank,
                comm_backend=comm_backend,
            )
        base_config = self._awex_infer_engine_config
        if infer_engine_config is not None:
            merged = InferenceConfig.from_dict(base_config.__dict__, False)
            for field in InferenceConfig.__dataclass_fields__:
                value = getattr(infer_engine_config, field, None)
                if value is not None:
                    setattr(merged, field, value)
            base_config = merged
        model_context = _awex_rank_info(self, base_config)
        model_context["scheduler"] = self
        model_context["infer_engine_config"] = base_config
        return model_context

    def awex_get_model_context(self):
        return _awex_rank_info(self, None)

    def awex_execute(
        self, task_module: str, task_qualname: str, task_kwargs: dict | None = None
    ):
        module = __import__(task_module, fromlist=["__dummy__"])
        target = module
        for attr in task_qualname.split("."):
            target = getattr(target, attr)
        task_kwargs = task_kwargs or {}
        infer_engine_config = task_kwargs.get("infer_engine_config")
        if isinstance(infer_engine_config, dict):
            infer_engine_config = InferenceConfig.from_dict(infer_engine_config)
            task_kwargs["infer_engine_config"] = infer_engine_config
        task_kwargs["model"] = self.model_runner.model
        task_kwargs["model_context"] = _awex_model_context(self, infer_engine_config)
        result = target(**task_kwargs)
        return _sanitize_for_ipc(result)

    WorkerBase.awex_get_model_context = awex_get_model_context
    WorkerBase.awex_execute = awex_execute
    WorkerBase.awex_update_weights_from_disk = awex_update_weights_from_disk
    WorkerBase.flush_cache = flush_cache

    def _make_awex_worker_method(task_module: str, task_qualname: str):
        method_name = task_qualname.split(".")[-1]

        def _method(self, **kwargs):
            filtered_kwargs = _filter_awex_kwargs(method_name, kwargs)
            return awex_execute(self, task_module, task_qualname, filtered_kwargs)

        return _method

    for method_name, (task_module, task_qualname) in _AWEX_WORKER_METHODS.items():
        setattr(
            WorkerBase,
            method_name,
            _make_awex_worker_method(task_module, task_qualname),
        )


def _filter_awex_kwargs(method_name: str, kwargs: dict) -> dict:
    signature = _AWEX_WORKER_SIGNATURES.get(method_name)
    if signature is None:
        return kwargs
    required = signature.get("required", [])
    optional = signature.get("optional", [])
    allowed = set(required) | set(optional)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    missing = [k for k in required if k not in filtered]
    if missing:
        raise ValueError(f"Missing required args for {method_name}: {missing}")
    return filtered


def awex_update_weights_from_disk(
    self, model_path: str, load_format: str | None = None
):
    from vllm.model_executor.model_loader import get_model_loader

    self.model_runner.model_config.model = model_path
    model_loader = get_model_loader(self.model_runner.vllm_config.load_config)
    model_loader.load_weights(
        self.model_runner.model, model_config=self.model_runner.model_config
    )
    return True


def flush_cache(self):
    flush_fn = getattr(self.model_runner, "flush_cache", None)
    if callable(flush_fn):
        return flush_fn()
    return True


def register_awex_plugin() -> None:
    """Register Awex endpoints and worker patches for vLLM."""
    global _awex_plugin_registered
    if _awex_plugin_registered:
        return
    _awex_plugin_registered = True

    _patch_awex_worker()

    @router.post("/areal_awex_init")
    async def awex_init(request: AwexInitRequest, raw_request: Request):
        try:
            logger.info("API server starts awex_init")
            llm = raw_request.app.state.engine_client
            adapter = AwexVLLMServerAdapter(
                llm,
                meta_server_addr=request.meta_server_addr,
                engine_rank=request.engine_rank,
                num_engines=request.num_engines,
                comm_backend=request.comm_backend,
                enable_debug_mode=request.enable_debug_mode,
                debug_mode_config=request.debug_mode_config,
                disable_weights_exchange_pipeline=request.disable_weights_exchange_pipeline,
                enable_colocate_mode=request.enable_colocate_mode,
                weights_exchange_ipc_backend=request.weights_exchange_ipc_backend,
                weights_comm_nccl_group_size=request.weights_comm_nccl_group_size,
                nnodes=request.nnodes,
                node_rank=request.node_rank,
                weights_validation_steps=request.weights_validation_steps,
                validate_weights_every_n_steps=request.validate_weights_every_n_steps,
                dump_weights_list_for_validation=request.dump_weights_list_for_validation,
                dump_weights_dir_for_validation=request.dump_weights_dir_for_validation,
                loop=asyncio.get_running_loop(),
            )
            await asyncio.to_thread(adapter.initialize)
            raw_request.app.state.awex_adapter = adapter
            return _to_json_response(True, "Awex initialized")
        except Exception as exc:
            logger.exception("Awex init failed")
            return _to_json_error(f"Awex init failed: {exc}")

    @router.post("/areal_awex_update")
    async def awex_update(request: AwexUpdateRequest, raw_request: Request):
        try:
            logger.info("API server starts awex_update, step_id=%s", request.step_id)
            adapter = _get_awex_adapter(raw_request)
            kwargs = request.kwargs or {}
            await asyncio.to_thread(adapter.update_weights, request.step_id, **kwargs)
            return _to_json_response(True, "Awex update done")
        except Exception as exc:
            logger.exception("Awex update failed")
            return _to_json_error(f"Awex update failed: {exc}")


def register_awex_routes() -> None:
    """Public entrypoint to register Awex routes without relying on vLLM plugin system."""
    register_awex_plugin()
