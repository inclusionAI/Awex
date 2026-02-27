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

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from awex.meta.meta_resolver import ParamMetaResolver, logger
from awex.meta.weight_meta import (
    ParameterMeta,
    compute_total_model_size,
    dump_parameters_meta,
)
from awex.models.registry import get_infer_weights_converter
from awex.sharding import get_rank_info_extractor, get_sharding_strategy_builder
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.util.common import to_dict


class InferParamMetaResolver(ParamMetaResolver):
    def __init__(
        self,
        inference_engine,
        convert_params=False,
        num_engines=1,
        engine_rank=0,
    ):
        """
        Args:
            inference_engine: The inference engine object that can execute tasks in model workers.
            convert_params: Whether to convert the parameters to the Hugging Face format.
        """
        super().__init__(inference_engine.hf_config)
        self._inference_engine = inference_engine
        self.infer_engine_config = inference_engine.config
        self.engine_name = inference_engine.engine_name
        self.convert_params = convert_params
        self.num_engines = num_engines
        self.engine_rank = engine_rank

        suffix = f"{engine_rank}_{os.getpid()}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
        if self._inference_engine.config.enable_debug_mode:
            non_converted_params_raw_meta = (
                inference_engine.execute_task_in_model_worker(
                    self._get_model_param_info,
                    engine_name=self.engine_name,
                    infer_engine_config=self.infer_engine_config,
                    engine_rank=engine_rank,
                    convert_params=False,
                )
            )
            filename = f"infer_params_non_converted_raw_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(non_converted_params_raw_meta), f, indent=4)
            logger.info(
                f"Inference rank {engine_rank}, non_converted_params_raw_meta: {abs_filename}"
            )
        self._params_raw_meta = inference_engine.execute_task_in_model_worker(
            self._get_model_param_info,
            engine_name=self.engine_name,
            infer_engine_config=self.infer_engine_config,
            engine_rank=engine_rank,
            convert_params=self.convert_params,
        )
        # vLLM IPC serializes RankInfo to dict; restore for internal usage.
        for info in self._params_raw_meta:
            rank_info = info.get("rank_info")
            if isinstance(rank_info, dict):
                info["rank_info"] = RankInfo(**rank_info)
        if self._inference_engine.config.enable_debug_mode:
            filename = f"infer_params_raw_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(self._params_raw_meta), f, indent=4)
            logger.info(
                f"Inference rank {engine_rank}, params_raw_meta: {abs_filename}"
            )
        self._rank0_meta = self._select_canonical_rank0_meta(self._params_raw_meta)
        self.rank0_info = self._rank0_meta["rank_info"]
        self._world_size = self.rank0_info.world_size
        self._model_arch_name = self._rank0_meta["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(self.engine_name)(
            self._model_arch_name,
            self.infer_engine_config,
            self.rank0_info,
        )
        self._params_meta = self._build_params_meta()
        if self._inference_engine.config.enable_debug_mode:
            filename = f"infer_params_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(dump_parameters_meta(self._params_meta), f, indent=4)
            logger.info(f"Inference rank {engine_rank}, params_meta: {abs_filename}")
        self.total_numel = sum(param.global_numel for param in self._params_meta)
        self.total_size = compute_total_model_size(self._params_meta)
        logger.info(
            f"Total number of elements in the model: {self.total_numel}, total size: {self.total_size} bytes"
        )

    def get_model_arch_name(self) -> str:
        return self._model_arch_name

    def get_parameters_meta(self) -> List[ParameterMeta]:
        """
        Returns the list of ParameterMeta objects for all parameters in the model.
        """
        return self._params_meta

    def _get_params_raw_meta(self) -> List[Dict[str, Any]]:
        return self._params_raw_meta

    @staticmethod
    def _meta_identity_key(info: Dict[str, Any]):
        rank_info = info["rank_info"]
        return (
            rank_info.global_rank,
            rank_info.dp_rank,
            rank_info.tp_rank,
            rank_info.pp_rank,
            rank_info.ep_rank,
            rank_info.attn_tp_rank,
            rank_info.attn_dp_rank,
        )

    @classmethod
    def _select_canonical_rank0_meta(
        cls, params_raw_meta: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not params_raw_meta:
            raise ValueError("No inference parameter metadata collected.")

        rank0_params = [
            info for info in params_raw_meta if info["rank_info"].global_rank == 0
        ]
        if rank0_params:
            by_identity = {}
            for info in rank0_params:
                by_identity[cls._meta_identity_key(info)] = info
            dedup_rank0 = list(by_identity.values())
            if len(dedup_rank0) > 1:
                logger.warning(
                    "Found %s rank0 metas (dedup=%s); selecting canonical by "
                    "(dp,tp,pp,ep,attn_tp,attn_dp).",
                    len(rank0_params),
                    len(dedup_rank0),
                )
            dedup_rank0.sort(
                key=lambda info: (
                    info["rank_info"].dp_rank,
                    info["rank_info"].tp_rank,
                    info["rank_info"].pp_rank,
                    info["rank_info"].ep_rank,
                    info["rank_info"].attn_tp_rank,
                    info["rank_info"].attn_dp_rank,
                )
            )
            return dedup_rank0[0]

        # Fallback: if no global_rank==0 exists (unexpected), use minimum identity.
        logger.warning(
            "No global_rank==0 meta found; falling back to minimum rank identity."
        )
        return min(params_raw_meta, key=cls._meta_identity_key)

    def _get_sharding_info(
        self, name: str, rank_info: RankInfo, param_meta: Dict[str, Any]
    ) -> Tuple[ShardingType, int, int]:
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )

    @staticmethod
    def _get_model_param_info(
        engine_name, infer_engine_config, convert_params=False, engine_rank=0, **kwargs
    ):
        """
        Static method to extract parameter meta information from a model and its context.
        Args:
            kwargs: Should contain 'model' and 'model_context'.
        Returns:
            dict: Metadata for the current rank, including rank_info, params_meta, and model_arch_name.
        """
        model = kwargs["model"]
        model_context = kwargs["model_context"]
        params_meta = []
        rank_info = get_rank_info_extractor(engine_name)(model_context, engine_rank)
        model_arch_name = type(model).__name__
        meta = {
            "rank_info": rank_info,
            "params_meta": params_meta,
            "model_arch_name": model_arch_name,
        }
        sglang_to_hf_weight_converter = get_infer_weights_converter(
            engine_name,
            model_arch_name,
            hf_config=model.config,
            infer_engine_config=infer_engine_config,
            rank_info=rank_info,
        )
        params = []
        for name, param in model.named_parameters():
            if convert_params:
                for hf_name, hf_param in sglang_to_hf_weight_converter.convert_param(
                    name, param
                ):
                    params.append((hf_name, hf_param))
            else:
                params.append((name, param))
        if convert_params and engine_name == "vllm":
            hf_config = getattr(model, "config", None) or getattr(
                model_context, "hf_config", None
            )
            if getattr(hf_config, "tie_word_embeddings", False):
                pp_rank = model_context.get("pp_rank", 0)
                pp_size = model_context.get("pp_size", 1)
                names = {n for n, _ in params}
                if (
                    pp_rank == pp_size - 1
                    and "lm_head.weight" not in names
                    and "model.embed_tokens.weight" in names
                ):
                    embed_tensor = None
                    for n, p in params:
                        if n == "model.embed_tokens.weight":
                            embed_tensor = p
                            break
                    if embed_tensor is not None:
                        # vLLM ties output weights to embeddings, but does not expose lm_head.
                        params.append(("lm_head.weight", embed_tensor))
                        logger.info(
                            "Infer meta: added lm_head.weight alias for tied embeddings in vLLM"
                        )
        if os.environ.get("AWEX_DEBUG_INFER_META", "0") == "1":
            names = [n for n, _ in params]
            has_lm_head = any(
                n.endswith("lm_head.weight")
                or n.endswith("lm_head")
                or n.endswith("model.lm_head.weight")
                for n in names
            )
            has_embed = any(
                n.endswith("embed_tokens.weight")
                or n.endswith("model.embed_tokens.weight")
                for n in names
            )
            logger.info(
                "Infer meta debug: engine=%s engine_rank=%s pp_rank=%s/%s tp_rank=%s "
                "convert_params=%s has_lm_head=%s has_embed=%s total_params=%s",
                engine_name,
                engine_rank,
                model_context.get("pp_rank"),
                model_context.get("pp_size"),
                model_context.get("tp_rank"),
                convert_params,
                has_lm_head,
                has_embed,
                len(names),
            )
        non_contiguous = []
        for name, param in params:
            if not param.is_contiguous():
                non_contiguous.append((name, tuple(param.shape)))
            dtype_str = str(param.dtype).replace("torch.", "")
            params_meta.append(
                {
                    "name": name,
                    "numel": param.numel(),
                    "shape": tuple(param.shape),
                    "dtype": dtype_str,
                }
            )
        if non_contiguous:
            sample = ", ".join(f"{name}:{shape}" for name, shape in non_contiguous[:8])
            logger.info(
                "Infer meta rank %s has %s non-contiguous params (showing up to 8): %s",
                rank_info.global_rank,
                len(non_contiguous),
                sample,
            )
            if len(non_contiguous) > 8:
                logger.debug(
                    "Infer meta rank %s remaining non-contiguous params: %s",
                    rank_info.global_rank,
                    [name for name, _ in non_contiguous[8:]],
                )
        return meta
