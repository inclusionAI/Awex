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
from typing import Any, Dict, List, Optional, Tuple

from torch import distributed as dist
from transformers import PretrainedConfig

from awex.meta.meta_resolver import ParamMetaResolver, logger
from awex.meta.weight_meta import (
    ParameterMeta,
    compute_total_model_size,
    dump_parameters_meta,
)
from awex.models.registry import get_train_weights_converter
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.util.common import to_dict


class McoreParamMetaResolver(ParamMetaResolver):
    def __init__(
        self,
        train_engine,
        hf_config: PretrainedConfig,
        infer_conf: Dict,
    ):
        super().__init__(hf_config)
        self._train_engine = train_engine
        self._mcore_model = train_engine.model
        if not isinstance(self._mcore_model, (list, tuple)):
            self._mcore_model = [self._mcore_model]
        self._model_arch_name = self.hf_config.architectures[0]
        self._tf_config = _maybe_get_tf_config(self._mcore_model)
        from awex.sharding.mcore_sharding import (
            get_mcore_rank_info,
            get_mcore_sharding_strategy,
        )

        self._rank_info = get_mcore_rank_info()
        self._sharding_strategy = get_mcore_sharding_strategy(
            self._model_arch_name,
            self._rank_info,
        )
        rank = self._rank_info.global_rank
        self._infer_conf = infer_conf
        self.infer_hf_config = infer_conf["hf_config"]
        self.num_hidden_layers = self.infer_hf_config.num_hidden_layers
        self._pp_stage_layer_id_map: Dict[Tuple[int, int], Dict[int, int]] = {}
        # yyyy_mm_dd_hh_mm_ss
        suffix = (
            f"_{rank}_{os.getpid()}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
        )
        if self._train_engine.enable_debug_mode:
            non_converted_params_raw_meta = self._collect_model_param_raw_info(False)
            filename = f"train_params_non_converted_raw_meta{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(non_converted_params_raw_meta), f, indent=4)
            logger.info(
                f"Training rank {rank}, non_converted_params_raw_meta: {abs_filename}"
            )
        self._params_raw_meta = self._collect_model_param_raw_info(True)
        if self._train_engine.enable_debug_mode:
            filename = f"train_params_raw_meta{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(self._params_raw_meta), f, indent=4)
            logger.info(f"Training rank {rank}, params_raw_meta: {abs_filename}")
        self._params_meta = self._build_params_meta()
        if self._train_engine.enable_debug_mode:
            filename = f"train_params_meta{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(dump_parameters_meta(self._params_meta), f, indent=4)
            logger.info(f"Training rank {rank}, params_meta: {abs_filename}")
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

    def get_pp_stage_layer_id_map(self) -> Dict[Tuple[int, int], Dict[int, int]]:
        return self._pp_stage_layer_id_map

    def _collect_model_param_raw_info(
        self, convert_params=False, **kwargs
    ) -> List[Dict[str, Any]]:
        params_meta = []
        from awex.sharding.mcore_sharding import get_mcore_rank_info

        rank_info = get_mcore_rank_info()
        meta = {
            "rank_info": rank_info,
            "params_meta": params_meta,
            "model_arch_name": self._model_arch_name,
        }
        from awex.converter.mcore_converter import get_mcore_model_parameters

        mcore_to_hf_weight_converter = get_train_weights_converter(
            self._train_engine.engine_name,
            self._model_arch_name,
            self.hf_config,
            self._rank_info,
            self._infer_conf,
            tf_config=self._tf_config,
        )
        for vp_stage, model in enumerate(self._mcore_model):
            params_dict = get_mcore_model_parameters(model)
            for name, param in params_dict.items():
                params = []
                if convert_params:
                    for hf_name, hf_param in mcore_to_hf_weight_converter.convert_param(
                        name, param, vp_stage=vp_stage
                    ):
                        params.append((hf_name, hf_param))
                else:
                    params.append((name, param))
                if convert_params and getattr(
                    self.hf_config, "tie_word_embeddings", False
                ):
                    names = {n for n, _ in params}
                    if (
                        rank_info.pp_rank == rank_info.pp_size - 1
                        and "lm_head.weight" not in names
                        and "model.embed_tokens.weight" in names
                    ):
                        embed_tensor = None
                        for n, p in params:
                            if n == "model.embed_tokens.weight":
                                embed_tensor = p
                                break
                        if embed_tensor is not None:
                            params.append(("lm_head.weight", embed_tensor))
                for name, param in params:
                    if not param.is_contiguous():
                        logger.info(
                            f"Parameter {name} is not contiguous, shape: {param.shape}, "
                            f"rank: {rank_info.global_rank}"
                        )
                    params_meta.append(
                        {
                            "name": name,
                            "numel": param.numel(),
                            "shape": tuple(param.shape),
                            "dtype": param.dtype,
                            "vp_stage": vp_stage,
                        }
                    )
        # use all gather to get the global meta
        global_metadata: List[Dict[str, Any]] = [None] * dist.get_world_size()  # type: ignore
        logger.info(
            f"Starting all_gather_object of {dist.get_world_size()}, current rank {dist.get_rank()}"
        )
        dist.all_gather_object(global_metadata, meta)
        if convert_params:
            self._pp_stage_layer_id_map = _build_pp_stage_layer_id_map(global_metadata)
            global_metadata = _canonicalize_pp_layer_names_in_global_meta(
                global_metadata, self._pp_stage_layer_id_map
            )
        else:
            self._pp_stage_layer_id_map = {}
        return global_metadata

    def _get_sharding_info(
        self, name: str, rank_info: RankInfo, param_meta: Dict[str, Any]
    ) -> Tuple[ShardingType, int, int]:
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )


def _maybe_get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                return cfg
    return None


def _extract_layer_id_from_param_name(name: str) -> Optional[int]:
    marker = ".layers."
    marker_idx = name.find(marker)
    if marker_idx < 0:
        return None
    start = marker_idx + len(marker)
    end = name.find(".", start)
    if end < 0:
        return None
    token = name[start:end]
    if not token.isdigit():
        return None
    return int(token)


def _replace_layer_id_in_param_name(name: str, new_layer_id: int) -> str:
    marker = ".layers."
    marker_idx = name.find(marker)
    if marker_idx < 0:
        return name
    start = marker_idx + len(marker)
    end = name.find(".", start)
    if end < 0:
        return name
    return f"{name[:start]}{new_layer_id}{name[end:]}"


def _build_pp_stage_layer_id_map(
    global_metadata: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], Dict[int, int]]:
    stage_local_ids: Dict[Tuple[int, int], set] = {}
    pp_size: Optional[int] = None
    for rank_meta in global_metadata:
        rank_info: RankInfo = rank_meta["rank_info"]
        if pp_size is None:
            pp_size = int(rank_info.pp_size)
        for param_meta in rank_meta["params_meta"]:
            local_layer_id = _extract_layer_id_from_param_name(param_meta["name"])
            if local_layer_id is None:
                continue
            vp_stage = int(param_meta.get("vp_stage", 0))
            stage_key = (int(rank_info.pp_rank), vp_stage)
            stage_local_ids.setdefault(stage_key, set()).add(local_layer_id)

    if not stage_local_ids:
        return {}
    if pp_size is None:
        raise ValueError("Failed to resolve pp_size from global metadata")

    vp_stages = sorted({stage_key[1] for stage_key in stage_local_ids})
    global_layer_id = 0
    stage_map: Dict[Tuple[int, int], Dict[int, int]] = {}
    for vp_stage in vp_stages:
        for pp_rank in range(pp_size):
            key = (pp_rank, vp_stage)
            if key not in stage_local_ids:
                continue
            local_ids = sorted(stage_local_ids[key])
            stage_map[key] = {
                local_layer_id: global_layer_id + offset
                for offset, local_layer_id in enumerate(local_ids)
            }
            global_layer_id += len(local_ids)
    return stage_map


def _canonicalize_pp_layer_names_in_global_meta(
    global_metadata: List[Dict[str, Any]],
    pp_stage_layer_id_map: Dict[Tuple[int, int], Dict[int, int]],
) -> List[Dict[str, Any]]:
    if not pp_stage_layer_id_map:
        return global_metadata

    for rank_meta in global_metadata:
        rank_info: RankInfo = rank_meta["rank_info"]
        pp_rank = int(rank_info.pp_rank)
        for param_meta in rank_meta["params_meta"]:
            local_layer_id = _extract_layer_id_from_param_name(param_meta["name"])
            if local_layer_id is None:
                continue
            vp_stage = int(param_meta.get("vp_stage", 0))
            stage_key = (pp_rank, vp_stage)
            stage_layer_map = pp_stage_layer_id_map.get(stage_key)
            if stage_layer_map is None:
                raise ValueError(
                    "Missing pp stage layer map while canonicalizing names: "
                    f"stage={stage_key}, param={param_meta['name']}, rank={rank_info.global_rank}"
                )
            if local_layer_id not in stage_layer_map:
                raise ValueError(
                    "Local layer id missing in pp stage layer map while canonicalizing names: "
                    f"stage={stage_key}, local_layer_id={local_layer_id}, "
                    f"param={param_meta['name']}, rank={rank_info.global_rank}"
                )
            param_meta["name"] = _replace_layer_id_in_param_name(
                param_meta["name"], stage_layer_map[local_layer_id]
            )
    return global_metadata
