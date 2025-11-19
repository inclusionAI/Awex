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
        self._model_arch_name = self.hf_config.architectures[0]
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
        )
        for model in self._mcore_model:
            params_dict = get_mcore_model_parameters(model)
            for name, param in params_dict.items():
                params = []
                if convert_params:
                    for hf_name, hf_param in mcore_to_hf_weight_converter.convert_param(
                        name, param
                    ):
                        params.append((hf_name, hf_param))
                else:
                    params.append((name, param))
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
                        }
                    )
        # use all gather to get the global meta
        global_metadata: List[Dict[str, Any]] = [None] * dist.get_world_size()  # type: ignore
        logger.info(
            f"Starting all_gather_object of {dist.get_world_size()}, current rank {dist.get_rank()}"
        )
        dist.all_gather_object(global_metadata, meta)
        return global_metadata

    def _get_sharding_info(
        self, name: str, rank_info: RankInfo, param_meta: Dict[str, Any]
    ) -> Tuple[ShardingType, int, int]:
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )
