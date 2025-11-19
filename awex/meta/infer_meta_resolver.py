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
        if self._inference_engine.config.enable_debug_mode:
            filename = f"infer_params_raw_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(self._params_raw_meta), f, indent=4)
            logger.info(
                f"Inference rank {engine_rank}, params_raw_meta: {abs_filename}"
            )
        rank0_params = [
            info for info in self._params_raw_meta if info["rank_info"].global_rank == 0
        ]
        if len(rank0_params) != 1:
            logger.error(f"Expected 1 rank0 meta, got {rank0_params}")
            raise ValueError(f"Expected 1 rank0 meta, got {len(rank0_params)}")
        [self._rank0_meta] = rank0_params
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
        return meta
