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

from typing import Dict, List, Tuple

import torch
from transformers import PretrainedConfig

from awex.converter.mcore_converter import LinearMLAMcoreConverterMixin
from awex.converter.sglang_converter import (
    LinearMLASGlangConverterMixin,
    SGlangToHFWeightConverter,
)
from awex.converter.weights_converter import per_block_cast_to_fp8
from awex.models.ling import (
    BailingMoeShardingStrategy,
    _build_mcore_converter_bailing_moe,
)
from awex.sharding.param_sharding import LinearMLAShardingMixin


class BailingLinearMoeShardingStrategy(
    LinearMLAShardingMixin, BailingMoeShardingStrategy
):
    pass


def _build_mcore_converter_bailing_moe_linear():
    BaseBailingMoeConverter = _build_mcore_converter_bailing_moe()

    class McoreToHFWeightConverterBailingMoeLinear(
        LinearMLAMcoreConverterMixin, BaseBailingMoeConverter
    ):
        def __init__(
            self,
            hf_config: PretrainedConfig,
            rank_info,
            infer_conf: Dict,
            tf_config,
        ):
            super().__init__(hf_config, rank_info, infer_conf, tf_config=tf_config)
            if self.quant_method == "fp8":
                self.fp8_weight_keys = {
                    "up_proj.weight",
                    "down_proj.weight",
                    "gate_proj.weight",
                }

        def _convert_lm_head_param(
            self, name: str, parameter: torch.Tensor
        ) -> List[Tuple[str, torch.Tensor]]:
            return [("lm_head.weight", parameter.to(torch.float32))]

        def _convert_expert_bias_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> Tuple[str, torch.Tensor]:
            if "expert_bias" in name:
                return ("mlp.gate.expert_bias", parameter.to(torch.float32))
            return super()._convert_expert_bias_param(name, parameter, layer_number)

        def _post_process_linear_mla_params(
            self, converted_params: List[Tuple[str, torch.Tensor]]
        ) -> List[Tuple[str, torch.Tensor]]:
            if self.quant_method != "fp8":
                return converted_params
            quantized_params: List[Tuple[str, torch.Tensor]] = []
            for param_name, param in converted_params:
                should_quantize = (
                    ".experts." in param_name
                    and "shared_experts" not in param_name
                    and any(key in param_name for key in self.fp8_weight_keys)
                )
                if not should_quantize:
                    quantized_params.append((param_name, param))
                    continue
                qw, scale = per_block_cast_to_fp8(param, False)
                quantized_params.append((param_name, qw))
                quantized_params.append((f"{param_name}_scale_inv", scale))
            return quantized_params

    return McoreToHFWeightConverterBailingMoeLinear


class SGlangToHFWeightConverterBailingMoeLinear(
    LinearMLASGlangConverterMixin,
    SGlangToHFWeightConverter,
):
    pass


CONFIG = [
    {
        "model_name": "BailingMoeV2ForCausalLM",
        "sharding_strategy": BailingLinearMoeShardingStrategy,
        "mcore_converter": _build_mcore_converter_bailing_moe_linear,
        "sglang_converter": SGlangToHFWeightConverterBailingMoeLinear,
    },
    {
        "model_name": "BailingMoeLinearForCausalLM",
        "sharding_strategy": BailingLinearMoeShardingStrategy,
        "mcore_converter": _build_mcore_converter_bailing_moe_linear,
        "sglang_converter": SGlangToHFWeightConverterBailingMoeLinear,
    },
    {
        "model_name": "BailingMoeV2_5ForCausalLM",
        "sharding_strategy": BailingLinearMoeShardingStrategy,
        "mcore_converter": _build_mcore_converter_bailing_moe_linear,
        "sglang_converter": SGlangToHFWeightConverterBailingMoeLinear,
    },
    {
        "model_name": "BailingMoeV2_5ForCausalLMNextN",
        "sharding_strategy": BailingLinearMoeShardingStrategy,
        "mcore_converter": _build_mcore_converter_bailing_moe_linear,
        "sglang_converter": SGlangToHFWeightConverterBailingMoeLinear,
    },
]
