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

from awex import logging
from awex.converter.mcore_converter import McoreToHFWeightConverter
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import per_block_cast_to_fp8
from awex.sharding.param_sharding import ShardingStrategy, ShardingType
from awex.sharding.rank_info import RankInfo

logger = logging.getLogger(__name__)


class BailingMoeShardingStrategy(ShardingStrategy):
    """
    Custom sharding strategy for BailingMoeForCausalLM model architecture.
    """

    def get_sharding_strategy(self, parameter_name, **kwargs):
        if self.engine_name == "mcore":
            if "query_key_value" in parameter_name:
                return ShardingType.NO_SHARDING, 0, 1
        return super().get_sharding_strategy(parameter_name, **kwargs)

    def get_embedding_sharding_strategy(self, parameter_name, **kwargs):
        tp_size = self.rank_info.tp_size
        if not self.enable_dp_attention and tp_size > 1:
            return ShardingType.TP_SHARDING, 0, tp_size
        else:
            return ShardingType.NO_SHARDING, 0, 1


class McoreToHFWeightConverterBailingMoe(McoreToHFWeightConverter):
    def __init__(
        self, hf_config: PretrainedConfig, rank_info: RankInfo, infer_conf: Dict
    ):
        super().__init__(hf_config, rank_info, infer_conf)
        self.quantization_config = getattr(hf_config, "quantization_config", {})
        self.quant_method = self.quantization_config.get("quant_method")
        self.fp8_weight_keys = set()
        if self.quant_method:
            assert self.quant_method == "fp8", "Only fp8 quantization is supported"
            self.quant_method = "fp8"
            self.fp8_weight_keys = {
                "up_proj.weight",
                "down_proj.weight",
                "gate_proj.weight",
                "attention.dense.weight",
                "attention.query_key_value.weight",
            }
            logger.info("Model is using fp8 quantization")

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        super_converted_params = super().convert_param(name, parameter)
        if not self.quant_method:
            return super_converted_params
        pair_list = []
        for param_name, param in super_converted_params:
            apply_fp8 = False
            scale_ue8m0 = False
            for fp8_key in self.fp8_weight_keys:
                # ue8m0 scale：
                # 1. attention.dense.weight
                # 2. up, gate in MoE(Note: dense MLP and shared_expert don't use ue8m0)
                if fp8_key in param_name:
                    apply_fp8 = True
                    if fp8_key == "attention.dense.weight":
                        scale_ue8m0 = True
                    if (
                        ".experts." in param_name
                        and "_proj.weight" in fp8_key
                        and "down_proj" not in fp8_key
                    ):
                        scale_ue8m0 = True
                    break
            if apply_fp8:
                qw, scale = per_block_cast_to_fp8(param, scale_ue8m0)
                pair_list.append((param_name, qw))
                pair_list.append((f"{param_name}_scale_inv", scale))
            else:
                pair_list.append((param_name, param))
        return pair_list


class SGlangToHFWeightConverterBailingMoe(SGlangToHFWeightConverter):
    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False


CONFIG = {
    "model_name": "BailingMoeForCausalLM",
    "sharding_strategy": BailingMoeShardingStrategy,
    "mcore_converter": McoreToHFWeightConverterBailingMoe,
    "sglang_converter": SGlangToHFWeightConverterBailingMoe,
}
