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


def _cfg_value(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_scale_inv_name(name: str) -> Tuple[str, bool]:
    suffix = "_scale_inv"
    if name.endswith(suffix):
        return name[: -len(suffix)], True
    return name, False


def append_scale_inv(name: str, has_scale_inv: bool) -> str:
    if has_scale_inv:
        return f"{name}_scale_inv"
    return name


class LinearMLAMcoreConverterMixin:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        rank_info,
        infer_conf: Dict,
        tf_config,
    ):
        super().__init__(hf_config, rank_info, infer_conf, tf_config=tf_config)
        self.layer_group_size = int(_cfg_value(tf_config, "layer_group_size", 1) or 1)
        self.linear_attn_norm_group_size = _cfg_value(
            tf_config, "linear_attn_norm_group_size", None
        )
        self.fuse_qkv_a_proj = getattr(hf_config, "q_lora_rank", None) is not None
        self.qkv_a_proj_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _post_process_linear_mla_params(
        self, converted_params: List[Tuple[str, torch.Tensor]]
    ) -> List[Tuple[str, torch.Tensor]]:
        return converted_params

    def _is_linear_layer(self, layer_number: int) -> bool:
        if self.layer_group_size <= 1:
            return False
        return (layer_number + 1) % self.layer_group_size != 0

    def _convert_g_norm_weight(self, parameter: torch.Tensor) -> torch.Tensor:
        if not self.linear_attn_norm_group_size:
            return parameter.clone().detach()
        group_size = int(self.linear_attn_norm_group_size)
        tp_size = max(int(self.rank_info.tp_size), 1)
        return parameter.clone().detach().reshape(group_size // tp_size, -1)

    def _convert_lightning_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        from awex.converter.mcore_converter import convert_qkv_weight_along_tp_attention

        if "self_attention.pre_gate_norm.te_norm.weight" in name:
            return []
        if "self_attention.pre_gate_norm.weight" in name:
            return [("attention.g_norm.weight", self._convert_g_norm_weight(parameter))]

        name_mapping = {
            "self_attention.input_layernorm.weight": "input_layernorm.weight",
            "self_attention.linear_gate.weight": "attention.g_proj.weight",
            "self_attention.linear_proj.weight": "attention.dense.weight",
            "self_attention.q_layernorm.weight": "attention.query_layernorm.weight",
            "self_attention.k_layernorm.weight": "attention.key_layernorm.weight",
        }
        for src_name, target_name in name_mapping.items():
            if src_name in name:
                return [(target_name, parameter)]

        if "self_attention.linear_qkv.weight" in name:
            parameter = convert_qkv_weight_along_tp_attention(
                parameter,
                self.infer_atten_tp_size,
                self.tf_config,
                train_tp_rank=self.rank_info.attn_tp_rank,
                train_tp_size=self.rank_info.attn_tp_size,
            )
            return [("attention.query_key_value.weight", parameter)]

        return super()._convert_attention_param(name, parameter, layer_number)

    def _try_fuse_qkv_a_proj(
        self, target_name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        layer_cache = self.qkv_a_proj_cache.setdefault(layer_number, {})
        if target_name.endswith("kv_a_proj_with_mqa.weight"):
            cache_key = "kv_a_proj"
            other_key = "q_a_proj"
        elif target_name.endswith("q_a_proj.weight"):
            cache_key = "q_a_proj"
            other_key = "kv_a_proj"
        else:
            return [(target_name, parameter)]

        layer_cache[cache_key] = parameter
        if other_key not in layer_cache:
            return []

        fused_tensor = torch.cat(
            [layer_cache["q_a_proj"], layer_cache["kv_a_proj"]], dim=0
        )
        del self.qkv_a_proj_cache[layer_number]
        return [("attention.fused_qkv_a_proj_with_mqa.weight", fused_tensor)]

    def _convert_mla_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        from awex.converter.mcore_converter import get_full_tensor

        name_mapping = {
            "input_layernorm.weight": "input_layernorm.weight",
            "self_attention.linear_proj.weight": "attention.dense.weight",
            "self_attention.linear_q_proj.weight": "attention.q_proj.weight",
            "self_attention.linear_kv_down_proj.weight": "attention.kv_a_proj_with_mqa.weight",
            "self_attention.linear_q_down_proj.weight": "attention.q_a_proj.weight",
            "self_attention.linear_kv_up_proj.weight": "attention.kv_b_proj.weight",
            "self_attention.linear_kv_up_proj.layer_norm_weight": "attention.kv_a_layernorm.weight",
            "self_attention.linear_q_up_proj.weight": "attention.q_b_proj.weight",
            "self_attention.linear_q_up_proj.layer_norm_weight": "attention.q_a_layernorm.weight",
        }
        for src_name, target_name in name_mapping.items():
            if src_name not in name:
                continue
            if target_name in {
                "attention.kv_a_proj_with_mqa.weight",
                "attention.q_a_proj.weight",
            }:
                parameter = get_full_tensor(parameter, dim=0)
                if self.rank_info.tp_rank != 0:
                    return []
                if self.fuse_qkv_a_proj:
                    return self._try_fuse_qkv_a_proj(
                        target_name, parameter, layer_number
                    )
            return [(target_name, parameter)]

        return super()._convert_attention_param(name, parameter, layer_number)

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if self._is_linear_layer(int(layer_number)):
            return self._convert_lightning_attention_param(
                name, parameter, layer_number
            )
        return self._convert_mla_attention_param(name, parameter, layer_number)

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> List[Tuple[str, torch.Tensor]]:
        from awex.converter.mcore_converter import _process_mcore_pp_name

        name = name.replace("module.", "")
        name = _process_mcore_pp_name(
            name,
            self.rank_info,
            self.hf_config,
            self.tf_config,
            vp_stage=vp_stage,
            pp_stage_layer_id_map=self._pp_stage_layer_id_map,
        )
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        if name in direct_name_mapping:
            return [(direct_name_mapping[name], parameter)]
        if "output_layer.weight" in name:
            return self._post_process_linear_mla_params(
                self._convert_lm_head_param(name, parameter)
            )

        if not name.startswith("decoder.layers."):
            raise NotImplementedError(f"Unsupported parameter name: {name}")

        layer_number, remaining_name = name.replace("decoder.layers.", "", 1).split(
            ".",
            1,
        )

        if "self_attention" in remaining_name:
            converted = [
                (f"model.layers.{layer_number}.{param_name}", param)
                for param_name, param in self._convert_attention_param(
                    remaining_name, parameter, layer_number
                )
            ]
            return self._post_process_linear_mla_params(converted)

        if (
            "input_layernorm.weight" in remaining_name
            or "g_norm.weight" in remaining_name
        ):
            return self._post_process_linear_mla_params(
                [(f"model.layers.{layer_number}.{remaining_name}", parameter)]
            )

        if "mlp" in remaining_name:
            if "mlp.gate.weight" in name or "mlp.router.weight" in name:
                gate_name, gate_param = self._convert_gate(name, parameter)
                converted = [(f"model.layers.{layer_number}.{gate_name}", gate_param)]
            elif "expert_bias" in name:
                bias_name, bias_param = self._convert_expert_bias_param(
                    name, parameter, layer_number
                )
                converted = [(f"model.layers.{layer_number}.{bias_name}", bias_param)]
            else:
                converted = [
                    (f"model.layers.{layer_number}.{param_name}", param)
                    for param_name, param in self._convert_mlp_param(
                        remaining_name, parameter, layer_number
                    )
                ]
            return self._post_process_linear_mla_params(converted)

        raise NotImplementedError(f"Unsupported parameter name: {name}")


class LinearMLASGlangConverterMixin:
    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        base_name, has_scale_inv = normalize_scale_inv_name(name)
        direct_params = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "attention.g_norm.weight",
            "attention.query_layernorm.weight",
            "attention.key_layernorm.weight",
            "attention.kv_a_layernorm.weight",
            "attention.q_a_layernorm.weight",
        }
        if base_name in direct_params:
            return [(append_scale_inv(base_name, has_scale_inv), parameter)]
        return super()._convert_layer_norm_param(name, parameter, layer_number)

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        base_name, has_scale_inv = normalize_scale_inv_name(name)
        name_mapping = {
            "attention.g_proj.weight": "attention.g_proj.weight",
            "attention.q_proj.weight": "attention.q_proj.weight",
            "attention.k_proj.weight": "attention.k_proj.weight",
            "attention.v_proj.weight": "attention.v_proj.weight",
            "attention.dense.weight": "attention.dense.weight",
            "attention.o_proj.weight": "attention.dense.weight",
            "attention.g_norm.weight": "attention.g_norm.weight",
            "attention.query_layernorm.weight": "attention.query_layernorm.weight",
            "attention.key_layernorm.weight": "attention.key_layernorm.weight",
            "attention.kv_a_proj_with_mqa.weight": "attention.kv_a_proj_with_mqa.weight",
            "attention.kv_b_proj.weight": "attention.kv_b_proj.weight",
            "attention.kv_a_layernorm.weight": "attention.kv_a_layernorm.weight",
            "attention.q_b_proj.weight": "attention.q_b_proj.weight",
            "attention.q_a_layernorm.weight": "attention.q_a_layernorm.weight",
            "attention.q_a_proj.weight": "attention.q_a_proj.weight",
            "attention.fused_qkv_a_proj_with_mqa.weight": "attention.fused_qkv_a_proj_with_mqa.weight",
        }
        if base_name in name_mapping:
            target_name = name_mapping[base_name]
            return [(append_scale_inv(target_name, has_scale_inv), parameter)]

        if base_name in {
            "attention.query_key_value.weight",
            "self_attn.qkv_proj.weight",
        } and self._fuse_qkv(base_name):
            return [
                (
                    append_scale_inv("attention.query_key_value.weight", has_scale_inv),
                    parameter,
                )
            ]

        if base_name in {
            "attention.query_key_value.bias",
            "self_attn.qkv_proj.bias",
        } and self._fuse_qkv(base_name):
            return [
                (
                    append_scale_inv("attention.query_key_value.bias", has_scale_inv),
                    parameter,
                )
            ]

        if "o_proj" in base_name or "dense" in base_name:
            return [(append_scale_inv(base_name, has_scale_inv), parameter)]

        raise NotImplementedError(f"Unsupported attention parameter name: {name}")
