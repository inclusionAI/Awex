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

from typing import List, Tuple

import torch
from transformers import PretrainedConfig


# all sglang related imports must be local imports to avoid import error if
# users use other engine.
class SGlangToHFWeightConverter:
    def __init__(
        self,
        model_config: PretrainedConfig,
        infer_engine_config,
        rank_info,
    ):
        self.model_config = model_config
        self.total_num_heads = model_config.num_attention_heads
        self.total_kv_heads = model_config.num_key_value_heads
        self.infer_engine_config = infer_engine_config
        self.rank_info = rank_info
        self.tp_size = infer_engine_config.tp_size
        self.tp_rank = self.rank_info.tp_rank
        self.ep_size = infer_engine_config.ep_size
        self.ep_rank = self.rank_info.ep_rank

    def _fuse_qkv(self, name: str) -> bool:
        """Override this method to control QKV fusion behavior"""
        # Megatron QKV is alternately packed, [Q, Q, Q, K, V, Q, Q, Q, K, V, Q, Q, Q, K, V]
        # SGlang QKV is packed differently, [Q, Q, Q, Q, Q, Q, K, K, V, V, V, V]
        # to support fuse QKV, we need more shoshicated methods for resharding, especally when both use TP
        # If megatron use DP only for QKV, then we only need to reorder the QKV weights, and then fuse them
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        """Override this method to control gate_up projection fusion behavior"""
        # sglang use MergedColumnParallelLinear for gate_up_proj:
        # if megatron use tp size 1, then we need to rearange the weights to tp foramt:
        # [Gate, Gate, Gate, UpProj, UpProj, UpProj] -> [Gate, UpProj, Gate, UpProj, Gate, UpProj]
        # sender must know the tp size of inference, and then rearange the weights to tp format
        # if megatron use tp size > 1, it will be more complex, so we don't support it for now
        return False

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert attention parameters from SGlang to HuggingFace format"""
        if "qkv_proj" in name or "query_key_value" in name:
            if self._fuse_qkv(name):
                # Keep fused format
                return [(name, parameter)]
            else:
                # Split into separate Q, K, V projections
                shape0 = parameter.shape[0]
                stride = shape0 // 3
                return [
                    (
                        name.replace("qkv_proj", "q_proj").replace(
                            "query_key_value", "q_proj"
                        ),
                        parameter.narrow(0, 0, stride),
                    ),
                    (
                        name.replace("qkv_proj", "k_proj").replace(
                            "query_key_value", "k_proj"
                        ),
                        parameter.narrow(0, stride, stride),
                    ),
                    (
                        name.replace("qkv_proj", "v_proj").replace(
                            "query_key_value", "v_proj"
                        ),
                        parameter.narrow(0, 2 * stride, stride),
                    ),
                ]
        elif "o_proj" in name or "dense" in name:
            return [(name, parameter)]
        else:
            raise NotImplementedError(f"Unsupported attention parameter name: {name}")

    def _convert_mlp_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert MLP parameters from SGlang to HuggingFace format
        Input name example:
            mlp.gate_up_proj.weight
            mlp.down_proj.weight
            mlp.w13_weight
            mlp.w2_weight
        Return example:
            mlp.gate_up_proj.weight
            mlp.down_proj.weight
        """
        if "gate_up_proj" in name:
            if self._fuse_gate_up_proj(name):
                # Keep fused format
                return [(name, parameter)]
            else:
                # Split into separate gate and up projections
                shape0 = parameter.shape[0]
                stride = shape0 // 2
                return [
                    (
                        name.replace("gate_up_proj", "gate_proj"),
                        parameter.narrow(0, 0, stride),
                    ),
                    (
                        name.replace("gate_up_proj", "up_proj"),
                        parameter.narrow(0, stride, stride),
                    ),
                ]
        elif "down_proj" in name:
            return [(name, parameter)]
        elif "w13_weight" in name:
            # Handle MoE expert weights (w13_weight contains both gate and up projections)
            if self._fuse_gate_up_proj(name):
                return [(name.replace("w13_weight", "gate_up_proj.weight"), parameter)]
            else:
                shape0 = parameter.shape[0]
                stride = shape0 // 2
                return [
                    (
                        name.replace("w13_weight", "gate_proj.weight"),
                        parameter.narrow(0, 0, stride),
                    ),
                    (
                        name.replace("w13_weight", "up_proj.weight"),
                        parameter.narrow(0, stride, stride),
                    ),
                ]
        elif "w2_weight" in name:
            # Handle MoE expert down projection
            return [(name.replace("w2_weight", "down_proj.weight"), parameter)]
        else:
            raise NotImplementedError(f"Unsupported MLP parameter name: {name}")

    def _convert_expert_tp_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert expert parameters from SGlang to HuggingFace format.

        Input name example:
            mlp.experts.w13_weight
            mlp.experts.w2_weight
            mlp.shared_experts.gate_up_weight
            mlp.shared_experts.down_weight
        Return example:
            mlp.experts.62.down_proj.weight
            mlp.experts.62.gate_up_proj.weight
            mlp.shared_experts.gate_up_proj.weight
            mlp.shared_experts.down_proj.weight
        """
        converted_params = []
        # Get number of router experts from config
        num_router_experts = (
            getattr(self.model_config, "num_experts", None)
            or self.model_config.n_routed_experts
        )
        if "expert_bias" in name:
            return [(name, parameter)]
        if "shared_experts" in name:
            # shared_experts not fused with normal experts
            return self._convert_mlp_param(name, parameter, layer_number)
        for expert_id in range(parameter.shape[0]):
            expert_parameter = parameter[expert_id]
            # Determine expert type and construct parameter name
            if expert_id >= num_router_experts:
                # Shared experts
                num_shared_experts = parameter.shape[0] - num_router_experts
                shared_expert_id = expert_id - num_router_experts
                if num_shared_experts > 1:
                    param_name = name.replace(
                        "experts", f"shared_experts.{shared_expert_id}"
                    )
                else:
                    param_name = name.replace("experts", "shared_experts")
            else:
                # Normal experts
                param_name = name.replace("experts", f"experts.{expert_id}")

            for converted_name, param_tensor in self._convert_mlp_param(
                param_name, expert_parameter, layer_number
            ):
                converted_params.append((converted_name, param_tensor))
        return converted_params

    def _convert_expert_moe_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert expert parameters from SGlang to HuggingFace format."""
        # w13_weight shape: num_experts_per_partition, 2 * intermediate_size, hidden_size
        # w2_weight shape: num_experts_per_partition, hidden_size, intermediate_size
        if "expert_bias" in name:
            return [(name, parameter)]
        converted_params = []
        num_local_experts = parameter.shape[0]
        for i in range(num_local_experts):
            expert_id = i + self.ep_rank * num_local_experts
            expert_parameter = parameter[i]
            if "w13_weight" in name:
                for converted_name, param in self._convert_mlp_param(
                    name, expert_parameter, layer_number
                ):
                    # mlp.experts.63.gate_up_proj.weight
                    # mlp.experts.63.gate_proj.weight
                    # mlp.experts.63.up_proj.weight
                    updated_name = converted_name.replace(
                        "mlp.", f"mlp.experts.{expert_id}."
                    )
                    converted_params.append((updated_name, param))
            elif "w2_weight" in name:
                for converted_name, param in self._convert_mlp_param(
                    name, expert_parameter, layer_number
                ):
                    # mlp.experts.63.down_proj.weight
                    updated_name = converted_name.replace(
                        "mlp.", f"mlp.experts.{expert_id}."
                    )
                    converted_params.append((updated_name, param))
            else:
                raise NotImplementedError(f"Unsupported expert parameter name: {name}")
        return converted_params

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert layer normalization parameters"""
        if "input_layernorm" in name:
            return [(name, parameter)]
        elif "post_attention_layernorm" in name:
            return [(name, parameter)]
        elif "query_layernorm" in name:
            return [(name, parameter)]
        elif "key_layernorm" in name:
            return [(name, parameter)]
        else:
            raise NotImplementedError(f"Unsupported layer norm parameter name: {name}")

    @torch.no_grad()
    def convert_param(
        self,
        name: str,
        parameter: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert a parameter from SGlang format to HuggingFace format"""
        # Handle direct name mappings first
        direct_name_mapping = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
            "model.lm_head.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [(direct_name_mapping[name], parameter)]

        # Handle layer-specific parameters
        if "model.layers." in name:
            # Extract layer number and remaining name
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_idx = parts[i + 1]
                    break

            if layer_idx is None:
                raise ValueError(
                    f"Could not extract layer number from parameter name: {name}"
                )

            # Reconstruct the remaining name after layer
            remaining_parts = parts[parts.index("layers") + 2 :]
            remaining_name = ".".join(remaining_parts)
            # Route to appropriate conversion method
            if "layernorm" in remaining_name or "norm" in remaining_name:
                converted_params = self._convert_layer_norm_param(
                    remaining_name, parameter, layer_idx
                )
                return [
                    (f"model.layers.{layer_idx}.{param_name}", param)
                    for param_name, param in converted_params
                ]
            elif "attention" in remaining_name or "self_attn" in remaining_name:
                converted_params = self._convert_attention_param(
                    remaining_name, parameter, layer_idx
                )
                return [
                    (f"model.layers.{layer_idx}.{param_name}", param)
                    for param_name, param in converted_params
                ]
            elif "mlp" in remaining_name:
                if "gate.weight" in remaining_name:
                    return [(name, parameter)]
                elif "router.weight" in remaining_name:
                    return [(f"model.layers.{layer_idx}.mlp.gate.weight", parameter)]

                # Check if this is an expert parameter
                if ".expert" in remaining_name or "experts." in remaining_name:
                    if self.ep_size == 1:
                        converted_params = self._convert_expert_tp_param(
                            remaining_name, parameter, layer_idx
                        )
                    else:
                        converted_params = self._convert_expert_moe_param(
                            remaining_name, parameter, layer_idx
                        )
                    # name example: model.layers.3.mlp.experts.62.down_proj.weight
                    return [
                        (f"model.layers.{layer_idx}.{param_name}", param)
                        for param_name, param in converted_params
                    ]
                else:
                    converted_params = self._convert_mlp_param(
                        remaining_name, parameter, layer_idx
                    )
                    return [
                        (f"model.layers.{layer_idx}.{param_name}", param)
                        for param_name, param in converted_params
                    ]
            else:
                # For other parameters, keep as is
                return [(name, parameter)]
        else:
            # For non-layer parameters, keep as is
            return [(name, parameter)]
