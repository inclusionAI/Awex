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

"""Qwen3-VL dense and MoE weight conversion support."""

import re
from types import SimpleNamespace
from typing import Any, List, Tuple

import torch

from awex.models.qwen3_moe import (
    SGlangToHFWeightConverterQwen3Moe,
    _build_mcore_converter_qwen3_moe,
)
from awex.sharding.param_sharding import (
    ShardingStrategy,
    ShardingType,
    get_default_sharding_dim,
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def get_qwen3_vl_text_config(config: Any) -> Any:
    """Return the text sub-config from a composite Qwen3-VL config."""
    return _config_value(config, "text_config", config)


def _uses_tied_embeddings(config: Any) -> bool:
    return bool(
        _config_value(get_qwen3_vl_text_config(config), "tie_word_embeddings", False)
    )


class Qwen3VLShardingStrategy(ShardingStrategy):
    """Describe the SGLang/Megatron TP layout of Qwen3-VL vision weights."""

    def _vision_tp_strategy(self, sharding_dim: int):
        if self.enable_dp_attention:
            tp_size = self.rank_info.attn_tp_size
            sharding_type = ShardingType.DP_TP_SHARDING
        else:
            tp_size = self.rank_info.tp_size
            sharding_type = ShardingType.TP_SHARDING
        if tp_size > 1:
            return sharding_type, sharding_dim, tp_size
        return ShardingType.NO_SHARDING, sharding_dim, 1

    def get_sharding_strategy(self, parameter_name: str, **kwargs):
        if not parameter_name.startswith("model.visual."):
            return super().get_sharding_strategy(parameter_name, **kwargs)

        sharding_dim = get_default_sharding_dim(parameter_name)
        replicated_suffixes = (
            ".bias",
            ".norm.weight",
            ".norm.bias",
            ".norm1.weight",
            ".norm1.bias",
            ".norm2.weight",
            ".norm2.bias",
        )
        if parameter_name.startswith("model.visual.patch_embed.") or (
            parameter_name.endswith(replicated_suffixes)
            and not parameter_name.endswith(("qkv.bias", "linear_fc1.bias"))
        ):
            return ShardingType.NO_SHARDING, sharding_dim, 1

        if parameter_name.endswith("pos_embed.weight"):
            # MBridge replicates the learned position embedding, while SGLang
            # partitions its runtime embedding table across vision TP ranks.
            if self.engine_name == "mcore":
                return ShardingType.NO_SHARDING, 0, 1
            return self._vision_tp_strategy(0)

        if parameter_name.endswith(
            (
                "attn.qkv.weight",
                "attn.qkv.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "merger.linear_fc1.weight",
                "merger.linear_fc1.bias",
            )
        ) or (
            ".deepstack_merger_list." in parameter_name
            and parameter_name.endswith(("linear_fc1.weight", "linear_fc1.bias"))
        ):
            return self._vision_tp_strategy(0)

        if parameter_name.endswith(
            (
                "attn.proj.weight",
                "mlp.linear_fc2.weight",
                "merger.linear_fc2.weight",
            )
        ) or (
            ".deepstack_merger_list." in parameter_name
            and parameter_name.endswith("linear_fc2.weight")
        ):
            return self._vision_tp_strategy(1)

        return ShardingType.NO_SHARDING, sharding_dim, 1


class Qwen3VLSGlangToHFWeightConverter(SGlangToHFWeightConverterQwen3Moe):
    """Reuse Qwen3 text conversion and add Qwen3-VL canonical namespaces."""

    def __init__(self, model_config, infer_engine_config, rank_info):
        self.vl_model_config = model_config
        super().__init__(
            get_qwen3_vl_text_config(model_config), infer_engine_config, rank_info
        )

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        if name.startswith("visual."):
            hf_name = f"model.{name}".replace(".attn.qkv_proj.", ".attn.qkv.")
            return [(hf_name, parameter)]

        converted = []
        for hf_name, hf_param in super().convert_param(name, parameter):
            if hf_name.startswith("model."):
                hf_name = hf_name.replace("model.", "model.language_model.", 1)
            converted.append((hf_name, hf_param))

        if name == "model.embed_tokens.weight" and _uses_tied_embeddings(
            self.vl_model_config
        ):
            converted.append(("lm_head.weight", parameter))
        return converted


_VISION_LAYER_PATTERN = re.compile(
    r"vision_model\.decoder\.layers\.(\d+)\.(.+)"
)
_VISION_DEEPSTACK_PATTERN = re.compile(
    r"vision_model\.decoder\.deepstack_merger_list\.(\d+)\.(.+)"
)
_VISION_DIRECT_NAMES = {
    "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
    "vision_model.merger.patch_norm.weight": "model.visual.merger.norm.weight",
    "vision_model.merger.patch_norm.bias": "model.visual.merger.norm.bias",
    "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
}


def _build_mcore_converter_qwen3_vl():
    # Keep Megatron imports lazy so inference-only workers do not apply patches.
    qwen3_mcore_converter = _build_mcore_converter_qwen3_moe()

    class Qwen3VLMcoreToHFWeightConverter(qwen3_mcore_converter):
        """Extend the Qwen3 converter with vision-tower parameter mappings."""

        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            self.vl_hf_config = hf_config
            super().__init__(
                get_qwen3_vl_text_config(hf_config),
                rank_info,
                infer_conf,
                tf_config,
            )

        def _convert_vision_qkv(
            self, layer_number: str, name: str, parameter: torch.Tensor
        ) -> List[Tuple[str, torch.Tensor]]:
            from awex.converter.mcore_converter import (
                convert_qkv_bias_along_tp_attention,
                convert_qkv_weight_along_tp_attention,
            )

            vision_config = _config_value(self.vl_hf_config, "vision_config")
            num_heads = _config_value(vision_config, "num_heads")
            hidden_size = _config_value(vision_config, "hidden_size")
            if num_heads is None or hidden_size is None:
                raise ValueError(
                    "Qwen3-VL vision_config.num_heads and hidden_size are required "
                    "for vision QKV conversion"
                )
            if int(hidden_size) % int(num_heads) != 0:
                raise ValueError(
                    "Qwen3-VL vision hidden_size must be divisible by num_heads: "
                    f"hidden_size={hidden_size}, num_heads={num_heads}"
                )

            vision_tf_config = SimpleNamespace(
                hidden_size=int(hidden_size),
                num_attention_heads=int(num_heads),
                num_query_groups=int(num_heads),
                kv_channels=int(hidden_size) // int(num_heads),
            )
            converter = (
                convert_qkv_weight_along_tp_attention
                if name.endswith("weight")
                else convert_qkv_bias_along_tp_attention
            )
            packed = converter(
                parameter,
                self.infer_atten_tp_size,
                vision_tf_config,
                train_tp_rank=int(self.rank_info.attn_tp_rank),
                train_tp_size=max(1, int(self.rank_info.attn_tp_size)),
            )
            kind = "weight" if name.endswith("weight") else "bias"
            return [(f"model.visual.blocks.{layer_number}.attn.qkv.{kind}", packed)]

        def _convert_vision_param(
            self, name: str, parameter: torch.Tensor
        ) -> List[Tuple[str, torch.Tensor]]:
            if name in _VISION_DIRECT_NAMES:
                return [(_VISION_DIRECT_NAMES[name], parameter)]

            match = _VISION_LAYER_PATTERN.fullmatch(name)
            if match is not None:
                layer_number, remaining_name = match.groups()
                base = f"model.visual.blocks.{layer_number}"
                if remaining_name in (
                    "self_attention.linear_qkv.weight",
                    "self_attention.linear_qkv.bias",
                ):
                    return self._convert_vision_qkv(
                        layer_number, remaining_name, parameter
                    )
                layer_names = {
                    "self_attention.linear_proj.weight": "attn.proj.weight",
                    "self_attention.linear_proj.bias": "attn.proj.bias",
                    "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
                    "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
                    "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
                    "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
                    "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
                    "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
                    "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
                    "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
                }
                if remaining_name in layer_names:
                    return [(f"{base}.{layer_names[remaining_name]}", parameter)]

            match = _VISION_DEEPSTACK_PATTERN.fullmatch(name)
            if match is not None:
                merger_number, remaining_name = match.groups()
                merger_names = {
                    "patch_norm.weight": "norm.weight",
                    "patch_norm.bias": "norm.bias",
                    "linear_fc1.weight": "linear_fc1.weight",
                    "linear_fc1.bias": "linear_fc1.bias",
                    "linear_fc2.weight": "linear_fc2.weight",
                    "linear_fc2.bias": "linear_fc2.bias",
                }
                if remaining_name in merger_names:
                    base = f"model.visual.deepstack_merger_list.{merger_number}"
                    return [(f"{base}.{merger_names[remaining_name]}", parameter)]

            raise ValueError(f"Unknown Qwen3-VL vision parameter: {name}")

        @torch.no_grad()
        def convert_param(
            self, name: str, parameter: torch.Tensor, vp_stage: int = None
        ) -> List[Tuple[str, torch.Tensor]]:
            name = name.replace("module.", "")
            if name.startswith("vision_model."):
                return self._convert_vision_param(name, parameter)
            if not name.startswith("language_model."):
                raise ValueError(f"Unknown Qwen3-VL parameter: {name}")

            language_name = name[len("language_model.") :]
            converted = []
            for hf_name, hf_param in super().convert_param(
                language_name, parameter, vp_stage=vp_stage
            ):
                if hf_name.startswith("model."):
                    hf_name = hf_name.replace(
                        "model.", "model.language_model.", 1
                    )
                converted.append((hf_name, hf_param))

            if (
                language_name == "embedding.word_embeddings.weight"
                and _uses_tied_embeddings(self.vl_hf_config)
            ):
                converted.append(("lm_head.weight", parameter))
            return converted

    return Qwen3VLMcoreToHFWeightConverter


CONFIG = [
    {
        "model_name": "Qwen3VLForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": _build_mcore_converter_qwen3_vl,
        "sglang_converter": Qwen3VLSGlangToHFWeightConverter,
    },
    {
        "model_name": "Qwen3VLMoeForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": _build_mcore_converter_qwen3_vl,
        "sglang_converter": Qwen3VLSGlangToHFWeightConverter,
    },
]
