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

"""Qwen3.5 dense/MoE and multimodal weight-update support.

Qwen3.5 combines Qwen3-style dense or expert MLPs with alternating full
attention and Gated DeltaNet layers.  Its full-attention query projection also
contains an output gate, so the Megatron ``linear_qkv`` tensor cannot be
handled by the ordinary Qwen3 GQA splitter.  This module keeps the existing
Qwen3 converters as the common path and only specializes the hybrid-attention
and vision layouts.

The canonical names used by Awex follow the Hugging Face checkpoint namespace:
``model.language_model.*`` for multimodal text weights and ``model.visual.*``
for the vision tower.  Checkpoints may also expose top-level ``mtp.*`` weights,
but RL training does not optimize the draft head, so both converter ingress
paths deliberately filter those parameters before metadata and communication
plans are built.  Dense and MoE architectures intentionally share this file;
the inherited Qwen3 MLP conversion already distinguishes dense,
routed-expert, and shared-expert parameters from their names.
"""

from types import SimpleNamespace
from typing import List, Tuple

import torch

from awex.models.qwen3_moe import (
    SGlangToHFWeightConverterQwen3Moe,
    _build_mcore_converter_qwen3_moe,
)
from awex.models.qwen3_vl import Qwen3VLShardingStrategy
from awex.sharding.param_sharding import ShardingType, get_default_sharding_dim


class _Qwen3_5Layout:
    """Tensor-layout operations shared by the train-side converter.

    These methods operate only on tensors and model geometry.  Keeping them in
    one helper class makes the Qwen3.5-specific packing rules explicit without
    adding another set of generic converter functions to the package.
    """

    @staticmethod
    def text_config(config):
        """Return the language config from either text-only or VLM config."""
        return getattr(config, "text_config", config)

    @classmethod
    def head_dim(cls, config) -> int:
        """Resolve the explicit Qwen3.5 head width with a safe fallback."""
        config = cls.text_config(config)
        value = getattr(config, "head_dim", None)
        if value:
            return int(value)
        return int(config.hidden_size // config.num_attention_heads)

    @staticmethod
    def split_rows(
        tensor: torch.Tensor, parts: int, description: str
    ) -> Tuple[torch.Tensor, ...]:
        """Split dim 0 evenly and fail with geometry-specific context."""
        if parts <= 0 or tensor.shape[0] % parts != 0:
            raise ValueError(
                f"{description} rows ({tensor.shape[0]}) must be divisible by "
                f"parallel size ({parts})"
            )
        return tuple(torch.chunk(tensor, parts, dim=0))

    @classmethod
    def pack_output_gated_qkv(
        cls, parameter: torch.Tensor, config, infer_tp_size: int
    ) -> torch.Tensor:
        """Repack Megatron gated GQA into SGLang rank-contiguous QKV blocks.

        Megatron stores one block per KV group as ``[Q, gate, K, V]``.  SGLang
        stores ``[Q-with-gate, K, V]`` for each inference TP rank and replicates
        K/V heads when TP is wider than the number of KV heads.  The returned
        tensor concatenates those rank blocks so Awex can reshard it between
        arbitrary training and inference TP degrees.
        """
        config = cls.text_config(config)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = cls.head_dim(config)
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"attention heads ({num_heads}) must be divisible by KV heads "
                f"({num_kv_heads})"
            )
        if num_heads % infer_tp_size != 0:
            raise ValueError(
                f"attention heads ({num_heads}) must be divisible by inference "
                f"TP ({infer_tp_size})"
            )

        queries_per_group = num_heads // num_kv_heads
        query_rows = queries_per_group * head_dim
        group_rows = 2 * query_rows + 2 * head_dim
        expected_rows = num_kv_heads * group_rows
        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "unexpected gated QKV rows: "
                f"got {parameter.shape[0]}, expected {expected_rows}"
            )

        tail = parameter.shape[1:]
        groups = parameter.reshape(num_kv_heads, group_rows, *tail)
        query = groups[:, :query_rows].reshape(num_heads, head_dim, *tail)
        gate = groups[:, query_rows : 2 * query_rows].reshape(
            num_heads, head_dim, *tail
        )
        query = torch.cat((query, gate), dim=1)
        key = groups[:, 2 * query_rows : 2 * query_rows + head_dim].reshape(
            num_kv_heads, head_dim, *tail
        )
        value = groups[:, 2 * query_rows + head_dim :].reshape(
            num_kv_heads, head_dim, *tail
        )

        query_shards = cls.split_rows(query, infer_tp_size, "query projection")
        if infer_tp_size >= num_kv_heads:
            if infer_tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"inference TP ({infer_tp_size}) must be divisible by KV "
                    f"heads ({num_kv_heads})"
                )
            replicas = infer_tp_size // num_kv_heads
            key_shards = [head for head in key for _ in range(replicas)]
            value_shards = [head for head in value for _ in range(replicas)]
        else:
            if num_kv_heads % infer_tp_size != 0:
                raise ValueError(
                    f"KV heads ({num_kv_heads}) must be divisible by inference "
                    f"TP ({infer_tp_size})"
                )
            key_shards = cls.split_rows(key, infer_tp_size, "key projection")
            value_shards = cls.split_rows(value, infer_tp_size, "value projection")

        blocks = []
        for query_shard, key_shard, value_shard in zip(
            query_shards, key_shards, value_shards
        ):
            blocks.append(
                torch.cat(
                    (
                        query_shard.reshape(-1, *tail),
                        key_shard.reshape(-1, *tail),
                        value_shard.reshape(-1, *tail),
                    ),
                    dim=0,
                )
            )
        return torch.cat(blocks, dim=0).contiguous()

    @classmethod
    def split_gdn_input(
        cls, parameter: torch.Tensor, config, train_tp_size: int
    ) -> Tuple[torch.Tensor, ...]:
        """Undo Megatron's rank-local ``[Q,K,V,Z,B,A]`` GDN packing."""
        config = cls.text_config(config)
        qk_dim = int(config.linear_num_key_heads * config.linear_key_head_dim)
        value_dim = int(config.linear_num_value_heads * config.linear_value_head_dim)
        value_heads = int(config.linear_num_value_heads)
        dimensions = (qk_dim, qk_dim, value_dim, value_dim, value_heads, value_heads)
        if any(size % train_tp_size != 0 for size in dimensions):
            raise ValueError("GDN input dimensions must be divisible by training TP")

        local_sizes = tuple(size // train_tp_size for size in dimensions)
        rows_per_rank = sum(local_sizes)
        if parameter.shape[0] != rows_per_rank * train_tp_size:
            raise ValueError(
                "unexpected GDN input rows: "
                f"got {parameter.shape[0]}, expected {rows_per_rank * train_tp_size}"
            )
        rank_blocks = parameter.reshape(
            train_tp_size, rows_per_rank, *parameter.shape[1:]
        )
        pieces = [[] for _ in local_sizes]
        for rank_block in rank_blocks:
            for target, piece in zip(
                pieces, torch.split(rank_block, local_sizes, dim=0)
            ):
                target.append(piece)
        return tuple(torch.cat(part, dim=0) for part in pieces)

    @classmethod
    def pack_gdn_input(
        cls,
        parameter: torch.Tensor,
        config,
        train_tp_size: int,
        infer_tp_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build SGLang's fused QKVZ and BA tensors in inference-rank order."""
        categories = cls.split_gdn_input(parameter, config, train_tp_size)
        shards = [
            cls.split_rows(tensor, infer_tp_size, f"GDN input category {index}")
            for index, tensor in enumerate(categories)
        ]
        qkvz = [
            torch.cat([shards[index][rank] for index in range(4)], dim=0)
            for rank in range(infer_tp_size)
        ]
        ba = [
            torch.cat([shards[index][rank] for index in range(4, 6)], dim=0)
            for rank in range(infer_tp_size)
        ]
        return torch.cat(qkvz, dim=0), torch.cat(ba, dim=0)

    @classmethod
    def pack_gdn_conv(
        cls,
        parameter: torch.Tensor,
        config,
        train_tp_size: int,
        infer_tp_size: int,
    ) -> torch.Tensor:
        """Reorder Megatron's rank-local Q/K/V convolution channels for SGLang."""
        config = cls.text_config(config)
        qk_dim = int(config.linear_num_key_heads * config.linear_key_head_dim)
        value_dim = int(config.linear_num_value_heads * config.linear_value_head_dim)
        if qk_dim % train_tp_size or value_dim % train_tp_size:
            raise ValueError("GDN convolution dimensions must divide training TP")
        local_qk = qk_dim // train_tp_size
        local_value = value_dim // train_tp_size
        rows_per_rank = 2 * local_qk + local_value
        if parameter.shape[0] != rows_per_rank * train_tp_size:
            raise ValueError("unexpected GDN convolution rows")

        query_parts, key_parts, value_parts = [], [], []
        for block in parameter.reshape(
            train_tp_size, rows_per_rank, *parameter.shape[1:]
        ):
            query, key, value = torch.split(
                block, (local_qk, local_qk, local_value), dim=0
            )
            query_parts.append(query)
            key_parts.append(key)
            value_parts.append(value)
        categories = (
            torch.cat(query_parts),
            torch.cat(key_parts),
            torch.cat(value_parts),
        )
        shards = [
            cls.split_rows(tensor, infer_tp_size, f"GDN conv category {index}")
            for index, tensor in enumerate(categories)
        ]
        return torch.cat(
            [
                torch.cat([shards[index][rank] for index in range(3)], dim=0)
                for rank in range(infer_tp_size)
            ],
            dim=0,
        )


class Qwen3_5ShardingStrategy(Qwen3VLShardingStrategy):
    """TP/EP rules for Qwen3.5 language and vision parameters.

    Pipeline parallelism is represented by parameter ownership in Awex rather
    than another sharding enum, so this class only declares the tensor or expert
    dimensions within the owning PP stage.  Vision rules come from Qwen3-VL;
    MTP parameters never reach this strategy because the converters filter
    them before transfer metadata is constructed.
    """

    def get_sharding_strategy(self, parameter_name: str, **kwargs):
        if ".linear_attn." in parameter_name:
            if ".norm." in parameter_name:
                return ShardingType.NO_SHARDING, 0, 1
            sharding_dim = 1 if parameter_name.endswith("out_proj.weight") else 0
            return self.get_attention_sharding_strategy(
                parameter_name, sharding_dim=sharding_dim, **kwargs
            )
        if ".self_attn." in parameter_name:
            if "_norm." in parameter_name:
                return ShardingType.NO_SHARDING, 0, 1
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)
        if "shared_expert_gate" in parameter_name:
            return ShardingType.NO_SHARDING, 0, 1
        if "shared_expert" in parameter_name:
            return self.get_shared_expert_sharding_strategy(parameter_name, **kwargs)
        if ".experts." in parameter_name:
            # The base strategy returns early when dense TP is one.  EP is an
            # independent axis and must still describe expert ownership.
            return self.get_expert_sharding_strategy(parameter_name, **kwargs)
        return super().get_sharding_strategy(parameter_name, **kwargs)

    def get_attention_sharding_strategy(self, parameter_name, **kwargs):
        """Honor an explicit hybrid-attention dimension when one is supplied."""
        explicit_dim = kwargs.pop("sharding_dim", None)
        if explicit_dim is None:
            return super().get_attention_sharding_strategy(parameter_name, **kwargs)
        if self.enable_dp_attention:
            size = self.rank_info.attn_tp_size
            kind = ShardingType.DP_TP_SHARDING
        else:
            size = self.rank_info.tp_size
            kind = ShardingType.TP_SHARDING
        if size <= 1:
            return ShardingType.NO_SHARDING, explicit_dim, 1
        return kind, explicit_dim, size

    def get_shared_expert_sharding_strategy(self, parameter_name, **kwargs):
        """Keep the standalone shared MLP on dense TP rather than routed EP."""
        sharding_dim = get_default_sharding_dim(parameter_name)
        if self.tp_size > 1:
            return ShardingType.TP_SHARDING, sharding_dim, self.tp_size
        return ShardingType.NO_SHARDING, sharding_dim, 1


class SGlangToHFWeightConverterQwen3_5(SGlangToHFWeightConverterQwen3Moe):
    """Normalize current SGLang Qwen3.5 target-model parameters.

    SGLang removes ``self_attn`` from full-attention module names and fuses the
    gated query/K/V weights into ``qkv_proj``.  The converter restores a stable
    checkpoint-like namespace while deliberately keeping that fused tensor;
    the Megatron converter emits the same layout, avoiding a duplicate gated-
    QKV split/repack path in the transfer planner.
    """

    def __init__(self, model_config, infer_engine_config, rank_info):
        self.root_config = model_config
        self.is_multimodal = hasattr(model_config, "text_config")
        super().__init__(
            _Qwen3_5Layout.text_config(model_config),
            infer_engine_config,
            rank_info,
        )

    def _fuse_qkv(self, name: str) -> bool:
        return True

    @staticmethod
    def _normalize_shared_expert(name: str) -> str:
        return name.replace(".shared_experts.", ".shared_expert.")

    def _canonical_main_name(self, name: str) -> str:
        if self.is_multimodal and name.startswith("model."):
            return name.replace("model.", "model.language_model.", 1)
        return name

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        name = name.replace("model.language_model.", "model.")
        if name.startswith(("mtp.", "model.mtp.")):
            # MTP is loaded with the inference model but remains frozen during
            # RL.  Returning no canonical parameters keeps it out of infer
            # metadata and therefore out of the generated transfer plan.
            return []
        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)
        if name.startswith("visual."):
            name = name.replace(".attn.qkv_proj.", ".attn.qkv.")
            return [(f"model.{name}", parameter)]
        parts = name.split(".")
        if len(parts) >= 5 and parts[:2] == ["model", "layers"]:
            layer_number = parts[2]
            remaining_name = ".".join(parts[3:])
            if remaining_name.startswith(
                ("qkv_proj.", "o_proj.", "q_norm.", "k_norm.")
            ):
                canonical = f"model.layers.{layer_number}.self_attn.{remaining_name}"
                return [(self._canonical_main_name(canonical), parameter)]
            if remaining_name.startswith("linear_attn.") or remaining_name.startswith(
                "mlp.shared_expert_gate."
            ):
                canonical = self._normalize_shared_expert(name)
                return [(self._canonical_main_name(canonical), parameter)]

        converted = []
        for converted_name, converted_parameter in super().convert_param(
            name, parameter
        ):
            converted_name = self._normalize_shared_expert(converted_name)
            converted.append(
                (
                    self._canonical_main_name(converted_name),
                    converted_parameter,
                )
            )
        return converted


class _Qwen3_5McoreConverterFactory:
    """Lazily build the Megatron converter without eager Megatron imports."""

    def __call__(self):
        base_converter = _build_mcore_converter_qwen3_moe()

        class McoreToHFWeightConverterQwen3_5(base_converter):
            """Convert trainable Megatron Qwen3.5 language and vision weights.

            Decoder PP layer ids continue through the existing Qwen3 converter,
            and expert id expansion plus EP offsets are inherited from the
            Qwen3 MoE path.  Megatron checkpoints can expose an ``mtp`` subtree,
            but RL leaves it frozen; ``convert_param`` returns an empty list for
            that subtree so it is absent from metadata, transfer planning, and
            each subsequent update.
            """

            _vision_direct_mapping = {
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
            _vision_layer_mapping = {
                "self_attention.linear_qkv.weight": "attn.qkv.weight",
                "self_attention.linear_qkv.bias": "attn.qkv.bias",
                "self_attention.linear_proj.weight": "attn.proj.weight",
                "self_attention.linear_proj.bias": "attn.proj.bias",
                "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
                "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
                "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
                "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
                "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
            }

            def __init__(self, hf_config, rank_info, infer_conf, tf_config):
                self.root_config = hf_config
                self.vision_config = getattr(hf_config, "vision_config", None)
                self.is_multimodal = hasattr(hf_config, "text_config")
                super().__init__(
                    _Qwen3_5Layout.text_config(hf_config),
                    rank_info,
                    infer_conf,
                    tf_config=tf_config,
                )

            def _full_tp_tensor(self, parameter: torch.Tensor) -> torch.Tensor:
                if int(self.rank_info.attn_tp_size) <= 1:
                    return parameter
                from awex.converter.mcore_converter import get_full_tensor

                return get_full_tensor(parameter, dim=0)

            def _take_train_tp_shard(self, parameter: torch.Tensor) -> torch.Tensor:
                train_tp_size = max(1, int(self.rank_info.attn_tp_size))
                if train_tp_size == 1:
                    return parameter
                shards = _Qwen3_5Layout.split_rows(
                    parameter, train_tp_size, "converted parameter"
                )
                return shards[int(self.rank_info.attn_tp_rank)].contiguous()

            def _convert_attention_param(self, name, parameter, layer_number):
                train_tp_size = max(1, int(self.rank_info.attn_tp_size))
                infer_tp_size = max(1, int(self.infer_atten_tp_size))
                if "self_attention.linear_qkv.weight" in name or (
                    "self_attention.linear_qkv.bias" in name
                ):
                    suffix = "weight" if name.endswith("weight") else "bias"
                    packed = _Qwen3_5Layout.pack_output_gated_qkv(
                        self._full_tp_tensor(parameter),
                        self.hf_config,
                        infer_tp_size,
                    )
                    return [
                        (
                            f"self_attn.qkv_proj.{suffix}",
                            self._take_train_tp_shard(packed),
                        )
                    ]
                if "self_attention.in_proj.layer_norm_weight" in name:
                    return [("input_layernorm.weight", parameter)]
                if "self_attention.in_proj.weight" in name:
                    qkvz, ba = _Qwen3_5Layout.pack_gdn_input(
                        self._full_tp_tensor(parameter),
                        self.hf_config,
                        train_tp_size,
                        infer_tp_size,
                    )
                    return [
                        (
                            "linear_attn.in_proj_qkvz.weight",
                            self._take_train_tp_shard(qkvz),
                        ),
                        (
                            "linear_attn.in_proj_ba.weight",
                            self._take_train_tp_shard(ba),
                        ),
                    ]
                if "self_attention.conv1d.weight" in name:
                    packed = _Qwen3_5Layout.pack_gdn_conv(
                        self._full_tp_tensor(parameter),
                        self.hf_config,
                        train_tp_size,
                        infer_tp_size,
                    )
                    return [
                        (
                            "linear_attn.conv1d.weight",
                            self._take_train_tp_shard(packed),
                        )
                    ]
                if "self_attention.out_proj.weight" in name:
                    return [("linear_attn.out_proj.weight", parameter)]
                if "self_attention.out_norm.weight" in name:
                    return [("linear_attn.norm.weight", parameter + 1.0)]
                if "self_attention.A_log" in name:
                    return [("linear_attn.A_log", parameter)]
                if "self_attention.dt_bias" in name:
                    return [("linear_attn.dt_bias", parameter)]
                return super()._convert_attention_param(name, parameter, layer_number)

            def _convert_mlp_param(self, name, parameter, layer_number):
                return [
                    (
                        converted_name.replace(".shared_experts.", ".shared_expert."),
                        converted_parameter,
                    )
                    for converted_name, converted_parameter in super()._convert_mlp_param(
                        name, parameter, layer_number
                    )
                ]

            def _convert_visual_param(self, name, parameter):
                target = self._vision_direct_mapping.get(name)
                if target is not None:
                    return [(target, parameter)]
                layer_prefix = "vision_model.decoder.layers."
                if name.startswith(layer_prefix):
                    layer_number, suffix = name[len(layer_prefix) :].split(".", 1)
                    target_suffix = self._vision_layer_mapping.get(suffix)
                    if target_suffix is None:
                        raise NotImplementedError(
                            f"Unsupported Qwen3.5 vision layer parameter: {name}"
                        )
                    if suffix in {
                        "self_attention.linear_qkv.weight",
                        "self_attention.linear_qkv.bias",
                    }:
                        if self.vision_config is None:
                            raise ValueError(
                                "vision_config is required for Qwen3.5 visual QKV"
                            )
                        from awex.converter.mcore_converter import (
                            convert_qkv_bias_along_tp_attention,
                            convert_qkv_weight_along_tp_attention,
                        )

                        vision_config = SimpleNamespace(
                            hidden_size=int(self.vision_config.hidden_size),
                            num_attention_heads=int(self.vision_config.num_heads),
                            num_query_groups=int(self.vision_config.num_heads),
                            kv_channels=int(self.vision_config.hidden_size)
                            // int(self.vision_config.num_heads),
                        )
                        converter = (
                            convert_qkv_weight_along_tp_attention
                            if suffix.endswith("weight")
                            else convert_qkv_bias_along_tp_attention
                        )
                        parameter = converter(
                            parameter,
                            self.infer_atten_tp_size,
                            vision_config,
                            train_tp_rank=int(self.rank_info.attn_tp_rank),
                            train_tp_size=max(1, int(self.rank_info.attn_tp_size)),
                        )
                    return [
                        (
                            f"model.visual.blocks.{layer_number}.{target_suffix}",
                            parameter,
                        )
                    ]
                raise NotImplementedError(
                    f"Unsupported Qwen3.5 vision parameter: {name}"
                )

            @torch.no_grad()
            def convert_param(self, name, parameter, vp_stage=None):
                name = name.replace("module.", "")
                if name.startswith(("language_model.mtp.", "model.mtp.", "mtp.")):
                    return []
                if name.startswith("vision_model."):
                    return self._convert_visual_param(name, parameter)

                if name.startswith("language_model."):
                    name = name[len("language_model.") :]
                converted = super().convert_param(name, parameter, vp_stage=vp_stage)
                if not self.is_multimodal:
                    return converted
                return [
                    (
                        converted_name.replace("model.", "model.language_model.", 1)
                        if converted_name.startswith("model.")
                        else converted_name,
                        converted_parameter,
                    )
                    for converted_name, converted_parameter in converted
                ]

        return McoreToHFWeightConverterQwen3_5


_MCORE_CONVERTER_FACTORY = _Qwen3_5McoreConverterFactory()

CONFIG = [
    {
        "model_name": "Qwen3_5ForCausalLM",
        "sharding_strategy": Qwen3_5ShardingStrategy,
        "mcore_converter": _MCORE_CONVERTER_FACTORY,
        "sglang_converter": SGlangToHFWeightConverterQwen3_5,
    },
    {
        "model_name": "Qwen3_5MoeForCausalLM",
        "sharding_strategy": Qwen3_5ShardingStrategy,
        "mcore_converter": _MCORE_CONVERTER_FACTORY,
        "sglang_converter": SGlangToHFWeightConverterQwen3_5,
    },
    {
        "model_name": "Qwen3_5ForConditionalGeneration",
        "sharding_strategy": Qwen3_5ShardingStrategy,
        "mcore_converter": _MCORE_CONVERTER_FACTORY,
        "sglang_converter": SGlangToHFWeightConverterQwen3_5,
    },
    {
        "model_name": "Qwen3_5MoeForConditionalGeneration",
        "sharding_strategy": Qwen3_5ShardingStrategy,
        "mcore_converter": _MCORE_CONVERTER_FACTORY,
        "sglang_converter": SGlangToHFWeightConverterQwen3_5,
    },
]
