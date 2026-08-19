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

"""CPU regression tests for Qwen3.5 dense/MoE and VLM conversion."""

from types import SimpleNamespace

import pytest
import torch

from awex.models.qwen3_5 import (
    _MCORE_CONVERTER_FACTORY,
    CONFIG,
    Qwen3_5ShardingStrategy,
    SGlangToHFWeightConverterQwen3_5,
    _Qwen3_5Layout,
)
from awex.models.registry import get_infer_weights_converter
from awex.sharding.param_sharding import ShardingType


@pytest.fixture
def text_config():
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        architectures=["Qwen3_5MoeForCausalLM"],
        hidden_size=8,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
        num_experts=4,
        tie_word_embeddings=False,
    )


@pytest.fixture
def vl_config(text_config):
    return SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text_config,
        vision_config=SimpleNamespace(num_heads=2, hidden_size=4),
        tie_word_embeddings=False,
    )


@pytest.fixture
def tf_config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        num_layers=4,
    )


@pytest.fixture
def rank_info():
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
    )


@pytest.fixture
def infer_engine_config():
    return SimpleNamespace(tp_size=1, ep_size=1, device_backend="cpu")


def make_train_converter(config, rank_info, tf_config, **infer_conf):
    converter_class = _MCORE_CONVERTER_FACTORY()
    return converter_class(
        config,
        rank_info,
        {"infer_atten_tp_size": 1, **infer_conf},
        tf_config,
    )


def make_gated_qkv(text_config):
    groups = []
    queries_per_group = (
        text_config.num_attention_heads // text_config.num_key_value_heads
    )
    query_rows = queries_per_group * text_config.head_dim
    for group in range(text_config.num_key_value_heads):
        groups.extend(
            [
                torch.full((query_rows, text_config.hidden_size), 10 + group),
                torch.full((query_rows, text_config.hidden_size), 20 + group),
                torch.full((text_config.head_dim, text_config.hidden_size), 30 + group),
                torch.full((text_config.head_dim, text_config.hidden_size), 40 + group),
            ]
        )
    return torch.cat(groups, dim=0)


def test_single_module_registers_dense_moe_and_vlm_architectures():
    assert {entry["model_name"] for entry in CONFIG} == {
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_registry_resolves_target_converter(
    architecture, vl_config, rank_info, infer_engine_config
):
    converter = get_infer_weights_converter(
        "sglang",
        architecture,
        vl_config,
        rank_info,
        infer_engine_config,
    )
    assert isinstance(converter, SGlangToHFWeightConverterQwen3_5)


def test_gated_qkv_layout_is_grouped_by_inference_tp(text_config):
    packed = _Qwen3_5Layout.pack_output_gated_qkv(
        make_gated_qkv(text_config), text_config, infer_tp_size=2
    )
    rank0, rank1 = torch.chunk(packed, 2, dim=0)

    assert rank0.shape == rank1.shape == (12, text_config.hidden_size)
    torch.testing.assert_close(rank0[:2], torch.full_like(rank0[:2], 10))
    torch.testing.assert_close(rank0[2:4], torch.full_like(rank0[2:4], 20))
    torch.testing.assert_close(rank0[4:6], torch.full_like(rank0[4:6], 10))
    torch.testing.assert_close(rank0[6:8], torch.full_like(rank0[6:8], 20))
    torch.testing.assert_close(rank0[8:10], torch.full_like(rank0[8:10], 30))
    torch.testing.assert_close(rank0[10:], torch.full_like(rank0[10:], 40))
    torch.testing.assert_close(rank1[:2], torch.full_like(rank1[:2], 11))
    torch.testing.assert_close(rank1[2:4], torch.full_like(rank1[2:4], 21))


def test_main_gated_qkv_has_train_infer_parity(
    vl_config, text_config, tf_config, rank_info, infer_engine_config
):
    train_converter = make_train_converter(vl_config, rank_info, tf_config)
    infer_converter = SGlangToHFWeightConverterQwen3_5(
        vl_config, infer_engine_config, rank_info
    )
    mcore_qkv = make_gated_qkv(text_config)

    train = train_converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_qkv.weight",
        mcore_qkv,
    )
    infer = infer_converter.convert_param("model.layers.0.qkv_proj.weight", train[0][1])

    expected = "model.language_model.layers.0.self_attn.qkv_proj.weight"
    assert train[0][0] == infer[0][0] == expected
    torch.testing.assert_close(train[0][1], infer[0][1], rtol=0, atol=0)


def test_gdn_conversion_matches_sglang_fused_names(vl_config, tf_config, rank_info):
    converter = make_train_converter(vl_config, rank_info, tf_config)
    # Q,K are 4 rows each; V,Z are 8 each; B,A are 4 each.
    parameter = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)

    converted = converter.convert_param(
        "module.module.language_model.decoder.layers.0.self_attention.in_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.language_model.layers.0.linear_attn.in_proj_ba.weight",
    ]
    assert converted[0][1].shape[0] == 24
    assert converted[1][1].shape[0] == 8


@pytest.mark.parametrize(
    "name",
    [
        "module.module.language_model.mtp.layers.0.eh_proj.weight",
        "module.module.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight",
        "module.module.model.mtp.layers.0.final_layernorm.weight",
    ],
)
def test_train_mtp_parameters_are_filtered(name, vl_config, tf_config, rank_info):
    converter = make_train_converter(vl_config, rank_info, tf_config)

    assert converter.convert_param(name, torch.ones(8, 8)) == []


@pytest.mark.parametrize("name", ["mtp.fc.weight", "model.mtp.layers.0.norm.weight"])
def test_infer_mtp_parameters_are_filtered(
    name, vl_config, rank_info, infer_engine_config
):
    converter = SGlangToHFWeightConverterQwen3_5(
        vl_config, infer_engine_config, rank_info
    )

    assert converter.convert_param(name, torch.ones(8, 8)) == []


def test_pipeline_mapping_reindexes_main_layers(vl_config, tf_config, rank_info):
    rank_info.pp_rank = 1
    rank_info.pp_size = 2
    converter = make_train_converter(
        vl_config,
        rank_info,
        tf_config,
        train_pp_stage_layer_id_map={(1, 0): {0: 3}},
    )

    main = converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_proj.weight",
        torch.ones(8, 8),
    )
    assert main[0][0] == "model.language_model.layers.3.self_attn.o_proj.weight"


def test_native_vision_names_match_sglang(
    vl_config, tf_config, rank_info, infer_engine_config
):
    train_converter = make_train_converter(vl_config, rank_info, tf_config)
    infer_converter = SGlangToHFWeightConverterQwen3_5(
        vl_config, infer_engine_config, rank_info
    )
    parameter = torch.ones(4, 4)

    train = train_converter.convert_param(
        "module.module.vision_model.decoder.layers.2.self_attention.linear_proj.weight",
        parameter,
    )
    infer = infer_converter.convert_param("visual.blocks.2.attn.proj.weight", parameter)

    assert train[0][0] == infer[0][0] == ("model.visual.blocks.2.attn.proj.weight")


@pytest.mark.parametrize(
    ("mcore_name", "canonical_name"),
    [
        (
            "vision_model.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.weight",
        ),
        (
            "vision_model.patch_embed.proj.bias",
            "model.visual.patch_embed.proj.bias",
        ),
        ("vision_model.pos_embed.weight", "model.visual.pos_embed.weight"),
        (
            "vision_model.merger.patch_norm.weight",
            "model.visual.merger.norm.weight",
        ),
        (
            "vision_model.merger.patch_norm.bias",
            "model.visual.merger.norm.bias",
        ),
        (
            "vision_model.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.weight",
        ),
        (
            "vision_model.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc1.bias",
        ),
        (
            "vision_model.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.weight",
        ),
        (
            "vision_model.merger.linear_fc2.bias",
            "model.visual.merger.linear_fc2.bias",
        ),
    ],
)
def test_native_vision_direct_names(
    mcore_name, canonical_name, vl_config, tf_config, rank_info
):
    """Cover the parameter names emitted by Megatron-Bridge Qwen3.5 VLM."""
    converter = make_train_converter(vl_config, rank_info, tf_config)
    parameter = torch.ones(4)

    assert converter.convert_param(f"module.module.{mcore_name}", parameter) == [
        (canonical_name, parameter)
    ]


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_dim", "expected_shards"),
    [
        (
            "model.language_model.layers.0.self_attn.qkv_proj.weight",
            ShardingType.TP_SHARDING,
            0,
            2,
        ),
        (
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            ShardingType.TP_SHARDING,
            1,
            2,
        ),
        (
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            ShardingType.EP_SHARDING,
            1,
            2,
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            ShardingType.TP_SHARDING,
            1,
            2,
        ),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            ShardingType.TP_SHARDING,
            0,
            2,
        ),
    ],
)
def test_tp_ep_and_vision_sharding(
    name, expected_type, expected_dim, expected_shards, rank_info
):
    rank_info.tp_size = 2
    rank_info.attn_tp_size = 2
    rank_info.ep_size = 2
    strategy = Qwen3_5ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=2,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    assert strategy.get_sharding_strategy(name) == (
        expected_type,
        expected_dim,
        expected_shards,
    )


def test_ep_is_preserved_when_dense_tp_is_one(rank_info):
    rank_info.ep_size = 2
    strategy = Qwen3_5ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=1,
        tp_size=1,
        ep_size=2,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    assert strategy.get_sharding_strategy(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    ) == (ShardingType.EP_SHARDING, 0, 2)
