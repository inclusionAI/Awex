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

"""CPU unit tests for Qwen3-VL dense and MoE weight conversion."""

from types import SimpleNamespace

import pytest
import torch

from awex.meta.meta_resolver import get_num_hidden_layers
from awex.models.qwen3_moe import SGlangToHFWeightConverterQwen3Moe
from awex.models.qwen3_vl import (
    Qwen3VLSGlangToHFWeightConverter,
    Qwen3VLShardingStrategy,
    _build_mcore_converter_qwen3_vl,
)
from awex.models.registry import (
    get_infer_weights_converter,
    get_sharding_strategy,
    get_train_weights_converter,
)
from awex.sharding.param_sharding import ShardingType


@pytest.fixture
def text_config():
    return SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_experts=4,
        tie_word_embeddings=False,
    )


@pytest.fixture
def vl_config(text_config):
    return SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"],
        text_config=text_config,
        vision_config=SimpleNamespace(num_heads=2, hidden_size=4),
    )


@pytest.fixture
def tf_config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        num_layers=2,
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


def _make_train_converter(vl_config, rank_info, tf_config, **infer_conf):
    config = {"infer_atten_tp_size": 1, **infer_conf}
    converter_class = _build_mcore_converter_qwen3_vl()
    return converter_class(vl_config, rank_info, config, tf_config)


def _make_infer_converter(vl_config, rank_info, infer_engine_config):
    return Qwen3VLSGlangToHFWeightConverter(
        vl_config, infer_engine_config, rank_info
    )


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"],
)
def test_registry_resolves_qwen3_vl_dense_and_moe(
    architecture, vl_config, rank_info, infer_engine_config
):
    vl_config.architectures = [architecture]

    converter = get_infer_weights_converter(
        "sglang",
        architecture,
        vl_config,
        rank_info,
        infer_engine_config,
    )

    assert isinstance(converter, Qwen3VLSGlangToHFWeightConverter)
    assert issubclass(
        Qwen3VLSGlangToHFWeightConverter, SGlangToHFWeightConverterQwen3Moe
    )
    assert get_sharding_strategy(architecture) is Qwen3VLShardingStrategy


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(num_hidden_layers=12), 12),
        (SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=28)), 28),
        ({"text_config": {"num_hidden_layers": 48}}, 48),
    ],
)
def test_num_hidden_layers_supports_flat_and_composite_configs(config, expected):
    assert get_num_hidden_layers(config) == expected


def test_num_hidden_layers_rejects_missing_decoder_depth():
    with pytest.raises(AttributeError, match="num_hidden_layers"):
        get_num_hidden_layers(SimpleNamespace(vision_config=SimpleNamespace()))


def test_language_qkv_has_train_infer_name_and_value_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    train_converter = _make_train_converter(
        vl_config, rank_info, tf_config, router_dtype="fp32"
    )
    infer_converter = _make_infer_converter(
        vl_config, rank_info, infer_engine_config
    )
    mcore_qkv = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_qkv.weight",
        mcore_qkv,
    )
    sglang_qkv = torch.cat([parameter for _, parameter in train_params], dim=0)
    infer_params = infer_converter.convert_param(
        "model.layers.0.self_attn.qkv_proj.weight", sglang_qkv
    )

    expected_names = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    ]
    assert [name for name, _ in train_params] == expected_names
    assert [name for name, _ in infer_params] == expected_names
    for (_, train_param), (_, infer_param) in zip(train_params, infer_params):
        torch.testing.assert_close(train_param, infer_param, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["weight", "bias"])
def test_vision_qkv_tp2_matches_sglang_rank_layout(
    kind, vl_config, tf_config, rank_info, monkeypatch
):
    monkeypatch.setattr(
        "awex.converter.mcore_converter.get_full_tensor",
        lambda parameter, dim=0: parameter,
    )
    converter = _make_train_converter(
        vl_config, rank_info, tf_config, infer_atten_tp_size=2
    )
    tail_shape = (4,) if kind == "weight" else ()
    q = [torch.full((2, *tail_shape), 10 + head) for head in range(2)]
    k = [torch.full((2, *tail_shape), 20 + head) for head in range(2)]
    v = [torch.full((2, *tail_shape), 30 + head) for head in range(2)]
    mcore_qkv = torch.cat(
        [torch.cat((q[head], k[head], v[head]), dim=0) for head in range(2)],
        dim=0,
    )

    [(name, packed_qkv)] = converter.convert_param(
        "module.module.vision_model.decoder.layers.1."
        f"self_attention.linear_qkv.{kind}",
        mcore_qkv,
    )

    assert name == f"model.visual.blocks.1.attn.qkv.{kind}"
    for rank, rank_shard in enumerate(torch.chunk(packed_qkv, 2, dim=0)):
        expected = torch.cat((q[rank], k[rank], v[rank]), dim=0)
        torch.testing.assert_close(rank_shard, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mcore_name", "sglang_name", "canonical_name"),
    [
        (
            "module.module.vision_model.patch_embed.proj.weight",
            "visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.weight",
        ),
        (
            "module.module.vision_model.decoder.layers.1.mlp.linear_fc1.weight",
            "visual.blocks.1.mlp.linear_fc1.weight",
            "model.visual.blocks.1.mlp.linear_fc1.weight",
        ),
        (
            "module.module.vision_model.merger.linear_fc2.weight",
            "visual.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.weight",
        ),
        (
            "module.module.vision_model.decoder.deepstack_merger_list.2."
            "patch_norm.weight",
            "visual.deepstack_merger_list.2.norm.weight",
            "model.visual.deepstack_merger_list.2.norm.weight",
        ),
    ],
)
def test_vision_parameters_have_train_infer_parity(
    mcore_name,
    sglang_name,
    canonical_name,
    vl_config,
    tf_config,
    rank_info,
    infer_engine_config,
):
    train_converter = _make_train_converter(vl_config, rank_info, tf_config)
    infer_converter = _make_infer_converter(
        vl_config, rank_info, infer_engine_config
    )
    parameter = torch.arange(8, dtype=torch.float32)

    train_params = train_converter.convert_param(mcore_name, parameter)
    infer_params = infer_converter.convert_param(sglang_name, parameter)

    assert [name for name, _ in train_params] == [canonical_name]
    assert [name for name, _ in infer_params] == [canonical_name]
    torch.testing.assert_close(train_params[0][1], infer_params[0][1], rtol=0, atol=0)


def test_moe_expert_reuses_qwen3_global_ep_numbering(
    vl_config, tf_config, rank_info
):
    rank_info.ep_size = 2
    rank_info.ep_rank = 1
    converter = _make_train_converter(vl_config, rank_info, tf_config)

    converted = converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "mlp.experts.linear_fc2.weight0",
        torch.ones(8, 6),
    )

    assert converted[0][0] == (
        "model.language_model.layers.0.mlp.experts.2.down_proj.weight"
    )


def test_tied_embedding_alias_has_train_infer_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    vl_config.text_config.tie_word_embeddings = True
    train_converter = _make_train_converter(vl_config, rank_info, tf_config)
    infer_converter = _make_infer_converter(
        vl_config, rank_info, infer_engine_config
    )
    embedding = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.embedding.word_embeddings.weight", embedding
    )
    infer_params = infer_converter.convert_param(
        "model.embed_tokens.weight", embedding
    )

    expected_names = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    assert [name for name, _ in train_params] == expected_names
    assert [name for name, _ in infer_params] == expected_names


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_dim"),
    [
        ("model.visual.patch_embed.proj.weight", ShardingType.NO_SHARDING, 0),
        ("model.visual.blocks.0.attn.qkv.weight", ShardingType.TP_SHARDING, 0),
        ("model.visual.blocks.0.attn.proj.weight", ShardingType.TP_SHARDING, 1),
        (
            "model.visual.deepstack_merger_list.0.linear_fc1.weight",
            ShardingType.TP_SHARDING,
            0,
        ),
        (
            "model.visual.deepstack_merger_list.0.linear_fc2.weight",
            ShardingType.TP_SHARDING,
            1,
        ),
        (
            "model.visual.deepstack_merger_list.0.linear_fc2.bias",
            ShardingType.NO_SHARDING,
            0,
        ),
    ],
)
def test_vision_sharding_matches_sglang_layout(
    name, expected_type, expected_dim, rank_info
):
    rank_info.tp_size = 2
    rank_info.attn_tp_size = 2
    strategy = Qwen3VLShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(name)

    assert sharding_type is expected_type
    assert sharding_dim == expected_dim
    assert num_shards == (2 if expected_type is ShardingType.TP_SHARDING else 1)


def test_vision_pos_embedding_is_replicated_on_mcore_and_sharded_on_sglang(
    rank_info,
):
    rank_info.tp_size = 2
    common = {
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "moe_dense_tp_size": 2,
        "tp_size": 2,
        "ep_size": 1,
        "ep_tp_size": 1,
        "rank_info": rank_info,
        "device_backend": "cpu",
    }

    mcore = Qwen3VLShardingStrategy(engine_name="mcore", **common)
    sglang = Qwen3VLShardingStrategy(engine_name="sglang", **common)

    name = "model.visual.pos_embed.weight"
    assert mcore.get_sharding_strategy(name) == (ShardingType.NO_SHARDING, 0, 1)
    assert sglang.get_sharding_strategy(name) == (ShardingType.TP_SHARDING, 0, 2)


def test_pipeline_mapping_only_rewrites_language_layers(
    vl_config, tf_config, rank_info
):
    rank_info.pp_rank = 1
    rank_info.pp_size = 2
    converter = _make_train_converter(
        vl_config,
        rank_info,
        tf_config,
        train_pp_stage_layer_id_map={(1, 0): {0: 1}},
    )
    language_weight = torch.ones(8, 8)
    vision_weight = torch.ones(4, 4)

    [(language_name, _)] = converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_proj.weight",
        language_weight,
    )
    [(vision_name, _)] = converter.convert_param(
        "module.module.vision_model.decoder.layers.0."
        "self_attention.linear_proj.weight",
        vision_weight,
    )

    assert language_name == "model.language_model.layers.1.self_attn.o_proj.weight"
    assert vision_name == "model.visual.blocks.0.attn.proj.weight"


def test_registry_builds_train_converter_with_composite_config(
    vl_config, tf_config, rank_info
):
    converter = get_train_weights_converter(
        "mcore",
        "Qwen3VLForConditionalGeneration",
        vl_config,
        rank_info,
        {"infer_atten_tp_size": 1},
        tf_config=tf_config,
    )

    assert converter.vl_hf_config is vl_config
    assert converter.hf_config is vl_config.text_config


def test_unknown_vision_parameter_fails_fast(vl_config, tf_config, rank_info):
    converter = _make_train_converter(vl_config, rank_info, tf_config)

    with pytest.raises(ValueError, match="Unknown Qwen3-VL vision parameter"):
        converter.convert_param(
            "module.module.vision_model.unknown.weight", torch.ones(2)
        )
