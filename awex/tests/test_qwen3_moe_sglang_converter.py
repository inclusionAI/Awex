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

"""Name-contract tests for the Qwen3-MoE SGLang->HF weight converter.

The colocate transfer plan matches inference-side and train-side parameters
by name. The train side (Megatron converter) emits canonical per-parameter
HF names, so the inference side must expand SGLang fused parameters —
including MoE experts — to the exact same name set. These tests pin that
contract on CPU without requiring SGLang or Megatron.
"""

from types import SimpleNamespace

import torch

from awex.models.qwen3_moe import SGlangToHFWeightConverterQwen3Moe
from awex.models.registry import get_infer_weights_converter

# Tiny Qwen3-MoE-like geometry: GQA with 8 query heads and 2 KV heads.
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 4
HIDDEN = 16
MOE_INTERMEDIATE = 8
NUM_EXPERTS = 4


def _model_config():
    return SimpleNamespace(
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        num_experts=NUM_EXPERTS,
        architectures=["Qwen3MoeForCausalLM"],
    )


def _rank_info(tp_rank=0, ep_rank=0):
    return SimpleNamespace(tp_rank=tp_rank, ep_rank=ep_rank)


def _infer_engine_config(tp_size=1, ep_size=1):
    return SimpleNamespace(tp_size=tp_size, ep_size=ep_size, device_backend="cuda")


def _make_converter(tp_size=1, ep_size=1, tp_rank=0, ep_rank=0):
    return SGlangToHFWeightConverterQwen3Moe(
        _model_config(),
        _infer_engine_config(tp_size=tp_size, ep_size=ep_size),
        _rank_info(tp_rank=tp_rank, ep_rank=ep_rank),
    )


def _sglang_named_params(num_local_experts=NUM_EXPERTS):
    """Parameter names/shapes as exposed by SGLang for one decoder layer."""
    qkv_rows = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    return {
        "model.embed_tokens.weight": torch.randn(32, HIDDEN),
        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(qkv_rows, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            HIDDEN, NUM_HEADS * HEAD_DIM
        ),
        "model.layers.0.self_attn.q_norm.weight": torch.randn(HEAD_DIM),
        "model.layers.0.self_attn.k_norm.weight": torch.randn(HEAD_DIM),
        "model.layers.0.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.mlp.gate.weight": torch.randn(NUM_EXPERTS, HIDDEN),
        "model.layers.0.mlp.experts.w13_weight": torch.randn(
            num_local_experts, 2 * MOE_INTERMEDIATE, HIDDEN
        ),
        "model.layers.0.mlp.experts.w2_weight": torch.randn(
            num_local_experts, HIDDEN, MOE_INTERMEDIATE
        ),
        "model.norm.weight": torch.randn(HIDDEN),
        "lm_head.weight": torch.randn(32, HIDDEN),
    }


def _expected_hf_names(expert_ids):
    """Canonical HF names the Megatron train-side converter reports."""
    names = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for expert_id in expert_ids:
        names.add(f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight")
        names.add(f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight")
        names.add(f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight")
    return names


def test_registry_resolves_qwen3_moe_sglang_converter():
    converter = get_infer_weights_converter(
        "sglang",
        "Qwen3MoeForCausalLM",
        _model_config(),
        _rank_info(),
        _infer_engine_config(),
    )
    assert isinstance(converter, SGlangToHFWeightConverterQwen3Moe)


def test_convert_param_name_set_matches_train_side_contract():
    converter = _make_converter()
    converted_names = set()
    for name, param in _sglang_named_params().items():
        for hf_name, _ in converter.convert_param(name, param):
            converted_names.add(hf_name)
    assert converted_names == _expected_hf_names(range(NUM_EXPERTS))


def test_convert_param_expert_names_use_global_ids_with_ep():
    # With ep_size=2 each rank holds half the experts; converted names must
    # carry global expert ids offset by ep_rank.
    num_local = NUM_EXPERTS // 2
    converter = _make_converter(ep_size=2, ep_rank=1)
    converted_names = set()
    for name, param in _sglang_named_params(num_local_experts=num_local).items():
        for hf_name, _ in converter.convert_param(name, param):
            converted_names.add(hf_name)
    assert converted_names == _expected_hf_names(range(num_local, NUM_EXPERTS))


def test_qkv_split_is_gqa_aware():
    converter = _make_converter()
    q = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN)
    k = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
    v = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
    fused = torch.cat([q, k, v], dim=0)

    result = dict(
        converter.convert_param("model.layers.0.self_attn.qkv_proj.weight", fused)
    )
    assert torch.equal(result["model.layers.0.self_attn.q_proj.weight"], q)
    assert torch.equal(result["model.layers.0.self_attn.k_proj.weight"], k)
    assert torch.equal(result["model.layers.0.self_attn.v_proj.weight"], v)


def test_qkv_split_rejects_indivisible_rows():
    converter = _make_converter()
    bad = torch.randn((NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM + 1, HIDDEN)
    try:
        converter.convert_param("model.layers.0.self_attn.qkv_proj.weight", bad)
    except ValueError as e:
        assert "not divisible" in str(e)
    else:
        raise AssertionError("expected ValueError for indivisible qkv rows")


def test_expert_split_values_match_fused_slices():
    converter = _make_converter()
    w13 = torch.randn(NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE)

    gate_up = dict(
        converter.convert_param("model.layers.0.mlp.experts.w13_weight", w13)
    )
    down = dict(converter.convert_param("model.layers.0.mlp.experts.w2_weight", w2))

    for expert_id in range(NUM_EXPERTS):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        assert torch.equal(
            gate_up[f"{prefix}.gate_proj.weight"], w13[expert_id, :MOE_INTERMEDIATE]
        )
        assert torch.equal(
            gate_up[f"{prefix}.up_proj.weight"], w13[expert_id, MOE_INTERMEDIATE:]
        )
        assert torch.equal(down[f"{prefix}.down_proj.weight"], w2[expert_id])
