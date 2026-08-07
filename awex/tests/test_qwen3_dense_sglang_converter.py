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

"""Name-contract tests for dense Qwen3 (``Qwen3ForCausalLM``)."""

from types import SimpleNamespace

import torch

from awex.models.qwen3 import CONFIG
from awex.models.qwen3_moe import SGlangToHFWeightConverterQwen3Moe
from awex.models.registry import get_infer_weights_converter

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 4
HIDDEN = 16
INTERMEDIATE = 8


def _model_config():
    return SimpleNamespace(
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        architectures=["Qwen3ForCausalLM"],
    )


def _rank_info():
    return SimpleNamespace(tp_rank=0, ep_rank=0)


def _infer_engine_config():
    return SimpleNamespace(tp_size=1, ep_size=1, device_backend="cuda")


def _converter():
    return SGlangToHFWeightConverterQwen3Moe(
        _model_config(), _infer_engine_config(), _rank_info()
    )


def test_config_registers_the_dense_architecture():
    assert CONFIG["model_name"] == "Qwen3ForCausalLM"


def test_registry_resolves_the_shared_sglang_converter():
    converter = get_infer_weights_converter(
        "sglang",
        "Qwen3ForCausalLM",
        _model_config(),
        _rank_info(),
        _infer_engine_config(),
    )
    assert isinstance(converter, SGlangToHFWeightConverterQwen3Moe)


def test_layer_names_match_the_train_side_contract():
    converter = _converter()
    qkv_rows = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    sglang_params = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(qkv_rows, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.q_norm.weight": torch.randn(HEAD_DIM),
        "model.layers.0.self_attn.k_norm.weight": torch.randn(HEAD_DIM),
        "model.layers.0.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(HIDDEN),
        # sglang serves the dense MLP through MergedColumnParallelLinear
        "model.layers.0.mlp.gate_up_proj.weight": torch.randn(2 * INTERMEDIATE, HIDDEN),
        "model.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, INTERMEDIATE),
    }

    produced = set()
    for name, param in sglang_params.items():
        for hf_name, _ in converter.convert_param(name, param):
            produced.add(hf_name)

    assert produced == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }


def test_qkv_split_is_gqa_aware():
    converter = _converter()
    qkv_rows = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    fused = torch.arange(qkv_rows * HIDDEN, dtype=torch.float32).reshape(
        qkv_rows, HIDDEN
    )

    shapes = {
        name: param.shape
        for name, param in converter.convert_param(
            "model.layers.0.self_attn.qkv_proj.weight", fused
        )
    }

    assert shapes["model.layers.0.self_attn.q_proj.weight"][0] == NUM_HEADS * HEAD_DIM
    assert (
        shapes["model.layers.0.self_attn.k_proj.weight"][0] == NUM_KV_HEADS * HEAD_DIM
    )
    assert (
        shapes["model.layers.0.self_attn.v_proj.weight"][0] == NUM_KV_HEADS * HEAD_DIM
    )
