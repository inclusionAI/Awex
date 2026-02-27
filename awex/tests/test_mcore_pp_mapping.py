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

import copy
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from awex.converter.mcore_converter import _process_mcore_pp_name
from awex.meta.train_meta_resolver import (
    _build_pp_stage_layer_id_map,
    _canonicalize_pp_layer_names_in_global_meta,
)
from awex.sharding.rank_info import RankInfo
from awex.writer.weights_writer import WeightsExchangeShardingWriter


def _make_rank_info(pp_rank: int, pp_size: int, global_rank: int) -> RankInfo:
    return RankInfo(
        tp_rank=0,
        tp_size=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=pp_size,
        global_rank=global_rank,
        local_rank=global_rank,
        engine_rank=0,
        is_infer=False,
    )


def _make_hf_config(num_hidden_layers: int = 8) -> PretrainedConfig:
    cfg = PretrainedConfig()
    cfg.num_hidden_layers = num_hidden_layers
    return cfg


def test_process_name_without_map_keeps_local_layer_id():
    rank_info = _make_rank_info(pp_rank=1, pp_size=2, global_rank=1)
    hf_config = _make_hf_config()
    tf_config = SimpleNamespace()

    src = "decoder.layers.2.self_attention.linear_qkv.weight"
    out = _process_mcore_pp_name(
        src,
        rank_info,
        hf_config,
        tf_config,
        vp_stage=0,
        pp_stage_layer_id_map=None,
    )

    assert out == src


def test_process_name_with_map_rewrites_local_to_global():
    rank_info = _make_rank_info(pp_rank=1, pp_size=2, global_rank=1)
    hf_config = _make_hf_config()
    tf_config = SimpleNamespace()
    layer_map = {(1, 0): {1: 6, 2: 7}}

    src = "decoder.layers.2.self_attention.linear_qkv.weight"
    out = _process_mcore_pp_name(
        src,
        rank_info,
        hf_config,
        tf_config,
        vp_stage=0,
        pp_stage_layer_id_map=layer_map,
    )

    assert out == "decoder.layers.7.self_attention.linear_qkv.weight"


def _build_mock_global_meta():
    # One-based local layer ids on each stage, vp_size=2, pp_size=2.
    # Stage ordering should be (vp0,pp0)->(vp0,pp1)->(vp1,pp0)->(vp1,pp1).
    return [
        {
            "rank_info": _make_rank_info(pp_rank=0, pp_size=2, global_rank=0),
            "params_meta": [
                {
                    "name": "model.layers.1.self_attn.q_proj.weight",
                    "shape": (4, 4),
                    "numel": 16,
                    "dtype": torch.float32,
                    "vp_stage": 0,
                },
                {
                    "name": "model.layers.2.self_attn.q_proj.weight",
                    "shape": (4, 4),
                    "numel": 16,
                    "dtype": torch.float32,
                    "vp_stage": 0,
                },
                {
                    "name": "model.layers.1.self_attn.o_proj.weight",
                    "shape": (4, 4),
                    "numel": 16,
                    "dtype": torch.float32,
                    "vp_stage": 1,
                },
            ],
            "model_arch_name": "Qwen2ForCausalLM",
        },
        {
            "rank_info": _make_rank_info(pp_rank=1, pp_size=2, global_rank=1),
            "params_meta": [
                {
                    "name": "model.layers.1.self_attn.q_proj.weight",
                    "shape": (4, 4),
                    "numel": 16,
                    "dtype": torch.float32,
                    "vp_stage": 0,
                },
                {
                    "name": "model.layers.1.self_attn.o_proj.weight",
                    "shape": (4, 4),
                    "numel": 16,
                    "dtype": torch.float32,
                    "vp_stage": 1,
                },
                {
                    "name": "model.layers.2.self_attn.o_proj.weight",
                    "shape": (4, 4),
                    "numel": 16,
                    "dtype": torch.float32,
                    "vp_stage": 1,
                },
            ],
            "model_arch_name": "Qwen2ForCausalLM",
        },
    ]


def test_build_pp_stage_layer_id_map_from_one_based_metadata():
    global_meta = _build_mock_global_meta()

    stage_map = _build_pp_stage_layer_id_map(global_meta)

    assert stage_map[(0, 0)] == {1: 0, 2: 1}
    assert stage_map[(1, 0)] == {1: 2}
    assert stage_map[(0, 1)] == {1: 3}
    assert stage_map[(1, 1)] == {1: 4, 2: 5}


def test_canonicalize_global_metadata_layer_names():
    global_meta = _build_mock_global_meta()
    stage_map = _build_pp_stage_layer_id_map(global_meta)

    canonical = _canonicalize_pp_layer_names_in_global_meta(
        copy.deepcopy(global_meta),
        stage_map,
    )

    rank0_names = [p["name"] for p in canonical[0]["params_meta"]]
    rank1_names = [p["name"] for p in canonical[1]["params_meta"]]
    assert rank0_names == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
    ]
    assert rank1_names == [
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.4.self_attn.o_proj.weight",
        "model.layers.5.self_attn.o_proj.weight",
    ]


class _DummyModel:
    def __init__(self, idx: int):
        self._idx = idx
        self._p = torch.nn.Parameter(torch.tensor([float(idx)]))

    def named_parameters(self):
        return [
            (f"decoder.layers.{self._idx}.self_attention.linear_proj.weight", self._p)
        ]

    def state_dict(self):
        return {
            f"decoder.layers.{self._idx}.self_attention.linear_proj.weight": self._p
        }


class _DummyConverter:
    def __init__(self):
        self.calls = []

    def convert_param(self, name, param, vp_stage=None):
        self.calls.append(vp_stage)
        return []


def test_writer_convert_parameters_passes_vp_stage(monkeypatch):
    writer = object.__new__(WeightsExchangeShardingWriter)
    writer.model = [_DummyModel(0), _DummyModel(1)]
    writer.weight_converter = _DummyConverter()
    writer.enable_mem_debug = False
    writer.transfer_rank = 0
    writer.hf_config = SimpleNamespace(tie_word_embeddings=False)
    writer.rank_info = SimpleNamespace(pp_rank=0, pp_size=1)

    monkeypatch.setattr(
        "awex.writer.weights_writer.get_mcore_model_parameters",
        lambda model: model.state_dict(),
    )

    writer.convert_parameters()

    assert writer.weight_converter.calls == [0, 1]
