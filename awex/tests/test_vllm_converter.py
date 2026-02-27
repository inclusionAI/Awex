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

import torch
from transformers import PretrainedConfig

from awex.config import InferenceConfig
from awex.converter.vllm_converter import VLLMToHFWeightConverter
from awex.sharding.rank_info import RankInfo


def _make_converter():
    hf_config = PretrainedConfig()
    hf_config.num_attention_heads = 8
    hf_config.num_key_value_heads = 8
    infer_config = InferenceConfig(tp_size=1, ep_size=1)
    rank_info = RankInfo(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=1,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=True,
    )
    return VLLMToHFWeightConverter(hf_config, infer_config, rank_info)


def test_qkv_proj_mapping():
    converter = _make_converter()
    weight = torch.zeros((12, 12))
    name = "model.layers.0.self_attn.qkv_proj.weight"
    converted = converter.convert_param(name, weight)
    assert converted == [
        ("model.layers.0.attention.query_key_value_proj.weight", weight)
    ]


def test_glm4v_qkv_mapping():
    converter = _make_converter()
    weight = torch.zeros((12, 12))
    name = "model.layers.0.self_attn.qkv.weight"
    converted = converter.convert_param(name, weight)
    assert converted == [
        ("model.layers.0.attention.query_key_value_proj.weight", weight)
    ]


def test_o_proj_mapping():
    converter = _make_converter()
    weight = torch.zeros((12, 12))
    name = "model.layers.0.self_attn.o_proj.weight"
    converted = converter.convert_param(name, weight)
    assert converted == [("model.layers.0.attention.dense.weight", weight)]


def test_glm4v_proj_mapping():
    converter = _make_converter()
    weight = torch.zeros((12, 12))
    name = "model.layers.0.self_attn.proj.weight"
    converted = converter.convert_param(name, weight)
    assert converted == [("model.layers.0.attention.dense.weight", weight)]
