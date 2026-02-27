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

import sys
import types

import torch

import awex.mbridge_loader as mbridge_loader
from awex.mbridge_loader import load_weights_from_hf_with_mbridge


class DummyBridge:
    def _weight_name_mapping_mcore_local_to_global(self, model):
        return {
            "qkv": "qkv.weight",
            "o_proj": "o_proj.weight",
            "gate_up": "gate_up.weight",
        }

    def _weight_name_mapping_mcore_to_hf(self, global_name):
        mapping = {
            "qkv.weight": [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
            "o_proj.weight": ["model.layers.0.self_attn.o_proj.weight"],
            "gate_up.weight": [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ],
        }
        return mapping[global_name]


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.zeros((6, 4)))
        self.o_proj = torch.nn.Parameter(torch.zeros((4, 4)))
        self.gate_up = torch.nn.Parameter(torch.zeros((8, 4)))


def test_mbridge_loader_merges_params():
    model = DummyModel()
    reader_map = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones((2, 4)),
        "model.layers.0.self_attn.k_proj.weight": torch.ones((2, 4)) * 2,
        "model.layers.0.self_attn.v_proj.weight": torch.ones((2, 4)) * 3,
        "model.layers.0.self_attn.o_proj.weight": torch.ones((4, 4)) * 4,
        "model.layers.0.mlp.gate_proj.weight": torch.ones((4, 4)) * 5,
        "model.layers.0.mlp.up_proj.weight": torch.ones((4, 4)) * 6,
    }

    def reader(_, name):
        return reader_map[name]

    load_weights_from_hf_with_mbridge(DummyBridge(), [model], "/unused", reader=reader)

    assert torch.allclose(
        model.qkv, torch.tensor([[1.0] * 4] * 2 + [[2.0] * 4] * 2 + [[3.0] * 4] * 2)
    )
    assert torch.allclose(model.o_proj, torch.ones((4, 4)) * 4)
    assert torch.allclose(
        model.gate_up,
        torch.tensor([[5.0] * 4] * 4 + [[6.0] * 4] * 4),
    )


def test_default_tensor_reader_joins_index_relative_path(tmp_path, monkeypatch):
    shard_name = "model-00001-of-00001.safetensors"
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        '{"weight_map": {"model.layers.0.self_attn.q_proj.weight": "'
        + shard_name
        + '"}}',
        encoding="utf-8",
    )

    observed = {}

    class DummySafeOpen:
        def __init__(self, path, framework, device):
            observed["path"] = path
            observed["framework"] = framework
            observed["device"] = device

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def get_tensor(self, name):
            observed["name"] = name
            return torch.ones((2, 4))

    monkeypatch.setitem(
        sys.modules, "safetensors", types.SimpleNamespace(safe_open=DummySafeOpen)
    )
    tensor = mbridge_loader._default_tensor_reader(
        str(tmp_path), "model.layers.0.self_attn.q_proj.weight"
    )

    assert observed["path"] == str(tmp_path / shard_name)
    assert observed["framework"] == "pt"
    assert observed["device"] == "cpu"
    assert observed["name"] == "model.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(tensor, torch.ones((2, 4)))
