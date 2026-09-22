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


from contextlib import nullcontext

import pytest
import torch

from awex.util.cuda_ipc import cuda_ipc_allocation


@pytest.mark.parametrize("fail", [False, True])
def test_ipc_allocation_restores_runtime_options_after_exit(monkeypatch, fail):
    original = (
        "expandable_segments:True,garbage_collection_threshold:0.7,"
        "max_split_size_mb:128,roundup_power2_divisions:[256:1,512:2,>:4]"
    )
    state = {"config": original}
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: {
            "allocator_settings": {
                "PYTORCH_CUDA_ALLOC_CONF": state["config"],
                "expandable_segments": "expandable_segments:True" in state["config"],
            }
        },
    )
    monkeypatch.setattr(
        torch.cuda.memory,
        "_set_allocator_settings",
        lambda config: state.update(config=config),
    )
    with pytest.raises(RuntimeError, match="packing failed") if fail else nullcontext():
        with cuda_ipc_allocation():
            assert "expandable_segments:False" in state["config"]
            assert "expandable_segments:True" not in state["config"]
            assert "garbage_collection_threshold:0.7" in state["config"]
            assert "max_split_size_mb:128" in state["config"]
            assert "roundup_power2_divisions:[256:1,512:2,>:4]" in state["config"]
            with cuda_ipc_allocation():
                assert "expandable_segments:False" in state["config"]
            if fail:
                raise RuntimeError("packing failed")
    assert state["config"] == original


def test_ipc_allocation_when_disabled_does_not_change_allocator(monkeypatch):
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: {"allocator_settings": {"expandable_segments": False}},
    )

    def unexpected(*args):
        pytest.fail("Disabled allocator must not be changed")

    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", unexpected)
    with cuda_ipc_allocation():
        pass


def test_ipc_allocation_restores_flag_omitted_from_last_runtime_update(monkeypatch):
    state = {"expandable": True, "config": "garbage_collection_threshold:0.7"}

    def set_config(config):
        state["config"] = config
        if "expandable_segments:" in config:
            state["expandable"] = "expandable_segments:True" in config

    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: {
            "allocator_settings": {
                "PYTORCH_CUDA_ALLOC_CONF": state["config"],
                "expandable_segments": state["expandable"],
            }
        },
    )
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", set_config)
    with cuda_ipc_allocation():
        assert not state["expandable"]
    assert state["expandable"]
    assert "garbage_collection_threshold:0.7" in state["config"]
