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

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from awex.config import InferenceConfig, InferenceConfigValidationError
from awex.engine import sglang, vllm
from awex.sharding.sglang_sharding import get_sglang_rank_info
from awex.util.common import stripped_env_vars
from awex.writer.nccl_writer import NCCLWeightsWriter


@pytest.mark.parametrize("backend", ["nccl", "hccl", "astate", "mooncake"])
def test_single_engine_requires_meta_server(backend):
    with pytest.raises(InferenceConfigValidationError, match="meta_server_addr"):
        InferenceConfig.from_dict({"comm_backend": backend})
    with pytest.raises(InferenceConfigValidationError, match="meta_server_addr"):
        InferenceConfig.from_sgl_server_args(SimpleNamespace(), comm_backend=backend)
    InferenceConfig(comm_backend=backend, meta_server_addr="127.0.0.1:12345").validate()
    InferenceConfig().validate()


@pytest.mark.parametrize(
    "module,cls", [(sglang, sglang.SGLangEngine), (vllm, vllm.VLLMEngine)]
)
def test_engine_validates_direct_config_and_owns_reader(monkeypatch, module, cls):
    hf_config = SimpleNamespace()
    native = SimpleNamespace(
        initialized=True,
        tokenizer_manager=SimpleNamespace(model_config=hf_config),
    )
    args = (native,) if cls is sglang.SGLangEngine else (native, hf_config)
    with pytest.raises(InferenceConfigValidationError, match="meta_server_addr"):
        cls(InferenceConfig(comm_backend="nccl"), *args)
    config = InferenceConfig(comm_backend="nccl", meta_server_addr="127.0.0.1:12345")
    reader = Mock()

    def create_reader(engine):
        # Reader construction immediately calls back into the engine.
        assert engine._initialized
        assert engine.node_rank == 0
        return reader

    monkeypatch.setattr(module, "get_weights_exchange_reader", create_reader)
    engine = cls(config, *args)
    engine.initialize()
    reader.initialize.assert_called_once_with()
    engine.set_global_step(1)
    engine.update_weights()
    reader.update_weights.assert_called_once_with(step_id=1)


def test_sglang_hook_cp_axes_reach_rank_info():
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(
            tp_rank=3,
            tp_size=8,
            pp_rank=0,
            pp_size=1,
            dp_rank=0,
            dp_size=2,
            moe_ep_rank=0,
            moe_ep_size=1,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_dp_rank=0,
            attn_cp_rank=1,
            attn_cp_size=2,
        ),
        world_group=SimpleNamespace(rank=3, world_size=8, local_rank=3),
        server_args=SimpleNamespace(
            nnodes=1, ep_size=1, dp_size=2, enable_dp_attention=True
        ),
        tp_worker=SimpleNamespace(),
    )
    info = get_sglang_rank_info(sglang._model_context(scheduler), engine_rank=0)
    assert (info.cp_rank, info.cp_size) == (1, 2)
    assert (info.global_rank, info.world_size) == (3, 8)


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cpu")
    train_engine = SimpleNamespace(
        enable_debug_mode=False,
        enable_colocate_mode=True,
        meta_server_addr="127.0.0.1:12345",
        model=[],
        hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
        comm_backend="nccl",
        config={"weights_exchange_ipc_backend": "cpu"},
        engine_name="mcore",
    )
    result = NCCLWeightsWriter(train_engine)
    yield result
    result.meta_server_client.close()


def test_colocate_writer_uses_engine_config(writer, monkeypatch):
    writer.transfer_rank = 8
    monkeypatch.setattr(writer, "_set_device", Mock())
    monkeypatch.setattr("awex.writer.nccl_writer.get_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr("awex.writer.nccl_writer.device_util.current_device", lambda: 0)
    add = Mock()
    monkeypatch.setattr(writer.meta_server_client, "add_object_to_set", add)
    writer._init_writer_in_colocate_mode()
    assert writer.ipc_backend == "cpu"
    add.assert_called_once_with("training_device_rank_entries", ("127.0.0.1", 0, 8))


@pytest.mark.parametrize(
    "backend,destroy", [("nccl", True), ("hccl", False), ("hccl", True)]
)
def test_writer_process_group_cleanup(writer, monkeypatch, backend, destroy):
    writer.comm_backend = backend
    writer.destroy_pg_after_update = destroy
    writer.weights_update_group = object()
    writer.already_initialized = True
    cleanup = Mock()
    npu = SimpleNamespace(synchronize=Mock(), empty_cache=Mock())
    monkeypatch.setattr(torch.distributed, "destroy_process_group", cleanup)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    writer._destroy_weights_exchange_process_group()
    if backend == "hccl" and destroy:
        cleanup.assert_called_once_with(writer.weights_update_group)
        npu.synchronize.assert_called_once_with()
        npu.empty_cache.assert_called_once_with()
        assert not writer.already_initialized
    else:
        cleanup.assert_not_called()
        npu.synchronize.assert_not_called()
        assert writer.already_initialized


def test_environment_logging_only_includes_known_settings():
    with patch.dict(
        os.environ,
        {
            "HF_TOKEN": "synthetic",
            "API_KEY": "synthetic",
            "PASSWORD": "synthetic",
            "AWEX_ACCESS_TOKEN": "synthetic",
            "NCCL_UNKNOWN_SECRET": "synthetic",
            "AWEX_MOONCAKE_PROTOCOL": "tcp",
            "RANK": "3",
            "NCCL_DEBUG": "INFO",
        },
        clear=True,
    ):
        assert stripped_env_vars() == {
            "AWEX_MOONCAKE_PROTOCOL": "tcp",
            "RANK": "3",
            "NCCL_DEBUG": "INFO",
        }
