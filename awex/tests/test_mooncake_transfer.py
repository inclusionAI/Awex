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

import ctypes
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from awex.config import InferenceConfig
from awex.tests.test_transfer_plan_cpu_sim import _build_param_meta, _ShardSpec
from awex.transfer.mooncake_comm import MooncakeTransport
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    TransferPlan,
    TransferPlanBuilder,
)


class MemoryMeta:
    def __init__(self):
        self.values = {}
        self.condition = threading.Condition()

    def put_object(self, key, value):
        with self.condition:
            self.values[key] = value
            self.condition.notify_all()

    def get_object(self, key, timeout=0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while key not in self.values:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(key)
                self.condition.wait(remaining)
            return self.values[key]

    def delete_if_exists(self, key):
        with self.condition:
            self.values.pop(key, None)


class MemoryEngine:
    instances = []

    def __init__(self):
        self.port = 1000 + len(self.instances)
        self.instances.append(self)
        self.registered = {}
        self.registration_count = 0
        self.read_result = 0
        self.register_result = 0

    def initialize(self, host, metadata, protocol, device):
        assert metadata == "P2PHANDSHAKE"
        return 0

    def get_rpc_port(self):
        return self.port

    def register_memory(self, address, size):
        if self.register_result:
            return self.register_result
        assert address not in self.registered
        self.registered[address] = size
        self.registration_count += 1
        return 0

    def unregister_memory(self, address):
        del self.registered[address]
        return 0

    def transfer_sync_read(self, endpoint, local, remote, size):
        assert self.registered[local] == size
        peer = next(e for e in self.instances if endpoint.endswith(f":{e.port}"))
        assert peer.registered[remote] == size
        if self.read_result:
            return self.read_result
        ctypes.memmove(local, remote, size)
        return 0


@pytest.fixture
def transport(monkeypatch):
    MemoryEngine.instances = []
    monkeypatch.setitem(
        sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=MemoryEngine)
    )
    monkeypatch.setenv("AWEX_MOONCAKE_HOST", "127.0.0.1")
    monkeypatch.setenv("AWEX_MOONCAKE_PROTOCOL", "tcp")
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cpu")
    meta = MemoryMeta()
    return lambda rank: MooncakeTransport(meta, rank, timeout=2)


def _operation(peer=0):
    return CommunicationOperation(
        send_rank=2,
        recv_rank=peer,
        send_shard_meta=SimpleNamespace(name="weight"),
        recv_shard_meta=SimpleNamespace(name="weight"),
        send_offset=(0, 1),
        recv_offset=(0, 1),
        overlap_shape=(3, 2),
        train_slices=(slice(None), slice(1, 3)),
        inf_slices=(slice(None), slice(1, 3)),
    )


def test_fanout_strided_views_repeated_step(transport):
    writer = transport(2)
    readers = [transport(i) for i in range(2)]
    ops = [_operation(i) for i in range(2)]
    plan = TransferPlan(operations={i: [op] for i, op in enumerate(ops)})
    for version in range(2):
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4) + version * 100
        targets = [torch.full((4, 3), -1.0).T for _ in readers]
        with ThreadPoolExecutor(max_workers=3) as pool:
            send = pool.submit(writer.send, {"weight": source}, plan, 42)
            receives = [
                pool.submit(
                    reader.receive,
                    {"weight": target},
                    TransferPlan(operations={2: [ops[i]]}),
                    42,
                )
                for i, (reader, target) in enumerate(zip(readers, targets))
            ]
            for receive in receives:
                receive.result(timeout=5)
            send.result(timeout=5)
        for target in targets:
            torch.testing.assert_close(target[:, 1:3], source[:, 1:3])
            assert torch.all(target[:, [0, 3]] == -1)
        assert not writer.buffers
        assert not writer.meta_client.values
        assert all(not e.registered for e in MemoryEngine.instances)
    # Identical source slices for multiple inference engines are registered once.
    assert writer.engine.registration_count == 2


@pytest.mark.parametrize("train_size,infer_size", [(1, 2), (2, 1)])
def test_real_shard_plans(transport, train_size, infer_size):
    def meta(size):
        return [
            _build_param_meta(
                "weight",
                [_ShardSpec(i, i * (8 // size), 8 // size) for i in range(size)],
                8,
                size,
            )
        ]

    builder = TransferPlanBuilder(infer_size, train_size, 1)
    train_meta, infer_meta = meta(train_size), meta(infer_size)
    source = torch.arange(8, dtype=torch.float32)
    targets = [torch.zeros(8 // infer_size) for _ in range(infer_size)]
    with ThreadPoolExecutor(max_workers=train_size + infer_size) as pool:
        futures = []
        for rank in range(train_size + infer_size):
            plan = builder.build_local_transfer_plan(infer_meta, train_meta, rank)
            instance = transport(rank)
            if rank < infer_size:
                futures.append(
                    pool.submit(instance.receive, {"weight": targets[rank]}, plan, 0)
                )
            else:
                offset = (rank - infer_size) * (8 // train_size)
                futures.append(
                    pool.submit(
                        instance.send,
                        {"weight": source[offset : offset + 8 // train_size]},
                        plan,
                        0,
                    )
                )
        for future in futures:
            future.result(timeout=5)
    torch.testing.assert_close(torch.cat(targets), source)


@pytest.mark.parametrize("failure", ["read", "shape"])
def test_reader_failure_preserves_exported_memory(transport, failure):
    writer, reader = transport(2), transport(0)
    op = _operation()
    if failure == "read":
        reader.engine.read_result = -1
    target = torch.zeros(
        (3, 4), dtype=torch.float64 if failure == "shape" else torch.float32
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        send = pool.submit(
            writer.send,
            {"weight": torch.ones(3, 4)},
            TransferPlan(operations={0: [op]}),
            0,
        )
        receive = pool.submit(
            reader.receive, {"weight": target}, TransferPlan(operations={2: [op]}), 0
        )
        with pytest.raises((RuntimeError, ValueError), match="Mooncake"):
            receive.result(timeout=5)
        with pytest.raises(RuntimeError, match="reader 0 failed"):
            send.result(timeout=5)
    assert writer.buffers
    assert writer.engine.registered
    with pytest.raises(RuntimeError, match="restart the job"):
        writer.send({}, TransferPlan(), 1)


def test_timeout_keeps_memory_registered(transport):
    writer = transport(2)
    writer.timeout = 0.01
    with pytest.raises(TimeoutError):
        writer.send(
            {"weight": torch.ones(3, 4)},
            TransferPlan(operations={0: [_operation()]}),
            0,
        )
    assert writer.buffers
    assert writer.engine.registered


def test_registration_failure_releases_unpublished_memory(transport):
    writer = transport(2)
    writer.engine.register_result = -1
    with pytest.raises(RuntimeError, match="register_memory"):
        writer.send(
            {"weight": torch.ones(3, 4)},
            TransferPlan(operations={0: [_operation()]}),
            0,
        )
    assert not writer.buffers
    assert not writer.meta_client.values


def test_empty_plan_colocate_offload_callback(transport):
    writer = transport(2)
    ready = []
    writer.send(
        {}, TransferPlan(), 0, offload=True, on_ready=lambda: ready.append(True)
    )
    assert ready == [True]
    assert not writer.engine.registered


def test_config_accepts_mooncake():
    assert (
        InferenceConfig.from_dict(
            {"comm_backend": "mooncake", "meta_server_addr": "127.0.0.1:12345"}
        ).comm_backend
        == "mooncake"
    )


def test_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake.engine", None)
    with pytest.raises(ImportError, match="awex\\[mooncake\\]"):
        MooncakeTransport(MemoryMeta(), 0, 1)


def test_reader_refreshes_replaced_parameters():
    from awex.reader.mooncake_reader import MooncakeWorkerWeightsReader

    reader = object.__new__(MooncakeWorkerWeightsReader)
    current = {"weight": torch.ones(2)}
    reader.model = SimpleNamespace(named_parameters=lambda: current.items())
    reader.weight_converter = SimpleNamespace(convert_param=lambda n, p: [(n, p)])
    reader.hf_config = SimpleNamespace(tie_word_embeddings=False)
    seen = []
    reader.transport = SimpleNamespace(
        receive=lambda parameters, plan, step: seen.append(parameters["weight"])
    )
    reader.transfer_plan = TransferPlan()
    reader._update_weights(0)
    current["weight"] = torch.zeros(2)
    reader._update_weights(1)
    assert seen[0] is not seen[1]
    assert seen[1] is current["weight"]


def test_reader_backend_dispatch(monkeypatch):
    import pickle

    from awex.reader.weights_reader import WeightsReader

    initialized = []

    class Reader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def initialize(self):
            initialized.append(self)

    monkeypatch.setattr(
        "awex.reader.mooncake_reader.MooncakeWorkerWeightsReader", Reader
    )
    scheduler = SimpleNamespace()
    WeightsReader._init_in_tp_worker(
        infer_conf_bytes=pickle.dumps({"engine_name": "vllm"}),
        parameters_meta_bytes=pickle.dumps([]),
        training_params_meta_bytes=pickle.dumps([]),
        engine_rank=0,
        num_engines=1,
        meta_server_addr="127.0.0.1:1",
        weights_comm_backend="mooncake",
        debug_mode_config={},
        disable_pipeline=False,
        enable_colocate_mode=False,
        ipc_backend="cpu",
        model=None,
        model_context={
            "scheduler": scheduler,
            "infer_engine_config": InferenceConfig(),
        },
    )
    assert initialized == [scheduler.awes_weights_reader]
    assert initialized[0].kwargs["engine_name"] == "vllm"


def test_writer_backend_dispatch(monkeypatch):
    from awex.writer.weights_writer import get_weights_exchange_writer

    expected = object()
    monkeypatch.setattr(
        "awex.writer.mooncake_writer.MooncakeWeightsWriter", lambda engine: expected
    )
    assert (
        get_weights_exchange_writer(SimpleNamespace(comm_backend="mooncake"))
        is expected
    )


def test_colocate_snapshot_survives_offload(transport):
    writer, reader = transport(2), transport(0)
    parameters = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    original = parameters["weight"]
    expected = original[:, 1:3].clone()
    target = torch.zeros_like(original)
    op = _operation()

    def offload():
        assert not parameters
        assert all(t.device.type == "cpu" for t in writer.buffers.values())
        original.zero_()

    with ThreadPoolExecutor(max_workers=2) as pool:
        send = pool.submit(
            writer.send,
            parameters,
            TransferPlan(operations={0: [op]}),
            0,
            offload=True,
            on_ready=offload,
        )
        receive = pool.submit(
            reader.receive, {"weight": target}, TransferPlan(operations={2: [op]}), 0
        )
        receive.result(timeout=5)
        send.result(timeout=5)
    torch.testing.assert_close(target[:, 1:3], expected)


def test_native_tcp_loopback(monkeypatch):
    """Runs the real Mooncake TCP engine when its optional wheel is installed."""
    pytest.importorskip("mooncake.engine")
    monkeypatch.setenv("AWEX_MOONCAKE_HOST", "127.0.0.1")
    monkeypatch.setenv("AWEX_MOONCAKE_PROTOCOL", "tcp")
    monkeypatch.setenv("AWEX_DEVICE_TYPE", "cpu")
    meta = MemoryMeta()
    writer, reader = MooncakeTransport(meta, 2, 10), MooncakeTransport(meta, 0, 10)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    target = torch.zeros_like(source)
    op = _operation()
    with ThreadPoolExecutor(max_workers=2) as pool:
        send = pool.submit(
            writer.send, {"weight": source}, TransferPlan(operations={0: [op]}), 0
        )
        receive = pool.submit(
            reader.receive, {"weight": target}, TransferPlan(operations={2: [op]}), 0
        )
        receive.result(timeout=20)
        send.result(timeout=20)
    torch.testing.assert_close(target[:, 1:3], source[:, 1:3])
