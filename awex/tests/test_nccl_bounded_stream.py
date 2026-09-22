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
from types import SimpleNamespace

import pytest
import torch

from awex.transfer import nccl_bounded_stream as colocate_transport
from awex.transfer.nccl_bounded_stream import (
    BoundedMemoryNcclColocateStreamBatchTransport,
)


@pytest.mark.parametrize("pending_slice_copy", [False, True])
def test_bounded_transport_defers_send_clones_until_execution(
    monkeypatch, pending_slice_copy
):
    """Building an AWEX transfer plan retains views instead of model-sized clones."""
    from awex.transfer import nccl_stream_batch
    from awex.util import device as device_util

    class _SourceTensor:
        def __init__(self) -> None:
            self.clone_calls = 0

        def clone(self):
            self.clone_calls += 1
            return self

    source = _SourceTensor()
    source.ready = not pending_slice_copy
    send_op = SimpleNamespace(
        send_shard_meta=SimpleNamespace(name="weight"),
        recv_rank=1,
    )
    send_plan = SimpleNamespace(operations={1: [send_op]})
    recv_op = SimpleNamespace(recv_shard_meta=SimpleNamespace(name="weight"))
    recv_plan = SimpleNamespace(operations={1: [recv_op]})
    recv_storage = torch.full((2, 4), float("nan"))
    recv_target = recv_storage[:, ::2]
    expected_recv = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)

    def _inspect_plan(
        transfer_rank,
        world_size,
        all_send_p2p_ops,
        all_recv_p2p_ops,
        weights_update_group,
        rank_coordinate,
        step_id,
    ) -> None:
        del (
            transfer_rank,
            world_size,
            weights_update_group,
            rank_coordinate,
            step_id,
        )
        assert all_send_p2p_ops[1][0][1].tensor is source
        assert source.clone_calls == 0
        # A materialized slice cannot be consumed on the transfer stream until
        # its asynchronous producer on the caller stream has completed.
        assert source.ready
        recv_buffer = all_recv_p2p_ops[1][0][1].tensor
        assert recv_buffer.is_contiguous()
        assert recv_buffer.data_ptr() != recv_target.data_ptr()
        recv_buffer.copy_(expected_recv)

    monkeypatch.setattr(transport, "_validate_pack_config", lambda group: None)
    transport.execute_recursive_partition_stream_transfer = _inspect_plan
    monkeypatch.setattr(
        nccl_stream_batch,
        "hang_detector",
        SimpleNamespace(submit=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "awex.transfer.nccl_comm.validate_rank_mappings", lambda *args: None
    )
    monkeypatch.setattr(
        "awex.transfer.transfer_plan.slice_tensor",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(
        device_util, "synchronize", lambda: setattr(source, "ready", True)
    )
    monkeypatch.setattr(
        torch.distributed,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    transport.update_weights_in_colocate_mode(
        train_to_infer_device_mapping={0: 0, 1: 1},
        infer_to_train_device_mapping={0: 0, 1: 1},
        transfer_rank=0,
        rank_coordinate="0-0-0",
        world_size=2,
        send_transfer_plan=send_plan,
        recv_transfer_plan=recv_plan,
        weights_update_group=object(),
        send_parameters={"weight": source},
        recv_parameters={"weight": recv_target},
        step_id=1,
    )

    assert source.clone_calls == 0
    torch.testing.assert_close(recv_target, expected_recv, rtol=0, atol=0)
    assert torch.isnan(recv_storage[:, 1::2]).all()


def test_bounded_transport_releases_each_send_clone_batch(monkeypatch):
    """Only one send tensor per active peer remains live during P2P execution."""
    from awex.util import device as device_util

    counters = {"live": 0, "max_live": 0, "clones": 0, "syncs": 0}

    class _Clone:
        def __init__(self) -> None:
            counters["live"] += 1
            counters["max_live"] = max(counters["max_live"], counters["live"])

        def __del__(self) -> None:
            counters["live"] -= 1

    class _SourceTensor:
        def clone(self):
            counters["clones"] += 1
            return _Clone()

    class _Work:
        def __init__(self, tensor) -> None:
            self.tensor = tensor

        def wait(self) -> None:
            self.tensor = None

    def _isend(tensor, peer, group):
        del peer, group
        return _Work(tensor)

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        device_util,
        "synchronize",
        lambda: counters.__setitem__("syncs", counters["syncs"] + 1),
    )

    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [object(), object()]
    transport._expert_pack_config = (1, 64 * 1024 * 1024)
    ops = {
        peer: [
            (
                SimpleNamespace(recv_shard_meta=SimpleNamespace(dtype=None)),
                SimpleNamespace(
                    op=_isend,
                    tensor=_SourceTensor(),
                    peer=peer,
                    group=object(),
                ),
            )
            for _ in range(3)
        ]
        for peer in (1, 2)
    }

    count = transport._execute_ops_concurrent(ops, range(1, 3))

    assert count == 6
    assert counters == {
        "live": 0,
        "max_live": 2,
        "clones": 6,
        "syncs": 3,
    }


def test_bounded_transport_casts_send_to_receiver_dtype(monkeypatch):
    """P2P sends use the receiver dtype so NCCL wire sizes match."""
    from awex.util import device as device_util

    sent = []

    class _Work:
        def wait(self) -> None:
            return None

    def _isend(tensor, peer, group):
        del peer, group
        sent.append(tensor)
        return _Work()

    source = torch.ones(2, dtype=torch.bfloat16)
    plan_op = SimpleNamespace(
        recv_shard_meta=SimpleNamespace(dtype=torch.float32),
    )
    p2p_op = SimpleNamespace(
        op=_isend,
        tensor=source,
        peer=1,
        group=object(),
    )

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(device_util, "synchronize", lambda: None)

    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [object()]
    transport._expert_pack_config = (1, 64 * 1024 * 1024)

    count = transport._execute_ops_concurrent(
        {1: [(plan_op, p2p_op)]},
        range(1, 2),
    )

    assert count == 1
    assert len(sent) == 1
    assert sent[0].dtype == torch.float32


def test_bounded_transport_prepares_send_on_transfer_stream(monkeypatch):
    """Send clones are ordered on the same stream as their NCCL operation."""
    from awex.util import device as device_util

    state = {"active_stream": None}
    transfer_stream = object()

    class _StreamContext:
        def __enter__(self):
            state["active_stream"] = transfer_stream

        def __exit__(self, exc_type, exc_value, traceback):
            state["active_stream"] = None

    class _SourceTensor:
        dtype = torch.bfloat16

        def clone(self):
            assert state["active_stream"] is transfer_stream
            return self

    class _Work:
        def wait(self) -> None:
            return None

    def _isend(tensor, peer, group):
        del tensor, peer, group
        assert state["active_stream"] is transfer_stream
        return _Work()

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: _StreamContext())
    monkeypatch.setattr(device_util, "synchronize", lambda: None)

    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [transfer_stream]
    transport._expert_pack_config = (1, 64 * 1024 * 1024)
    plan_op = SimpleNamespace(
        recv_shard_meta=SimpleNamespace(dtype=torch.bfloat16),
    )
    p2p_op = SimpleNamespace(
        op=_isend,
        tensor=_SourceTensor(),
        peer=1,
        group=object(),
    )

    count = transport._execute_ops_concurrent(
        {1: [(plan_op, p2p_op)]},
        range(1, 2),
    )

    assert count == 1
    assert state["active_stream"] is None


def test_expert_pack_defaults_to_validated_optimized_batch(monkeypatch):
    monkeypatch.delenv("AWEX_EXPERT_PACK_OPS", raising=False)
    monkeypatch.delenv("AWEX_EXPERT_PACK_MB", raising=False)

    assert colocate_transport._expert_pack_limits() == (
        64,
        64 * 1024 * 1024,
    )


def test_bounded_transport_partitions_only_compatible_expert_ops():
    """Expert packs respect FIFO, byte limits, and non-expert boundaries."""
    group = object()

    def _send(tensor, peer, group):
        del tensor, peer, group

    def _operation(param_class, dtype=torch.bfloat16):
        plan_op = SimpleNamespace(
            param_class=param_class,
            overlap_shape=(2,),
            recv_shard_meta=SimpleNamespace(dtype=dtype),
        )
        p2p_op = SimpleNamespace(
            op=_send,
            tensor=torch.zeros(2, dtype=dtype),
            peer=1,
            group=group,
        )
        return plan_op, p2p_op

    operations = [
        _operation("expert"),
        _operation("expert"),
        _operation("expert"),
        _operation("dense_other"),
        _operation("expert", torch.float32),
    ]

    batches = (
        BoundedMemoryNcclColocateStreamBatchTransport._partition_expert_operations(
            operations,
            max_pack_ops=4,
            max_pack_bytes=8,
        )
    )

    assert [len(batch) for batch in batches] == [2, 1, 1, 1]
    flattened = [item for batch in batches for item in batch]
    assert all(
        actual_plan is expected_plan and actual_p2p is expected_p2p
        for (actual_plan, actual_p2p), (expected_plan, expected_p2p) in zip(
            flattened, operations
        )
    )


def test_bounded_transport_packs_expert_sends(monkeypatch):
    """Consecutive expert sends share one flat wire tensor per bounded pack."""
    from awex.util import device as device_util

    sent = []
    syncs = []

    class _Work:
        def wait(self) -> None:
            return None

    def _isend(tensor, peer, group):
        del peer, group
        sent.append(tensor.clone())
        return _Work()

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    monkeypatch.setattr(device_util, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(device_util, "synchronize", lambda: syncs.append(None))

    group = object()
    source_tensors = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([5.0, 6.0]),
        torch.tensor([[7.0, 8.0]]),
    ]
    operations = []
    for tensor in source_tensors:
        plan_op = SimpleNamespace(
            param_class="expert",
            overlap_shape=tuple(tensor.shape),
            recv_shard_meta=SimpleNamespace(dtype=torch.float32),
        )
        p2p_op = SimpleNamespace(
            op=_isend,
            tensor=tensor,
            peer=1,
            group=group,
        )
        operations.append((plan_op, p2p_op))

    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [object()]
    transport._expert_pack_config = (2, 64 * 1024 * 1024)
    transport._expert_pack_stats = {
        "logical_ops": 0,
        "wire_ops": 0,
        "packed_wire_ops": 0,
    }

    count = transport._execute_ops_concurrent({1: operations}, range(1, 2))

    assert count == 2
    assert len(syncs) == 2
    assert transport._expert_pack_stats == {
        "logical_ops": 4,
        "wire_ops": 2,
        "packed_wire_ops": 2,
    }
    torch.testing.assert_close(
        sent[0], torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        sent[1], torch.tensor([5.0, 6.0, 7.0, 8.0]), rtol=0.0, atol=0.0
    )


def test_bounded_transport_unpacks_expert_receives(monkeypatch):
    """A flat expert receive is copied back into its original tensor views."""
    from awex.util import device as device_util

    payload = torch.tensor([1.0, 2.0, 3.0, 4.0])

    class _Work:
        def wait(self) -> None:
            return None

    def _irecv(tensor, peer, group):
        del peer, group
        tensor.copy_(payload)
        return _Work()

    monkeypatch.setattr(torch.distributed, "irecv", _irecv)
    monkeypatch.setattr(device_util, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(device_util, "synchronize", lambda: None)

    group = object()
    destinations = [torch.zeros(2), torch.zeros(1, 2)]
    operations = []
    for tensor in destinations:
        plan_op = SimpleNamespace(
            param_class="expert",
            overlap_shape=tuple(tensor.shape),
            recv_shard_meta=SimpleNamespace(dtype=torch.float32),
        )
        p2p_op = SimpleNamespace(
            op=_irecv,
            tensor=tensor,
            peer=1,
            group=group,
        )
        operations.append((plan_op, p2p_op))

    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
    transport._stream_pool = [object()]
    transport._expert_pack_config = (2, 64 * 1024 * 1024)
    transport._expert_pack_stats = None

    count = transport._execute_ops_concurrent({1: operations}, range(1, 2))

    assert count == 1
    torch.testing.assert_close(
        destinations[0], torch.tensor([1.0, 2.0]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        destinations[1], torch.tensor([[3.0, 4.0]]), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("mismatch", [False, True])
def test_wire_config_rejects_rank_disagreement(monkeypatch, mismatch):
    transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
    transport._expert_pack_config = (64, 64 * 1024 * 1024)
    group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "get_backend", lambda pg: "gloo")

    def reduce(tensor, op, group):
        calls.append(group)
        if mismatch and op == torch.distributed.ReduceOp.MIN:
            tensor[0] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    expected = (
        pytest.raises(ValueError, match="differs across ranks")
        if mismatch
        else nullcontext()
    )
    with expected:
        transport._validate_pack_config(group)
    assert calls == [group, group]


def _check_pack_config_worker(rank, rendezvous, mismatch):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        transport = object.__new__(BoundedMemoryNcclColocateStreamBatchTransport)
        transport._expert_pack_config = (
            64 + (rank if mismatch else 0),
            64 * 1024 * 1024,
        )
        if mismatch:
            with pytest.raises(ValueError, match="differs across ranks"):
                transport._validate_pack_config(dist.group.WORLD)
        else:
            transport._validate_pack_config(dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", [False, True])
def test_pack_config_agrees_or_fails_on_all_real_gloo_ranks(tmp_path, mismatch):
    import torch.multiprocessing as mp

    mp.spawn(
        _check_pack_config_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", mismatch),
        nprocs=2,
        join=True,
    )
