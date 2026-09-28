"""CPU contracts for AWEX's SGLang callback transport."""

import asyncio
import pickle
import sys
from types import ModuleType, SimpleNamespace

import cloudpickle
import pytest

from awex import sglang_patch as patch


@pytest.fixture
def client(monkeypatch):
    io = ModuleType("sglang.srt.managers.io_struct")
    managers = ModuleType("sglang.srt.managers")
    managers.io_struct = io
    monkeypatch.setitem(sys.modules, "sglang.srt.managers", managers)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io)
    manager = SimpleNamespace(
        rid_to_state={},
        auto_create_handle_loop=lambda: None,
        _result_dispatcher=lambda obj: obj,
    )
    patch._install_tokenizer(manager)
    engine = SimpleNamespace(
        tokenizer_manager=manager,
        server_args=SimpleNamespace(
            tp_size=2, pp_size=2, dp_size=1, enable_dp_attention=False
        ),
    )
    sent = []

    async def send(task):
        sent.append(task)
        fn, kwargs = cloudpickle.loads(task.payload)
        # Deliberately return replies in reverse order, including a duplicate.
        for pp, tp in [(1, 1), (0, 1), (0, 1), (1, 0), (0, 0)]:
            manager._result_dispatcher(
                patch._WorkerResult(
                    task.task_id, (0, pp, tp), pickle.dumps(fn(pp=pp, tp=tp, **kwargs))
                )
            )

    manager.send_to_scheduler = SimpleNamespace(send_pyobj=send)
    return engine, sent, io


def test_callback_closure_and_rank_order(client):
    engine, sent, _ = client
    offset = 10
    result = asyncio.run(
        patch._execute_task_async(engine, lambda pp, tp: offset + pp * 2 + tp)
    )
    assert result == [10, 11, 12, 13]
    assert len(sent) == 1
    assert not engine.tokenizer_manager._awex_pending_tasks


def test_msgpack_envelope_is_explicit(client):
    engine, sent, io = client
    io.wrap_as_pickle = lambda obj: ("pickle", obj)

    async def send(socket, envelope):
        assert envelope[0] == "pickle"
        await socket.send_pyobj(envelope[1])

    io.async_sock_send = send
    assert asyncio.run(patch._execute_task_async(engine, lambda **kw: 42)) == [42] * 4
    assert len(sent) == 1


def test_busy_generation_rejected_before_dispatch(client):
    engine, sent, _ = client
    engine.tokenizer_manager.rid_to_state["running"] = object()
    with pytest.raises(RuntimeError, match="idle generation"):
        asyncio.run(patch._execute_task_async(engine, lambda **kw: None))
    assert not sent


def test_rank_failure_is_drained_and_next_task_works(client):
    engine, _, _ = client
    manager = engine.tokenizer_manager
    original_send = manager.send_to_scheduler.send_pyobj

    async def fail(task):
        for pp in range(2):
            for tp in range(2):
                manager._result_dispatcher(
                    patch._WorkerResult(task.task_id, (0, pp, tp), error="broken")
                )

    async def run():
        manager.send_to_scheduler.send_pyobj = fail
        with pytest.raises(RuntimeError, match=r"rank \(0, 1, 1\).*broken"):
            await patch._execute_task_async(engine, lambda: None)
        manager.send_to_scheduler.send_pyobj = original_send
        assert await patch._execute_task_async(engine, lambda **kw: 7) == [7] * 4

    asyncio.run(run())


def test_worker_payload_isolated_and_errors_returned(monkeypatch):
    ranks = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
    )
    scheduler = SimpleNamespace(
        ps=ranks,
        world_group=SimpleNamespace(world_size=1, rank=0, local_rank=0),
        server_args=SimpleNamespace(nnodes=1),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model="model")),
        _awex_result_socket=object(),
    )
    replies = []
    monkeypatch.setattr(patch, "_send", lambda socket, obj: replies.append(obj))

    def callback(values, model, model_runner, model_context):
        values.append(model)
        assert model_context["scheduler"].tp_worker.model_runner is model_runner
        return values

    task = patch._WorkerTask("task", cloudpickle.dumps((callback, {"values": []})))
    patch._execute_worker_task(scheduler, task)
    patch._execute_worker_task(scheduler, task)
    assert [pickle.loads(r.payload) for r in replies] == [["model"], ["model"]]
    bad = patch._WorkerTask("bad", cloudpickle.dumps((lambda **kw: 1 / 0, {})))
    patch._execute_worker_task(scheduler, bad)
    assert "ZeroDivisionError" in replies[-1].error


def test_pp_forwards_before_callback_exactly_once(monkeypatch):
    managers = ModuleType("sglang.srt.managers")
    module = ModuleType("sglang.srt.managers.scheduler")
    pp_module = ModuleType("sglang.srt.managers.scheduler_pp_mixin")
    managers.scheduler = module
    managers.scheduler_pp_mixin = pp_module
    for name, value in (
        ("sglang.srt.managers", managers),
        ("sglang.srt.managers.scheduler", module),
        ("sglang.srt.managers.scheduler_pp_mixin", pp_module),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    events = []
    pp_module.point_to_point_pyobj = lambda data, **kw: events.append("forward") or []

    class Scheduler:
        def __init__(self, server_args, port_args):
            self.ps = SimpleNamespace(
                pp_rank=0, pp_size=2, attn_tp_rank=0, attn_cp_rank=0
            )
            self.send_req_work = []

        def process_input_requests(self, requests):
            events.append("ordinary")

        def _pp_send_pyobj_to_next_stage(self, data, **kw):
            return pp_module.point_to_point_pyobj(data, **kw)

        def _pp_commit_comm_work(self, work):
            pass

    module.Scheduler = Scheduler
    monkeypatch.setattr(
        patch, "_execute_worker_task", lambda *args: events.append("task")
    )
    patch._patch_scheduler()
    scheduler = Scheduler(None, SimpleNamespace(tokenizer_ipc_name="inproc://test"))
    requests = [patch._WorkerTask("task", b"")]
    try:
        scheduler.process_input_requests(requests)
        scheduler._pp_send_pyobj_to_next_stage(requests, async_send=True)
        assert events == ["forward", "task"]
        scheduler._pp_send_pyobj_to_next_stage([], async_send=True)
        assert events == ["forward", "task", "forward"]
    finally:
        scheduler._awex_result_socket.close(linger=0)
        scheduler._awex_zmq_context.term()
