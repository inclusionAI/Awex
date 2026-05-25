import asyncio
import sys
import types
from types import SimpleNamespace

import awex.sglang_patch as sglang_patch


def _drop_sglang_modules():
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            sys.modules.pop(name, None)


def _install_pkg(monkeypatch, name):
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_sglang_pkgs(monkeypatch):
    _install_pkg(monkeypatch, "sglang")
    _install_pkg(monkeypatch, "sglang.srt")
    _install_pkg(monkeypatch, "sglang.srt.managers")
    _install_pkg(monkeypatch, "sglang.srt.entrypoints")
    _install_pkg(monkeypatch, "sglang.srt.model_executor")


def test_import_hook_patches_future_io_struct(tmp_path, monkeypatch):
    _drop_sglang_modules()
    pkg_root = tmp_path / "sglang"
    managers = pkg_root / "srt" / "managers"
    managers.mkdir(parents=True)
    for path in (pkg_root, pkg_root / "srt", managers):
        (path / "__init__.py").write_text("")
    (managers / "io_struct.py").write_text(
        "def _check_all_req_types():\n"
        "    if 'ModelWorkerTask' in globals():\n"
        "        raise AssertionError('old checker sees ModelWorkerTask')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    sglang_patch.ensure_sglang_patched()
    import sglang.srt.managers.io_struct as io_struct

    task = io_struct.ModelWorkerTask(task_func=lambda: None, kwargs={"x": 1})
    output = io_struct.ModelWorkerTaskOutput(result="ok", tp_rank=1)

    assert task.kwargs == {"x": 1}
    assert output.result == "ok"
    io_struct._check_all_req_types()


def test_loaded_modules_are_patched(monkeypatch):
    _drop_sglang_modules()
    _install_sglang_pkgs(monkeypatch)
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)

    sglang_patch.ensure_sglang_patched()

    assert hasattr(io_struct, "ModelWorkerTask")
    assert hasattr(io_struct, "ModelWorkerTaskOutput")


def test_engine_module_gets_child_wrappers(monkeypatch):
    _drop_sglang_modules()
    _install_sglang_pkgs(monkeypatch)
    engine_mod = types.ModuleType("sglang.srt.entrypoints.engine")

    class Engine:
        def __init__(self):
            self.loop = asyncio.new_event_loop()
            self.tokenizer_manager = SimpleNamespace(
                execute_task_in_model_worker=lambda fn, **kwargs: _async_result(
                    fn, kwargs
                )
            )

    engine_mod.Engine = Engine
    engine_mod.run_scheduler_process = lambda *args, **kwargs: ("old", args, kwargs)
    engine_mod.run_data_parallel_controller_process = lambda *args, **kwargs: (
        "old-dp",
        args,
        kwargs,
    )
    monkeypatch.setitem(sys.modules, engine_mod.__name__, engine_mod)

    sglang_patch.ensure_sglang_patched()

    assert engine_mod.run_scheduler_process is sglang_patch.run_scheduler_awex
    assert engine_mod.run_data_parallel_controller_process is (
        sglang_patch.run_dp_controller_awex
    )
    assert hasattr(Engine, "execute_task_in_model_worker")


def test_child_scheduler_wrapper_installs_patch(monkeypatch):
    _drop_sglang_modules()
    _install_sglang_pkgs(monkeypatch)
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)
    scheduler_mod = types.ModuleType("sglang.srt.managers.scheduler")
    scheduler_mod.run_scheduler_process = lambda value: ("ran", value)
    monkeypatch.setitem(sys.modules, scheduler_mod.__name__, scheduler_mod)

    result = sglang_patch.run_scheduler_awex("task")

    assert result == ("ran", "task")
    assert hasattr(io_struct, "ModelWorkerTask")


def test_control_manager_orders_and_filters_dp(monkeypatch):
    _drop_sglang_modules()
    _install_sglang_pkgs(monkeypatch)
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)
    sglang_patch.ensure_sglang_patched()
    output_cls = io_struct.ModelWorkerTaskOutput

    class Communicator:
        async def __call__(self, task):
            return [
                output_cls(result="dp1", dp_rank=1, pp_rank=0, tp_rank=0),
                output_cls(result="tp1", dp_rank=0, pp_rank=0, tp_rank=1),
                output_cls(result="tp0", dp_rank=0, pp_rank=0, tp_rank=0),
            ]

    manager = SimpleNamespace(
        server_args=SimpleNamespace(enable_dp_attention=False, dp_size=2),
        model_worker_execute_task_group_communicator=Communicator(),
    )
    control = sglang_patch._AwexControlManager(
        context=None,
        port_args=None,
        handler=lambda obj: None,
        enabled=False,
        manager=manager,
    )

    result = asyncio.run(control.execute_task_in_model_worker(lambda: None))

    assert result == ["tp0", "tp1"]


def test_scheduler_worker_task_injects_context(monkeypatch):
    _drop_sglang_modules()
    _install_sglang_pkgs(monkeypatch)
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)
    sglang_patch.ensure_sglang_patched()
    task_cls = io_struct.ModelWorkerTask

    class Sender:
        def __init__(self):
            self.outputs = []

        def send_pyobj(self, obj):
            self.outputs.append(obj)

    class Worker:
        def execute_task_in_model_worker(self, task):
            context = task.kwargs["model_context"]
            assert context["scheduler"] is scheduler
            assert context["tp_rank"] == 2
            assert context["real_tp_worker"] == "real"
            return "worker-result"

    sender = Sender()
    scheduler = SimpleNamespace(
        tp_rank=2,
        pp_rank=1,
        tp_size=4,
        pp_size=2,
        dp_rank=0,
        dp_size=1,
        attn_tp_rank=2,
        attn_tp_size=4,
        attn_dp_rank=0,
        world_group=SimpleNamespace(world_size=8, rank=6, local_rank=2),
        server_args=SimpleNamespace(nnodes=1, context_parallel_size=1),
        tp_worker=SimpleNamespace(worker="real"),
        model_worker=Worker(),
        send_to_control_plane=sender,
    )

    result = sglang_patch._scheduler_execute_task(
        scheduler, task_cls(task_func=_noop_task, kwargs={})
    )

    assert result is None
    assert sender.outputs[0].result == "worker-result"
    assert sender.outputs[0].tp_rank == 2


async def _async_result(fn, kwargs):
    return [fn, kwargs]


def _noop_task():
    return None
