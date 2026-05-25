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

from __future__ import annotations

import asyncio
import copy
import dataclasses
import importlib
import importlib.abc
import importlib.machinery
import logging
import pickle
import sys
import tempfile
import types
from typing import Any, Callable

logger = logging.getLogger(__name__)


_PATCH_MARK = "__awex_sglang_patched__"
_HOOK_MARK = "__awex_sglang_hook_installed__"
_TASK_COMM_MARK = "__awex_model_task_comm_installed__"
_CONTROL_TASK = "_awex_control_task"
_TARGET_MODULES = {
    "sglang.srt.entrypoints.engine",
    "sglang.srt.managers.data_parallel_controller",
    "sglang.srt.managers.io_struct",
    "sglang.srt.managers.multi_tokenizer_mixin",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.tokenizer_communicator_mixin",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.tp_worker",
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.server_args",
}

_finder: _SGLangPatchFinder | None = None


def ensure_sglang_patched() -> None:
    """Install Awex's SGLang worker-task patch.

    The hook is deliberately process-local. Parent processes install it before
    SGLang starts subprocesses, and wrapped subprocess targets install it again
    before importing and running SGLang child process entrypoints.
    """

    _install_import_hook()
    _patch_loaded_modules()


def run_scheduler_awex(*args, **kwargs):
    ensure_sglang_patched()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_dp_controller_awex(*args, **kwargs):
    ensure_sglang_patched()
    from sglang.srt.managers.data_parallel_controller import (
        run_data_parallel_controller_process,
    )

    return run_data_parallel_controller_process(*args, **kwargs)


class _SGLangPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _SGLangPatchLoader):
            return spec
        spec.loader = _SGLangPatchLoader(spec.loader)
        return spec


class _SGLangPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_module(module)


class _AwexCommunicator:
    def __init__(self, sender, fan_out: int, mode: str = "queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event = None
        self._result_values = None

    async def __call__(self, obj):
        if self._result_event is not None:
            raise RuntimeError(
                "SGLang model-worker task execution already has an in-flight call."
            )
        self._result_event = asyncio.Event()
        self._result_values = []
        if obj is not None:
            self._sender.send_pyobj(obj)
        await self._result_event.wait()
        result_values = copy.deepcopy(self._result_values)
        self._result_event = None
        self._result_values = None
        return result_values

    def handle_recv(self, recv_obj):
        if self._result_values is None or self._result_event is None:
            raise RuntimeError("Received SGLang model-worker task output too early.")
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()


class _AwexControlManager:
    """Control-plane receiver and model-worker task dispatcher.

    This mirrors the newer SGLang implementation used by ant-sglang: worker task
    results return over an independent ZMQ control plane instead of collecting
    outputs through a model-worker collective. That separation prevents Awex
    weight-exchange control tasks from adding a result all-gather while inference
    kernels may already be using distributed communication.
    """

    def __init__(
        self,
        context,
        port_args,
        handler: Callable[[Any], Any],
        enabled: bool = True,
        manager: Any | None = None,
    ):
        self._handler = handler
        self._manager = manager
        self._recv_socket = None
        if not enabled:
            return
        try:
            import zmq
            from sglang.srt.utils import get_zmq_socket

            control_ipc = _ensure_control_ipc(
                port_args, getattr(manager, "server_args", None)
            )
            self._recv_socket = get_zmq_socket(context, zmq.PULL, control_ipc, True)
        except Exception as exc:
            logger.warning("Failed to create SGLang Awex control socket: %s", exc)

    async def handle_loop(self):
        if self._recv_socket is None:
            return
        while True:
            recv_obj = await self._recv_socket.recv_pyobj()
            result = self._handler(recv_obj)
            if asyncio.iscoroutine(result):
                await result

    async def execute_task_in_model_worker(self, task_func: Callable, **kwargs):
        manager = self._manager
        if manager is None:
            raise RuntimeError("SGLang Awex ControlManager is not bound.")
        _ensure_model_comm(manager, manager.server_args)
        task_cls = _get_io_type("ModelWorkerTask")
        task = task_cls(task_func=task_func, kwargs=kwargs)
        tokenizer_ipc = getattr(manager, "tokenizer_ipc_name", None)
        if tokenizer_ipc is None:
            tokenizer_ipc = getattr(
                getattr(manager, "port_args", None), "tokenizer_ipc_name", None
            )
        if tokenizer_ipc is not None and hasattr(task, "http_worker_ipc"):
            task.http_worker_ipc = tokenizer_ipc
        communicator = manager.model_worker_execute_task_group_communicator
        results = await communicator(task)
        return _ordered_task_results(manager, results)


def _install_import_hook() -> None:
    global _finder
    if getattr(sys, _HOOK_MARK, False):
        return
    _finder = _SGLangPatchFinder()
    sys.meta_path.insert(0, _finder)
    setattr(sys, _HOOK_MARK, True)


def _patch_loaded_modules() -> None:
    for name, module in list(sys.modules.items()):
        if name in _TARGET_MODULES and isinstance(module, types.ModuleType):
            _patch_module(module)


def _patch_module(module: types.ModuleType) -> None:
    name = module.__name__
    if name == "sglang.srt.managers.io_struct":
        _patch_io_struct(module)
    elif name == "sglang.srt.server_args":
        _patch_server_args(module)
    elif name == "sglang.srt.managers.tokenizer_communicator_mixin":
        _patch_tokenizer_comm(module)
    elif name == "sglang.srt.managers.tokenizer_manager":
        _patch_tokenizer_manager(module)
    elif name == "sglang.srt.managers.multi_tokenizer_mixin":
        _patch_multi_tokenizer(module)
    elif name == "sglang.srt.managers.scheduler":
        _patch_scheduler(module)
    elif name == "sglang.srt.managers.tp_worker":
        _patch_tp_worker(module)
    elif name == "sglang.srt.model_executor.model_runner":
        _patch_model_runner(module)
    elif name == "sglang.srt.entrypoints.engine":
        _patch_engine(module)
    elif name == "sglang.srt.managers.data_parallel_controller":
        _patch_dp_controller(module)


def _patch_io_struct(module: types.ModuleType) -> None:
    if not hasattr(module, "ModelWorkerTask"):
        module.ModelWorkerTask = _make_dataclass(
            module.__name__,
            "ModelWorkerTask",
            {
                "task_func": Callable,
                "kwargs": dict,
                "http_worker_ipc": Any,
            },
            {"http_worker_ipc": None},
        )
    if not hasattr(module, "ModelWorkerTaskOutput"):
        module.ModelWorkerTaskOutput = _make_dataclass(
            module.__name__,
            "ModelWorkerTaskOutput",
            {
                "result": Any,
                "tp_rank": Any,
                "pp_rank": Any,
                "dp_rank": Any,
                "http_worker_ipc": Any,
            },
            {
                "tp_rank": None,
                "pp_rank": None,
                "dp_rank": None,
                "http_worker_ipc": None,
            },
        )
    _patch_io_checker(module)


def _make_dataclass(
    module_name: str,
    class_name: str,
    annotations: dict[str, Any],
    defaults: dict[str, Any],
):
    namespace = {"__module__": module_name, "__annotations__": annotations}
    namespace.update(defaults)
    return dataclasses.dataclass(type(class_name, (), namespace))


def _patch_io_checker(module: types.ModuleType) -> None:
    check_fn = getattr(module, "_check_all_req_types", None)
    if not callable(check_fn) or getattr(check_fn, _PATCH_MARK, False):
        return

    def _awex_check_all_req_types():
        hidden = {}
        for name in ("ModelWorkerTask", "ModelWorkerTaskOutput"):
            if hasattr(module, name):
                hidden[name] = getattr(module, name)
                delattr(module, name)
        try:
            return check_fn()
        finally:
            for name, value in hidden.items():
                setattr(module, name, value)

    setattr(_awex_check_all_req_types, _PATCH_MARK, True)
    module._check_all_req_types = _awex_check_all_req_types


def _patch_server_args(module: types.ModuleType) -> None:
    port_args_cls = getattr(module, "PortArgs", None)
    if port_args_cls is None:
        return
    init_new = getattr(port_args_cls, "init_new", None)
    if not callable(init_new) or getattr(init_new, _PATCH_MARK, False):
        return

    def _awex_init_new(*args, **kwargs):
        port_args = init_new(*args, **kwargs)
        server_args = args[0] if args else kwargs.get("server_args")
        _ensure_control_ipc(port_args, server_args)
        return port_args

    setattr(_awex_init_new, _PATCH_MARK, True)
    port_args_cls.init_new = staticmethod(_awex_init_new)


def _patch_tokenizer_comm(module: types.ModuleType) -> None:
    mixin = getattr(module, "TokenizerCommunicatorMixin", None)
    if mixin is None:
        return
    if not hasattr(mixin, "_get_model_worker_task_fan_out"):
        mixin._get_model_worker_task_fan_out = _get_task_fan_out
    init_communicators = getattr(mixin, "init_communicators", None)
    if callable(init_communicators) and not getattr(
        init_communicators, _PATCH_MARK, False
    ):

        def _awex_init_communicators(self, server_args):
            result = init_communicators(self, server_args)
            _ensure_model_comm(self, server_args)
            return result

        setattr(_awex_init_communicators, _PATCH_MARK, True)
        mixin.init_communicators = _awex_init_communicators


def _patch_tokenizer_manager(module: types.ModuleType) -> None:
    manager_cls = getattr(module, "TokenizerManager", None)
    if manager_cls is None:
        return
    _patch_tokenizer_init(manager_cls)
    _patch_tokenizer_execute(manager_cls)
    _patch_tokenizer_loop(manager_cls)
    _patch_tokenizer_output(manager_cls)


def _patch_tokenizer_init(manager_cls) -> None:
    init_fn = getattr(manager_cls, "__init__", None)
    if not callable(init_fn) or getattr(init_fn, _PATCH_MARK, False):
        return

    def _awex_init(self, *args, **kwargs):
        result = init_fn(self, *args, **kwargs)
        server_args = args[0] if args else kwargs.get("server_args")
        port_args = args[1] if len(args) > 1 else kwargs.get("port_args")
        if server_args is None:
            server_args = getattr(self, "server_args", None)
        if port_args is None:
            port_args = getattr(self, "port_args", None)
        _ensure_model_comm(self, server_args)
        _attach_control_manager(self, server_args, port_args)
        return result

    setattr(_awex_init, _PATCH_MARK, True)
    manager_cls.__init__ = _awex_init


def _patch_tokenizer_execute(manager_cls) -> None:
    existing = getattr(manager_cls, "execute_task_in_model_worker", None)
    if callable(existing) and getattr(existing, _PATCH_MARK, False):
        return

    async def execute_task_in_model_worker(self, task_func: Callable, **kwargs):
        auto_loop = getattr(self, "auto_create_handle_loop", None)
        if callable(auto_loop):
            auto_loop()
        if not hasattr(self, "control_manager"):
            _attach_control_manager(
                self,
                getattr(self, "server_args", None),
                getattr(self, "port_args", None),
            )
        return await self.control_manager.execute_task_in_model_worker(
            task_func, **kwargs
        )

    setattr(execute_task_in_model_worker, _PATCH_MARK, True)
    manager_cls.execute_task_in_model_worker = execute_task_in_model_worker


def _patch_tokenizer_loop(manager_cls) -> None:
    loop_fn = getattr(manager_cls, "auto_create_handle_loop", None)
    if not callable(loop_fn) or getattr(loop_fn, _PATCH_MARK, False):
        return

    def _awex_auto_create_loop(self, *args, **kwargs):
        result = loop_fn(self, *args, **kwargs)
        _start_control_loop(self)
        return result

    setattr(_awex_auto_create_loop, _PATCH_MARK, True)
    manager_cls.auto_create_handle_loop = _awex_auto_create_loop


def _patch_tokenizer_output(manager_cls) -> None:
    output_fn = getattr(manager_cls, "_handle_control_plane_output", None)
    if callable(output_fn) and getattr(output_fn, _PATCH_MARK, False):
        return
    if callable(output_fn):
        return

    def _handle_control_plane_output(self, recv_obj):
        dispatcher = getattr(self, "_result_dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("SGLang TokenizerManager has no result dispatcher.")
        return dispatcher(recv_obj)

    setattr(_handle_control_plane_output, _PATCH_MARK, True)
    manager_cls._handle_control_plane_output = _handle_control_plane_output


def _patch_multi_tokenizer(module: types.ModuleType) -> None:
    router_cls = getattr(module, "MultiTokenizerRouter", None)
    if router_cls is None:
        return
    _patch_tokenizer_execute(router_cls)


def _patch_engine(module: types.ModuleType) -> None:
    if hasattr(module, "run_scheduler_process"):
        module.run_scheduler_process = run_scheduler_awex
    if hasattr(module, "run_data_parallel_controller_process"):
        module.run_data_parallel_controller_process = run_dp_controller_awex

    engine_cls = getattr(module, "Engine", None)
    if engine_cls is None:
        return
    async_execute = getattr(engine_cls, "async_execute_task_in_model_worker", None)
    if not callable(async_execute):

        async def async_execute_task_in_model_worker(
            self, task_func: Callable, **kwargs
        ):
            return await self.tokenizer_manager.execute_task_in_model_worker(
                task_func, **kwargs
            )

        engine_cls.async_execute_task_in_model_worker = (
            async_execute_task_in_model_worker
        )

    execute = getattr(engine_cls, "execute_task_in_model_worker", None)
    if callable(execute) and getattr(execute, _PATCH_MARK, False):
        return

    def execute_task_in_model_worker(self, task_func: Callable, **kwargs):
        loop = getattr(self, "loop", None)
        if loop is None:
            loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.execute_task_in_model_worker(task_func, **kwargs)
        )

    setattr(execute_task_in_model_worker, _PATCH_MARK, True)
    engine_cls.execute_task_in_model_worker = execute_task_in_model_worker


def _patch_dp_controller(module: types.ModuleType) -> None:
    if hasattr(module, "run_scheduler_process"):
        module.run_scheduler_process = run_scheduler_awex


def _patch_scheduler(module: types.ModuleType) -> None:
    scheduler_cls = getattr(module, "Scheduler", None)
    if scheduler_cls is None:
        return
    _patch_scheduler_init(scheduler_cls)
    _patch_scheduler_sockets(scheduler_cls)
    scheduler_cls.execute_task_in_model_worker = _scheduler_execute_task


def _patch_scheduler_init(scheduler_cls) -> None:
    init_fn = getattr(scheduler_cls, "__init__", None)
    if not callable(init_fn) or getattr(init_fn, _PATCH_MARK, False):
        return

    def _awex_scheduler_init(self, *args, **kwargs):
        result = init_fn(self, *args, **kwargs)
        _ensure_scheduler_dispatch(self)
        return result

    setattr(_awex_scheduler_init, _PATCH_MARK, True)
    scheduler_cls.__init__ = _awex_scheduler_init


def _patch_scheduler_sockets(scheduler_cls) -> None:
    sockets_fn = getattr(scheduler_cls, "init_sockets", None)
    if not callable(sockets_fn) or getattr(sockets_fn, _PATCH_MARK, False):
        return

    def _awex_init_sockets(self, server_args, port_args):
        result = sockets_fn(self, server_args, port_args)
        _attach_control_sender(self, server_args, port_args)
        return result

    setattr(_awex_init_sockets, _PATCH_MARK, True)
    scheduler_cls.init_sockets = _awex_init_sockets


def _scheduler_execute_task(self, task_spec):
    task_spec = pickle.loads(pickle.dumps(task_spec))
    task_spec.kwargs = dict(getattr(task_spec, "kwargs", None) or {})
    task_spec.kwargs["model_context"] = _build_model_context(self)
    worker = getattr(self, "model_worker", None)
    if worker is None:
        worker = getattr(self, "tp_worker", None)
    if worker is None or not hasattr(worker, "execute_task_in_model_worker"):
        raise RuntimeError("SGLang scheduler has no model worker for Awex task.")
    result = worker.execute_task_in_model_worker(task_spec)
    output_cls = _get_io_type("ModelWorkerTaskOutput")
    output = output_cls(
        result=result,
        tp_rank=getattr(self, "tp_rank", 0),
        pp_rank=getattr(self, "pp_rank", 0),
        dp_rank=getattr(self, "dp_rank", 0),
        http_worker_ipc=getattr(task_spec, "http_worker_ipc", None),
    )
    sender = getattr(self, "send_to_control_plane", None)
    if sender is not None:
        sender.send_pyobj(output)
        return None
    return output


def _patch_tp_worker(module: types.ModuleType) -> None:
    for class_name in ("BaseTpWorker", "TpModelWorker"):
        cls = getattr(module, class_name, None)
        if cls is not None:
            cls.execute_task_in_model_worker = _tp_worker_execute_task


def _tp_worker_execute_task(self, task_spec, models=None):
    runner = getattr(self, "model_runner", None)
    if runner is None:
        raise RuntimeError("SGLang TP worker has no model_runner for Awex task.")
    return runner.execute_task_in_model_worker(task_spec, models=models)


def _patch_model_runner(module: types.ModuleType) -> None:
    runner_cls = getattr(module, "ModelRunner", None)
    if runner_cls is not None:
        runner_cls.execute_task_in_model_worker = _runner_execute_task


def _runner_execute_task(self, task_spec, models=None):
    task_func = task_spec.task_func
    kwargs = dict(getattr(task_spec, "kwargs", None) or {})
    kwargs["model"] = models or self.model
    kwargs["model_runner"] = self
    return task_func(**kwargs)


def _attach_control_manager(manager, server_args, port_args) -> None:
    if server_args is None:
        server_args = getattr(manager, "server_args", None)
    if port_args is None:
        port_args = getattr(manager, "port_args", None)
    if port_args is None:
        return
    _ensure_control_ipc(port_args, server_args)
    existing = getattr(manager, "control_manager", None)
    if existing is not None:
        if hasattr(existing, "_manager"):
            existing._manager = manager
        return
    try:
        import zmq.asyncio

        context = zmq.asyncio.Context(1)
    except Exception as exc:
        logger.warning("Failed to create SGLang Awex control context: %s", exc)
        return
    handler = getattr(manager, "_handle_control_plane_output", None)
    if handler is None:

        def handler(obj):
            return manager._result_dispatcher(obj)

    enabled = not getattr(manager, "disable_control_plane_receiver", False)
    manager.control_manager = _AwexControlManager(
        context,
        port_args,
        handler=handler,
        enabled=enabled,
        manager=manager,
    )


def _attach_control_sender(scheduler, server_args, port_args) -> None:
    if getattr(scheduler, "send_to_control_plane", None) is not None:
        return
    try:
        import zmq
        from sglang.srt.utils import get_zmq_socket

        control_ipc = _ensure_control_ipc(port_args, server_args)
        context = zmq.Context(1)
        scheduler.send_to_control_plane = get_zmq_socket(
            context, zmq.PUSH, control_ipc, False
        )
    except Exception as exc:
        logger.warning("Failed to create SGLang Awex control sender: %s", exc)


def _ensure_model_comm(manager, server_args) -> None:
    if server_args is None:
        server_args = getattr(manager, "server_args", None)
    if server_args is None or not hasattr(manager, "send_to_scheduler"):
        return
    communicator = getattr(
        manager, "model_worker_execute_task_group_communicator", None
    )
    if communicator is None:
        comm_cls = _get_comm_cls()
        fan_out = _get_task_fan_out(manager, server_args)
        manager.model_worker_execute_task_group_communicator = comm_cls(
            manager.send_to_scheduler, fan_out
        )
    _install_comm_dispatcher(manager)


def _install_comm_dispatcher(manager) -> None:
    if getattr(manager, _TASK_COMM_MARK, False):
        return
    dispatcher = getattr(manager, "_result_dispatcher", None)
    if dispatcher is None:
        return
    output_cls = _get_io_type("ModelWorkerTaskOutput")
    communicator = manager.model_worker_execute_task_group_communicator
    try:
        from sglang.utils import TypeBasedDispatcher

        dispatcher += TypeBasedDispatcher([(output_cls, communicator.handle_recv)])
    except Exception:
        mapping = getattr(dispatcher, "_mapping", None)
        if isinstance(mapping, list):
            mapping.insert(0, (output_cls, communicator.handle_recv))
        else:
            raise
    setattr(manager, _TASK_COMM_MARK, True)


def _ensure_scheduler_dispatch(scheduler) -> None:
    dispatcher = getattr(scheduler, "_request_dispatcher", None)
    mapping = getattr(dispatcher, "_mapping", None)
    if not isinstance(mapping, list):
        return
    task_cls = _get_io_type("ModelWorkerTask")
    if any(ty is task_cls for ty, _ in mapping):
        return
    mapping.insert(0, (task_cls, scheduler.execute_task_in_model_worker))


def _get_comm_cls():
    try:
        module = importlib.import_module(
            "sglang.srt.managers.tokenizer_communicator_mixin"
        )
        return getattr(module, "_Communicator", _AwexCommunicator)
    except Exception:
        return _AwexCommunicator


def _get_io_type(name: str):
    module = importlib.import_module("sglang.srt.managers.io_struct")
    _patch_io_struct(module)
    return getattr(module, name)


def _get_task_fan_out(self, server_args) -> int:
    if getattr(server_args, "enable_dp_attention", False):
        return int(server_args.tp_size) * int(server_args.pp_size)
    return (
        int(getattr(server_args, "dp_size", 1) or 1)
        * int(getattr(server_args, "tp_size", 1) or 1)
        * int(getattr(server_args, "pp_size", 1) or 1)
    )


def _ordered_task_results(manager, results):
    server_args = manager.server_args

    def _norm_rank(value):
        return 0 if value is None else value

    if getattr(server_args, "enable_dp_attention", False):
        ordered = sorted(
            results,
            key=lambda r: (
                _norm_rank(getattr(r, "pp_rank", None)),
                _norm_rank(getattr(r, "tp_rank", None)),
            ),
        )
    else:
        ordered = sorted(
            results,
            key=lambda r: (
                _norm_rank(getattr(r, "dp_rank", None)),
                _norm_rank(getattr(r, "pp_rank", None)),
                _norm_rank(getattr(r, "tp_rank", None)),
            ),
        )
    if (
        not getattr(server_args, "enable_dp_attention", False)
        and int(getattr(server_args, "dp_size", 1) or 1) > 1
    ):
        ordered = [r for r in ordered if _norm_rank(getattr(r, "dp_rank", None)) == 0]
    return [getattr(r, "result", r) for r in ordered]


def _build_model_context(scheduler) -> dict[str, Any]:
    server_args = getattr(scheduler, "server_args", None)
    tp_worker = getattr(scheduler, "tp_worker", None)
    return {
        "tp_rank": getattr(scheduler, "tp_rank", 0),
        "pp_rank": getattr(scheduler, "pp_rank", 0),
        "tp_size": getattr(scheduler, "tp_size", 1),
        "pp_size": getattr(scheduler, "pp_size", 1),
        "dp_rank": getattr(scheduler, "dp_rank", 0),
        "dp_size": getattr(scheduler, "dp_size", getattr(server_args, "dp_size", 1)),
        "attn_tp_rank": getattr(
            scheduler, "attn_tp_rank", getattr(scheduler, "tp_rank", 0)
        ),
        "attn_tp_size": getattr(
            scheduler, "attn_tp_size", getattr(scheduler, "tp_size", 1)
        ),
        "attn_dp_rank": getattr(scheduler, "attn_dp_rank", 0),
        "world_size": _world_attr(scheduler, "world_size", 1),
        "global_rank": _world_attr(scheduler, "rank", getattr(scheduler, "tp_rank", 0)),
        "local_rank": _world_attr(
            scheduler, "local_rank", getattr(scheduler, "tp_rank", 0)
        ),
        "nnodes": getattr(server_args, "nnodes", 1),
        "server_args": server_args,
        "scheduler": scheduler,
        "tp_worker": tp_worker,
        "real_tp_worker": _real_tp_worker(tp_worker),
        "infer_engine_config": server_args,
        "cp_rank": getattr(scheduler, "cp_rank", 0),
        "cp_size": getattr(
            scheduler, "cp_size", getattr(server_args, "context_parallel_size", 1)
        ),
        "cp_mode": getattr(
            scheduler, "cp_mode", getattr(server_args, "context_parallel_mode", None)
        ),
    }


def _world_attr(scheduler, name: str, default: Any):
    world_group = getattr(scheduler, "world_group", None)
    return getattr(world_group, name, default)


def _real_tp_worker(tp_worker):
    return getattr(tp_worker, "worker", tp_worker)


def _ensure_control_ipc(port_args, server_args=None) -> str:
    control_ipc = getattr(port_args, "control_plane_ipc_name", None)
    if control_ipc:
        return control_ipc
    control_ipc = _make_control_ipc(port_args, server_args)
    port_args.control_plane_ipc_name = control_ipc
    return control_ipc


def _make_control_ipc(port_args, server_args=None) -> str:
    tokenizer_ipc = getattr(port_args, "tokenizer_ipc_name", "")
    if isinstance(tokenizer_ipc, str) and tokenizer_ipc.startswith("tcp://"):
        host, port = _parse_tcp_addr(tokenizer_ipc)
        return f"tcp://{host}:{port + 5}"
    dist_addr = getattr(server_args, "dist_init_addr", None)
    if isinstance(dist_addr, str) and ":" in dist_addr:
        host, port_text = dist_addr.rsplit(":", 1)
        if port_text.isdigit():
            return f"tcp://{host}:{int(port_text) + 6}"
    return f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"


def _parse_tcp_addr(addr: str) -> tuple[str, int]:
    rest = addr[len("tcp://") :]
    host, port_text = rest.rsplit(":", 1)
    return host, int(port_text)


def _start_control_loop(manager) -> None:
    control_manager = getattr(manager, "control_manager", None)
    if control_manager is None:
        return
    existing = getattr(manager, _CONTROL_TASK, None)
    if existing is not None and not existing.done():
        return
    try:
        loop = getattr(manager, "_chosen_loop", None) or asyncio.get_event_loop()
        wrapper = _get_exception_wrapper(manager)
        task = loop.create_task(wrapper(control_manager.handle_loop))
        setattr(manager, _CONTROL_TASK, task)
        task_set = getattr(manager, "asyncio_tasks", None)
        if task_set is not None:
            task_set.add(task)
    except Exception as exc:
        logger.warning("Failed to start SGLang Awex control loop: %s", exc)


def _get_exception_wrapper(manager):
    module = sys.modules.get(type(manager).__module__)
    wrapper = getattr(module, "print_exception_wrapper", None)
    if wrapper is not None:
        return wrapper

    async def _wrapper(func):
        await func()

    return _wrapper


ensure_sglang_patched()
