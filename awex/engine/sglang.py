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
import functools
import pickle
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from awex import logging
from awex.config import InferenceConfig
from awex.engine.core import InferenceEngine
from awex.reader.weights_reader import get_weights_exchange_reader
from awex.util.gpu import get_gpu_status

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    def __init__(self, config: dict[str, Any] | InferenceConfig, sgl_engine):
        super().__init__(sgl_engine.tokenizer_manager.model_config)
        if isinstance(config, dict):
            config = InferenceConfig.from_dict(config)
        config.validate()
        self._config = config
        self._sgl_engine = sgl_engine
        self.node_rank = config.node_rank or 0
        self.released_tags = set()
        self.weights_exchange_reader = None
        self.rank_coordinate = f"{config.engine_rank}-{self.node_rank}"
        self._initialized = False

    @property
    def engine_name(self):
        return "sglang"

    @property
    def config(self):
        return self._config

    def initialize(self) -> None:
        if self.node_rank == 0:
            logger.info(
                f"Start to initialize weights exchange reader for {self.rank_coordinate}"
            )
            self._initialized = True
            self.weights_exchange_reader = get_weights_exchange_reader(self)
            self.weights_exchange_reader.initialize()
            logger.info(
                f"Finished initializing weights exchange reader for {self.rank_coordinate}"
            )
        else:
            logger.info(
                f"Skip initializing weights exchange reader for {self.rank_coordinate}"
            )

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        """Update model weights for inference."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        logger.info(
            f"Start to update weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}, load_format: {load_format}"
        )
        if self.node_rank != 0:
            logger.info("Non-zero rank node, skipping update weights from disk")
            return
        self._sgl_engine.update_weights_from_disk(
            model_path=model_path, load_format=load_format
        )
        logger.info(
            f"Finished updating weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}, load_format: {load_format}"
        )

    def update_weights(self, **kwargs):
        logger.info(
            f"Start to update weights for step {self.global_step} for {self.rank_coordinate}"
        )
        start_time = time.time()
        self.weights_exchange_reader.update_weights(step_id=self.global_step, **kwargs)
        duration = time.time() - start_time
        logger.info(
            f"Finished updating weights for step {self.global_step} for {self.rank_coordinate}, "
            f"took {duration:.3f} seconds"
        )

    def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        if self._initialized and self.node_rank == 0:
            logger.info(
                f"Release memory occupation {tags}, released_tags {self.released_tags}"
            )
            if set(tags) - self.released_tags != set(tags):
                tags = list(set(tags) - self.released_tags)
            self.released_tags.update(tags)
            if not tags:
                logger.info("No memory occupation to release")
                return
            logger.info(f"Start to release memory occupation {tags}")
            logger.info(f"GPU status before release:\n{get_gpu_status()}")
            self._sgl_engine.release_memory_occupation(tags=tags)
            logger.info("Finished releasing memory occupation")
            logger.info(f"GPU status after release:\n{get_gpu_status()}")

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        """Resume memory occupation for the engine.
        tags: kv_cache, weights, default is both
        """
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        if self._initialized and self.node_rank == 0:
            logger.info(
                f"Resume memory occupation {tags}, released_tags {self.released_tags}"
            )
            tags = list(self.released_tags & set(tags))
            self.released_tags.difference_update(tags)
            if not tags:
                logger.info("No memory occupation to resume")
                return
            logger.info(f"Start to resume memory occupation {tags}")
            logger.info(f"GPU status before resume:\n{get_gpu_status()}")
            self._sgl_engine.resume_memory_occupation(tags=tags)
            logger.info("Finished resuming memory occupation")
            logger.info(f"GPU status after resume:\n{get_gpu_status()}")

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call `initialize` first.")
        if self.node_rank != 0:
            raise RuntimeError(
                f"Non-zero rank node {self.rank_coordinate} is not allowed to "
                f"execute task in model workers"
            )
        return self._sgl_engine.execute_task_in_model_worker(fn, **kwargs)

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank


def extract_sgl_config(config: dict[str, Any]) -> dict[str, Any]:
    from sglang.srt.server_args import ServerArgs

    engine_kwargs = {
        k: v for k, v in config.items() if k in ServerArgs.__dataclass_fields__
    }
    return engine_kwargs


@dataclass
class _WorkerTask:
    task_id: str
    payload: bytes


@dataclass
class _WorkerResult:
    task_id: str
    rank: tuple
    payload: bytes | None = None
    error: str | None = None


def _send(socket, obj):
    from sglang.srt.managers import io_struct

    # 0.5.0 uses pickle sockets; recent SGLang requires an explicit envelope
    # for Python callbacks. Keep its ordinary inference messages on msgpack.
    if hasattr(io_struct, "wrap_as_pickle"):
        return io_struct.sock_send(socket, io_struct.wrap_as_pickle(obj))
    return socket.send_pyobj(obj)


def _parallel(scheduler):
    # Release 0.5.20 stores ranks in ps; main moved them to runtime_context.
    # 0.5.0 keeps them directly on Scheduler.
    if hasattr(scheduler, "ps"):
        return scheduler.ps
    if hasattr(scheduler, "tp_rank"):
        return scheduler
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def _model_context(scheduler):
    ranks = _parallel(scheduler)
    world = scheduler.world_group
    context = {
        name: getattr(ranks, name)
        for name in (
            "tp_rank",
            "tp_size",
            "pp_rank",
            "pp_size",
            "dp_rank",
            "dp_size",
            "moe_ep_rank",
            "moe_ep_size",
            "attn_tp_rank",
            "attn_tp_size",
            "attn_dp_rank",
        )
    }
    context.update(
        attn_cp_rank=getattr(ranks, "attn_cp_rank", 0),
        attn_cp_size=getattr(ranks, "attn_cp_size", 1),
        world_size=world.world_size,
        global_rank=world.rank,
        local_rank=world.local_rank,
        nnodes=scheduler.server_args.nnodes,
        server_args=scheduler.server_args,
        scheduler=scheduler,
        tp_worker=scheduler.tp_worker,
        real_tp_worker=getattr(scheduler.tp_worker, "worker", scheduler.tp_worker),
    )
    return context


def _execute_worker_task(scheduler, task):
    import cloudpickle
    import torch.distributed as dist

    ranks = _parallel(scheduler)
    result = _WorkerResult(
        task.task_id, (ranks.dp_rank or 0, ranks.pp_rank, ranks.tp_rank)
    )
    try:
        # Each rank gets a fresh payload, including the broadcast source rank.
        # Never attach live CUDA models or the scheduler to the forwarded task.
        fn, kwargs = cloudpickle.loads(task.payload)
        worker = scheduler.tp_worker
        # 0.5.0's overlap scheduler delegates forwards to a thread-backed
        # TpModelWorkerClient. Its target model lives under .worker.
        if hasattr(worker, "worker"):
            worker.forward_stream.synchronize()
            worker = worker.worker
        runner = worker.model_runner
        kwargs.update(
            model=runner.model,
            model_runner=runner,
            model_context=_model_context(scheduler),
        )
        result.payload = pickle.dumps(fn(**kwargs), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        result.error = traceback.format_exc()
    # Non-DP-attention uses node-local tokenizer IPC even for multi-node TP.
    # Gather through the existing world CPU group so remote ranks never try
    # to connect to node 0's Unix socket. Each ordinary DP replica has a world.
    world = scheduler.world_group
    results = [None] * world.world_size if world.rank == 0 else None
    dist.gather_object(result, results, dst=world.ranks[0], group=world.cpu_group)
    if results is not None:
        for result in results:
            _send(scheduler._awex_result_socket, result)


def _install_scheduler_hooks():
    import zmq
    from sglang.srt.managers import scheduler as module

    cls = module.Scheduler
    if getattr(cls, "_awex_worker_hooks_installed", False):
        return
    original_init = cls.__init__
    original_process = cls.process_input_requests
    # Old SGLang has the PP loop inline; newer releases put it in a mixin.
    if hasattr(cls, "_pp_send_pyobj_to_next_stage"):
        from sglang.srt.managers import scheduler_pp_mixin as pp_module
    else:
        pp_module = module
    original_p2p = pp_module.point_to_point_pyobj
    forwarded = None

    def point_to_point(data, *args, **kwargs):
        nonlocal forwarded
        if data is forwarded:
            forwarded = None
            return []
        return original_p2p(data, *args, **kwargs)

    @functools.wraps(original_init)
    def initialize(self, server_args, port_args, *args, **kwargs):
        original_init(self, server_args, port_args, *args, **kwargs)
        if self.world_group.rank == 0:
            self._awex_zmq_context = zmq.Context()
            self._awex_result_socket = self._awex_zmq_context.socket(zmq.PUSH)
            self._awex_result_socket.connect(port_args.tokenizer_ipc_name)

    @functools.wraps(original_process)
    def process(self, requests):
        nonlocal forwarded
        if not any(isinstance(req, _WorkerTask) for req in requests):
            return original_process(self, requests)
        ranks = _parallel(self)
        if (
            ranks.pp_rank < ranks.pp_size - 1
            and ranks.attn_tp_rank == 0
            and getattr(ranks, "attn_cp_rank", 0) == 0
        ):
            # A callback can enter collectives across PP stages. Forward first,
            # and suppress exactly the normal loop's send of this same list.
            # Sending a second (even empty) list would shift the PP pipeline.
            if hasattr(self, "_pp_send_pyobj_to_next_stage"):
                self._pp_commit_comm_work(self.send_req_work)
                self.send_req_work = []
                self._pp_send_pyobj_to_next_stage(requests, async_send=False)
            else:
                offset = ranks.attn_dp_rank * ranks.attn_tp_size
                src = ranks.pp_rank * ranks.tp_size + offset
                original_p2p(
                    requests,
                    src,
                    self.world_group.device_group,
                    src,
                    src + ranks.tp_size,
                )
            forwarded = requests
        for req in requests:
            if isinstance(req, _WorkerTask):
                _execute_worker_task(self, req)
            else:
                original_process(self, [req])

    cls.__init__ = initialize
    cls.process_input_requests = process
    pp_module.point_to_point_pyobj = point_to_point
    cls._awex_worker_hooks_installed = True


def run_scheduler_process(*args, **kwargs):
    """Spawn-safe entry point: install AWEX inside each model subprocess."""
    from sglang.srt.managers.scheduler import run_scheduler_process as run

    _install_scheduler_hooks()
    return run(*args, **kwargs)


def _run_dp_controller(*args, **kwargs):
    # In 0.5.0 the controller imports its own scheduler target after spawn.
    from sglang.srt.managers import data_parallel_controller as module

    module.run_scheduler_process = run_scheduler_process
    if hasattr(module, "sock_send"):
        original_send = module.sock_send

        def send(socket, obj, *args, **kwargs):
            if isinstance(obj, _WorkerTask):
                from sglang.srt.managers.io_struct import wrap_as_pickle

                obj = wrap_as_pickle(obj)
            return original_send(socket, obj, *args, **kwargs)

        module.sock_send = send
    return module.run_data_parallel_controller_process(*args, **kwargs)


def _install_tokenizer(manager):
    manager._awex_task_lock = asyncio.Lock()
    manager._awex_pending_tasks = {}
    original_dispatch = manager._result_dispatcher

    def dispatch(obj):
        if not isinstance(obj, _WorkerResult):
            return original_dispatch(obj)
        pending = manager._awex_pending_tasks.get(obj.task_id)
        if pending is None:
            return
        future, expected, results = pending
        results[obj.rank] = obj
        if len(results) == expected and not future.done():
            future.set_result([results[rank] for rank in sorted(results)])

    manager._result_dispatcher = dispatch


async def _execute_task_async(engine, fn: Callable, **kwargs) -> list[Any]:
    import cloudpickle

    manager = engine.tokenizer_manager
    if manager is None or not hasattr(manager, "_awex_pending_tasks"):
        raise RuntimeError(
            "Call install_sglang_worker_hooks() before constructing sglang.Engine on node 0."
        )
    async with manager._awex_task_lock:
        # Callers must stop submitting generation while updating model weights.
        # PP early forwarding is safe only after pending inference has drained.
        if manager.rid_to_state:
            raise RuntimeError("Model-worker tasks require idle generation requests.")
        manager.auto_create_handle_loop()
        task = _WorkerTask(uuid.uuid4().hex, cloudpickle.dumps((fn, kwargs)))
        args = engine.server_args
        expected = args.tp_size * args.pp_size
        if not args.enable_dp_attention:
            expected *= args.dp_size
        future = asyncio.get_running_loop().create_future()
        manager._awex_pending_tasks[task.task_id] = (future, expected, {})
        try:
            from sglang.srt.managers import io_struct

            if hasattr(io_struct, "wrap_as_pickle"):
                await io_struct.async_sock_send(
                    manager.send_to_scheduler, io_struct.wrap_as_pickle(task)
                )
            else:
                await manager.send_to_scheduler.send_pyobj(task)
            try:
                results = await asyncio.shield(future)
            except asyncio.CancelledError:
                # Keep subsequent collective callbacks serialized even if the
                # caller is cancelled after dispatch; ranks are still running.
                await future
                raise
        finally:
            manager._awex_pending_tasks.pop(task.task_id, None)
        errors = [f"rank {r.rank}: {r.error}" for r in results if r.error]
        if errors:
            raise RuntimeError("SGLang model-worker task failed:\n" + "\n".join(errors))
        # Ordinary DP replicas contain the same weights. Preserve AWEX's
        # existing metadata contract while still executing on every replica.
        if not args.enable_dp_attention and args.dp_size > 1:
            results = [r for r in results if r.rank[0] == 0]
        if args.enable_dp_attention:
            results.sort(key=lambda r: r.rank[1:])
        return [pickle.loads(r.payload) for r in results]


def _execute_task(engine, fn: Callable, **kwargs) -> list[Any]:
    loop = getattr(engine, "loop", None)
    if loop is None:  # 0.5.0 obtains the loop at each API call.
        loop = asyncio.get_event_loop()
    if loop.is_running():
        raise RuntimeError("Use async_execute_task_in_model_worker in an async loop.")
    return loop.run_until_complete(_execute_task_async(engine, fn, **kwargs))


def install_sglang_worker_hooks() -> None:
    """Enable AWEX on subsequent sglang.Engine instances in this process.

    Call before Engine construction on every node. Tasks run on each target
    model's scheduler thread and receive model, model_runner and model_context.
    Generation must be idle. These hooks target the Python Engine API with one
    tokenizer, using SGLang's native multiprocessing launcher.
    """
    from sglang.srt.entrypoints import engine as module

    cls = module.Engine
    if getattr(cls, "_awex_worker_hooks_installed", False):
        return
    original_init = cls.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        server_args = kwargs.get("server_args")
        tokenizer_count = (
            getattr(server_args, "tokenizer_worker_num", 1)
            if server_args is not None
            else kwargs.get("tokenizer_worker_num", 1)
        )
        if tokenizer_count != 1:
            raise ValueError(
                "AWEX's SGLang worker hooks require tokenizer_worker_num=1."
            )
        original_init(self, *args, **kwargs)
        if self.tokenizer_manager is not None:
            _install_tokenizer(self.tokenizer_manager)

    if hasattr(cls, "run_scheduler_process_func"):
        cls.run_scheduler_process_func = staticmethod(run_scheduler_process)
    else:
        module.run_scheduler_process = run_scheduler_process
    module.run_data_parallel_controller_process = _run_dp_controller
    cls.__init__ = initialize
    cls.execute_task_in_model_worker = _execute_task
    cls.async_execute_task_in_model_worker = _execute_task_async
    cls._awex_worker_hooks_installed = True
