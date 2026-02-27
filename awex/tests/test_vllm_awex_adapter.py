import types

from awex.vllm_awex_adapter import AwexVLLMServerAdapter


class _DummyEngineCore:
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [{"source": "engine_core", "method": method}]


class _DummyEngineClient:
    def __init__(self, with_dp_cores: bool):
        self.engine_core = _DummyEngineCore()
        self.vllm_config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                enable_expert_parallel=False,
                prefill_context_parallel_size=1,
                nnodes=1,
                node_rank=0,
            )
        )
        self.model_config = types.SimpleNamespace(hf_config=object())
        self.collective_calls = []
        self.utility_calls = []
        if with_dp_cores:
            self.core_engines = [b"dp0", b"dp1"]

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.collective_calls.append((method, timeout, args, kwargs))
        return [{"source": "collective_rpc", "method": method}]

    # Intentionally sync in this test; adapter supports both sync and async calls.
    def _call_utility_async(self, utility_method, *args, engine=None):
        self.utility_calls.append((utility_method, args, engine))
        rpc_method = args[0]
        if rpc_method == "awex_execute":
            if engine == b"dp0":
                return [{"rank_info": {"global_rank": 0}, "from": "dp0"}]
            if engine == b"dp1":
                return [{"rank_info": {"global_rank": 1}, "from": "dp1"}]
        return [{"source": "utility", "engine": engine}]


def _build_adapter(with_dp_cores: bool) -> AwexVLLMServerAdapter:
    client = _DummyEngineClient(with_dp_cores=with_dp_cores)
    adapter = AwexVLLMServerAdapter(client, meta_server_addr="127.0.0.1:9999")
    adapter._initialized = True
    return adapter


def _build_adapter_async_llm_style(with_dp_cores: bool) -> AwexVLLMServerAdapter:
    core_client = _DummyEngineClient(with_dp_cores=with_dp_cores)
    async_llm_like = types.SimpleNamespace(
        engine_core=core_client,
        vllm_config=core_client.vllm_config,
        model_config=core_client.model_config,
        collective_rpc=core_client.collective_rpc,
    )
    adapter = AwexVLLMServerAdapter(async_llm_like, meta_server_addr="127.0.0.1:9999")
    adapter._initialized = True
    return adapter


def test_meta_collection_aggregates_all_dp_cores():
    adapter = _build_adapter(with_dp_cores=True)

    def _get_model_param_info():
        return None

    results = adapter.execute_task_in_model_worker(
        _get_model_param_info, engine_name="vllm"
    )

    assert len(results) == 2
    assert {r["from"] for r in results} == {"dp0", "dp1"}
    assert len(adapter._engine_client.utility_calls) == 2
    assert len(adapter._engine_client.collective_calls) == 0


def test_meta_collection_aggregates_via_engine_core_client():
    adapter = _build_adapter_async_llm_style(with_dp_cores=True)

    def _get_model_param_info():
        return None

    results = adapter.execute_task_in_model_worker(
        _get_model_param_info, engine_name="vllm"
    )

    assert len(results) == 2
    assert {r["from"] for r in results} == {"dp0", "dp1"}
    assert len(adapter._engine_client.engine_core.utility_calls) == 2
    assert len(adapter._engine_client.engine_core.collective_calls) == 0


def test_non_meta_task_keeps_collective_rpc_path():
    adapter = _build_adapter(with_dp_cores=True)

    results = adapter.execute_task_in_model_worker(
        "_update_parameters_in_tp_worker", step_id=1
    )

    assert results == [
        {"source": "collective_rpc", "method": "_update_parameters_in_tp_worker"}
    ]
    assert len(adapter._engine_client.collective_calls) == 1


def test_meta_collection_without_dp_core_falls_back_to_collective_rpc():
    adapter = _build_adapter(with_dp_cores=False)

    results = adapter.execute_task_in_model_worker("_get_model_param_info")

    assert results == [{"source": "collective_rpc", "method": "_get_model_param_info"}]
    assert len(adapter._engine_client.collective_calls) == 1
