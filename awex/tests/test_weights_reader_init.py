import pickle
from types import SimpleNamespace

import torch

from awex.config import InferenceConfig
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.reader.weights_reader import WeightsReader
from awex.sharding.param_sharding import ShardingType


class _DummyHFConfig:
    architectures = ["Qwen2ForCausalLM"]
    num_hidden_layers = 1
    router_dtype = "bf16"

    def to_dict(self):
        return {
            "architectures": self.architectures,
            "num_hidden_layers": self.num_hidden_layers,
            "router_dtype": self.router_dtype,
        }


class _DummyMetaServerClient:
    def __init__(self, *args, **kwargs):
        self.objects = {}

    def put_object(self, key, value):
        self.objects[key] = value

    def get_object(self, key, timeout=None):
        return self.objects[key]


class _DummyMetaResolver:
    def __init__(self, params_meta):
        self.rank0_info = SimpleNamespace(attn_tp_size=1)
        self._params_meta = params_meta

    def get_parameters_meta(self):
        return self._params_meta

    def get_model_arch_name(self):
        return "Qwen2ForCausalLM"


class _DummyInferenceEngine:
    engine_name = "vllm"
    num_engines = 1
    engine_rank = 0

    def __init__(self, config):
        self.config = config
        self.hf_config = _DummyHFConfig()
        self.received_task_kwargs = None

    def execute_task_in_model_worker(self, task, **kwargs):
        self.received_task_kwargs = kwargs


def _build_param_meta():
    shard = ParameterShardMeta(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=0,
        world_size=1,
        engine_rank=0,
        name="model.layers.0.self_attn.q_proj.weight",
        shape=(4, 4),
        numel=16,
        dtype=torch.float16,
        global_offset=(0, 0),
        sharding_type=ShardingType.NO_SHARDING,
        num_shards=1,
        sharding_dim=0,
    )
    replica = ParameterReplicaMeta(shards=[shard])
    return ParameterMeta(
        name=shard.name,
        global_numel=16,
        global_shape=(4, 4),
        dtype=torch.float16,
        shards=[shard],
        replicas=[replica],
    )


def test_weights_reader_infer_conf_carries_engine_name(monkeypatch):
    params_meta = [_build_param_meta()]
    meta_server = _DummyMetaServerClient()
    meta_server.objects["training_params_meta"] = params_meta

    monkeypatch.setattr(
        "awex.reader.weights_reader.MetaServerClient",
        lambda *args, **kwargs: meta_server,
    )
    monkeypatch.setattr(
        "awex.reader.weights_reader.check_train_infer_params_meta",
        lambda *args, **kwargs: None,
    )

    infer_config = InferenceConfig(
        meta_server_addr="127.0.0.1:12345",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        num_engines=1,
        engine_rank=0,
        comm_backend="nccl",
        enable_debug_mode=True,
    )
    engine = _DummyInferenceEngine(infer_config)
    reader = WeightsReader(engine, meta_resolver=_DummyMetaResolver(params_meta))

    reader._initialize()

    assert meta_server.objects["infer_conf"]["engine_name"] == "vllm"
    assert engine.received_task_kwargs is not None
    init_infer_conf = pickle.loads(engine.received_task_kwargs["infer_conf_bytes"])
    assert init_infer_conf["engine_name"] == "vllm"
