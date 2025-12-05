import json
from unittest.mock import MagicMock

import pytest
import torch

from awex.meta.infer_meta_resolver import InferParamMetaResolver
from awex.meta.meta_resolver import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
    ShardingType,
)
from awex.sharding.param_sharding import RankInfo
from awex.tests.test_utils import get_local_model_dir
from awex.util.common import to_dict

MODEL_PATH = "DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ARCH_NAME = "Qwen2ForCausalLM"

# ----------------------
# HELPER FUNCTIONS
# ----------------------


def make_param_meta(name, shape, dtype=torch.float32):
    numel = 1
    for d in shape:
        numel *= d
    return {"name": name, "shape": shape, "numel": numel, "dtype": dtype}


def make_rank_info(
    tp_rank=0,
    tp_size=1,
    dp_size=1,
    attn_tp_rank=0,
    attn_tp_size=1,
    attn_dp_rank=0,
    world_size=1,
    global_rank=0,
    local_rank=0,
    engine_rank=0,
    is_infer=False,
):
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        dp_size=dp_size,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_dp_rank=attn_dp_rank,
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        engine_rank=engine_rank,
        is_infer=is_infer,
    )


def make_all_params_meta(param_defs, ranks):
    all_params_meta = []
    for rank in ranks:
        rank_info = make_rank_info(*rank)
        params_meta = [make_param_meta(name, shape) for name, shape in param_defs]
        all_params_meta.append(
            {
                "rank_info": rank_info,
                "params_meta": params_meta,
                "model_arch_name": MODEL_ARCH_NAME,
            }
        )
    return all_params_meta


def create_dummy_engine(sharding_case=None, param_defs=None, ranks=None):
    """Create a DummySGLangEngine with proper server_args configuration."""

    class DummySGLangEngine:
        def __init__(self):
            # Minimal interface expected by InferParamMetaResolver and
            # sglang sharding helpers.
            self.engine_name = "sglang"

            # server_args is what real SGlang scheduler exposes; we only
            # populate the fields used by sharding detection.
            server_args = MagicMock()
            if sharding_case == "tp_sharding":
                server_args.tp_size = 2
                server_args.dp_size = 1
                server_args.enable_dp_attention = False
            elif sharding_case == "dp_tp_sharding":
                server_args.tp_size = 2
                server_args.dp_size = 2
                server_args.enable_dp_attention = True
            else:  # no_sharding or default
                server_args.tp_size = 1
                server_args.dp_size = 1
                server_args.enable_dp_attention = False
            # Other fields referenced by sharding code with safe defaults
            server_args.enable_dp_lm_head = False
            server_args.moe_dense_tp_size = 1
            server_args.ep_size = 1

            self.engine = MagicMock()
            self.engine.server_args = server_args

            # In real SGLangEngine, InferenceParamMetaResolver reads
            # sharding-related fields from `config` (an InferenceConfig).
            # For the dummy engine we mirror the necessary fields from
            # server_args onto `config` so the default ShardingStrategy
            # sees the expected values.
            self.config = MagicMock()
            self.config.enable_debug_mode = False
            self.config.enable_dp_attention = server_args.enable_dp_attention
            self.config.enable_dp_lm_head = server_args.enable_dp_lm_head
            self.config.moe_dense_tp_size = server_args.moe_dense_tp_size
            self.config.ep_size = server_args.ep_size

            # Add hf_config attribute that InferParamMetaResolver expects
            self.hf_config = MagicMock()
            self.hf_config.num_hidden_layers = 4

        def execute_task_in_model_worker(self, fn, **kwargs):
            if param_defs and ranks:
                return make_all_params_meta(param_defs, ranks)
            return []

    return DummySGLangEngine()


def create_test_shard(
    name="test.weight",
    shape=(4, 4),
    tp_rank=0,
    global_rank=0,
    world_size=1,
    sharding_type=ShardingType.NO_SHARDING,
    num_shards=1,
    global_offset=(0, 0),
    dtype=torch.float32,
):
    """Create a test ParameterShardMeta with common defaults."""
    numel = 1
    for d in shape:
        numel *= d
    return ParameterShardMeta(
        tp_rank=tp_rank,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=global_rank,
        world_size=world_size,
        engine_rank=global_rank,
        name=name,
        shape=shape,
        numel=numel,
        dtype=dtype,
        global_offset=global_offset,
        sharding_type=sharding_type,
        num_shards=num_shards,
        sharding_dim=0,
    )


def create_test_replica(shards):
    """Create a test ParameterReplicaMeta."""
    return ParameterReplicaMeta(shards=shards)


def create_test_param(
    name="test.weight",
    shape=(4, 4),
    shards=None,
    replicas=None,
    dtype=torch.float32,
):
    """Create a test ParameterMeta with common defaults."""
    if shards is None:
        shards = [create_test_shard(name, shape, dtype=dtype)]
    if replicas is None:
        replicas = [create_test_replica(shards)]

    global_numel = sum(s.numel for s in shards)
    return ParameterMeta(
        name=name,
        global_numel=global_numel,
        global_shape=shape,
        dtype=dtype,
        shards=shards,
        replicas=replicas,
    )


def create_real_engine_config(model_path, tp_size=1, pp_size=1, dp_size=1):
    """Create a real SGlangBackend configuration with meta server."""
    from awex.meta.meta_server import start_meta_server

    # Start meta server and get address
    meta_server_host, meta_server_port = start_meta_server()
    meta_server_addr = f"{meta_server_host}:{meta_server_port}"

    # Ensure model is available locally (via HF or ModelScope) so
    # sglang can load it without requiring network access.
    local_model_path = get_local_model_dir(model_path)

    # Create config as a dictionary that SGlangBackend can handle
    config_dict = {
        "model_path": local_model_path,
        "served_model_name": model_path,
        "attention_backend": "torch_native",
        "tp_size": tp_size,
        "pp_size": pp_size,
        "trust_remote_code": True,
        "dp_size": dp_size,
        "log_level": "error",
        "stream_output": False,
        "enable_debug_mode": False,
        "engine_rank": 0,
        "num_engines": 1,
        "node_rank": 0,
        "comm_backend": "nccl",
        "meta_server_addr": meta_server_addr,
        "disable_shared_experts_fusion": True,
        "mem_fraction_static": 0.5,
    }

    return config_dict


# ----------------------
# 1. MOCKED ENGINE TESTS
# ----------------------
@pytest.mark.parametrize(
    "sharding_case,param_defs,ranks,expected",
    [
        (
            "no_sharding",
            [
                ("embed_tokens.weight", (8, 8)),
                ("mlp.weight", (8, 8)),
                ("attention.weight", (8, 8)),
            ],
            [(0, 1, 1, 0, 1, 0, 1, 0, 0, 0, False)],
            {
                "sharding_types": {
                    "embed_tokens.weight": ShardingType.NO_SHARDING,
                    "mlp.weight": ShardingType.NO_SHARDING,
                    "attention.weight": ShardingType.NO_SHARDING,
                },
                "num_shards": 1,
                "num_replicas": 1,
            },
        ),
        (
            "tp_sharding",
            [("mlp.weight", (4, 8)), ("attention.weight", (4, 8))],
            [
                (0, 2, 1, 0, 1, 0, 2, 0, 0, 0, False),
                (1, 2, 1, 0, 1, 0, 2, 1, 1, 1, False),
            ],
            {
                "sharding_types": {
                    "mlp.weight": ShardingType.TP_SHARDING,
                    "attention.weight": ShardingType.TP_SHARDING,
                },
                "num_shards": 2,
                "num_replicas": 1,
            },
        ),
        (
            "dp_tp_sharding",
            [("attention.weight", (2, 8))],
            [
                (0, 2, 2, 0, 2, 0, 4, 0, 0, 0, False),
                (1, 2, 2, 1, 2, 0, 4, 1, 1, 1, False),
                (0, 2, 2, 0, 2, 1, 4, 2, 2, 2, False),
                (1, 2, 2, 1, 2, 1, 4, 3, 3, 3, False),
            ],
            {
                "sharding_types": {"attention.weight": ShardingType.DP_TP_SHARDING},
                "num_shards": 4,
                "num_replicas": 2,
            },
        ),
    ],
)
def test_meta_resolver_sharding_mocked(sharding_case, param_defs, ranks, expected):
    backend = create_dummy_engine(sharding_case, param_defs, ranks)
    resolver = InferParamMetaResolver(backend)
    params_meta = resolver.get_parameters_meta()

    for param in params_meta:
        assert param.name in expected["sharding_types"], (
            f"Unexpected param {param.name} in {sharding_case}"
        )

        for shard in param.shards:
            assert shard.sharding_type == expected["sharding_types"][param.name], (
                f"{param.name} wrong sharding type in {sharding_case}"
            )

        assert len(param.shards) == expected["num_shards"], (
            f"{param.name} wrong number of shards in {sharding_case}"
        )
        assert len(param.replicas) == expected["num_replicas"], (
            f"{param.name} wrong number of replicas in {sharding_case}"
        )

        for replica in param.replicas:
            assert isinstance(replica, ParameterReplicaMeta)
            assert all(isinstance(s, ParameterShardMeta) for s in replica.shards)
            total_numel = sum(s.numel for s in param.replicas[0].shards)
            assert total_numel == param.global_numel

        offsets = [tuple(s.global_offset) for s in param.shards]
        if sharding_case == "dp_tp_sharding":
            # Should have unique offsets per TP rank, but can be duplicated across DP ranks
            unique_offsets = set(offsets)
            assert len(unique_offsets) == 2, (
                f"Expected 2 unique offsets for DP_TP sharding, got {len(unique_offsets)}"
            )
        else:
            # For other sharding types, each shard should have a unique offset
            assert len(set(offsets)) == len(offsets)

        assert param.dtype == torch.float32


def test_inconsistent_replicas_mocked():
    param_defs = [("mlp.weight", (4, 8))]
    ranks = [
        (0, 2, 1, 0, 1, 0, 2, 0, 0, 0, False),
        (0, 2, 1, 0, 1, 0, 2, 1, 1, 1, False),
        (1, 2, 1, 0, 1, 0, 2, 2, 2, 2, False),
    ]

    backend = create_dummy_engine("tp_sharding", param_defs, ranks)
    with pytest.raises(AssertionError):
        InferParamMetaResolver(backend)


def test_no_sharding_offset_computation():
    """Test that NO_SHARDING parameters have correct global offsets (0, 0) for each replica."""
    from awex.meta.meta_resolver import ParamMetaResolver

    class TestParamMetaResolver(ParamMetaResolver):
        def __init__(self):
            # Create a mock hf_config for the parent class
            mock_hf_config = MagicMock()
            mock_hf_config.num_hidden_layers = 4
            super().__init__(mock_hf_config)

        def get_model_arch_name(self) -> str:
            return "TestModel"

        def get_parameters_meta(self):
            return self._build_params_meta()

        def _get_params_raw_meta(self):
            # Simulate multiple ranks with NO_SHARDING parameter
            return [
                {
                    "rank_info": make_rank_info(
                        world_size=4, global_rank=i, local_rank=i, engine_rank=i
                    ),
                    "params_meta": [
                        {
                            "name": "model.word_embeddings.weight",
                            "numel": 258998272,
                            "shape": [126464, 2048],
                            "dtype": torch.bfloat16,
                        }
                    ],
                }
                for i in range(4)
            ]

        def _get_sharding_info(self, name, rank_info, param_meta):
            # For word_embeddings, it should be NO_SHARDING
            return ShardingType.NO_SHARDING, 0, 1

    resolver = TestParamMetaResolver()
    params_meta = resolver.get_parameters_meta()

    # Find the word_embeddings parameter
    word_embeddings_param = next(
        (p for p in params_meta if p.name == "model.word_embeddings.weight"), None
    )
    assert word_embeddings_param is not None, (
        "Could not find model.word_embeddings.weight parameter"
    )

    assert word_embeddings_param.global_shape == (126464, 2048)
    assert len(word_embeddings_param.shards) == 4
    assert len(word_embeddings_param.replicas) == 4

    # Check that each shard has the correct offset (0, 0) for NO_SHARDING
    for i, shard in enumerate(word_embeddings_param.shards):
        assert shard.global_offset == (
            0,
            0,
        ), f"Shard {i} has wrong offset {shard.global_offset}, expected (0, 0)"
        assert shard.global_offset[0] <= 100000, (
            f"Shard {i} has suspiciously large offset {shard.global_offset}"
        )


@pytest.mark.parametrize(
    "test_case",
    [
        ("tp_sharding", 2, ShardingType.TP_SHARDING, [(0, 0), (4, 0)]),
        ("no_sharding", 2, ShardingType.NO_SHARDING, [(0, 0), (0, 0)]),
    ],
)
def test_parameter_meta_to_local(test_case):
    """Test the to_local_parameter_meta method with different sharding types."""
    sharding_type, world_size, expected_sharding_type, expected_offsets = test_case

    # Create shards for multiple ranks
    shards = [
        create_test_shard(
            name="test.weight",
            shape=(4, 4),
            tp_rank=i,
            global_rank=i,
            world_size=world_size,
            sharding_type=expected_sharding_type,
            num_shards=world_size,
            global_offset=expected_offsets[i],
        )
        for i in range(world_size)
    ]

    # Each replica should contain only the shards for that specific rank
    replicas = [create_test_replica([shards[i]]) for i in range(world_size)]

    param_meta = create_test_param(
        name="test.weight",
        shape=(8, 4) if sharding_type == "tp_sharding" else (4, 4),
        shards=shards,
        replicas=replicas,
    )

    # Test for each rank
    for rank in range(world_size):
        local_meta = param_meta.to_local_parameter_meta(global_rank=rank)
        assert len(local_meta.shards) == 1
        assert local_meta.shards[0].global_rank == rank
        assert len(local_meta.replicas) == 1
        assert len(local_meta.replicas[0].shards) == 1
        assert local_meta.replicas[0].shards[0].global_rank == rank


def test_parameter_meta_to_local_invalid_rank():
    """Test to_local_parameter_meta with invalid rank."""
    param_meta = create_test_param()

    # Test with invalid rank should return empty result
    local_meta = param_meta.to_local_parameter_meta(global_rank=999)
    assert len(local_meta.shards) == 0
    assert len(local_meta.replicas) == 0


def test_dump_parameters_meta():
    """Test the dump_parameters_meta function."""
    from awex.meta.weight_meta import dump_parameters_meta

    param_meta = create_test_param()
    dumped_data = dump_parameters_meta([param_meta])

    assert len(dumped_data) == 1
    assert dumped_data[0]["name"] == "test.weight"
    assert dumped_data[0]["global_numel"] == 16
    assert dumped_data[0]["global_shape"] == [4, 4]  # to_dict converts tuples to lists
    assert dumped_data[0]["num_shards"] == 1
    assert dumped_data[0]["num_replicas"] == 1


def test_compute_total_model_size():
    """Test the compute_total_model_size function."""
    from awex.meta.weight_meta import compute_total_model_size

    # Create multiple parameters with different sizes
    param1 = create_test_param("param1.weight", (4, 4))
    param2 = create_test_param("param2.weight", (2, 3))

    total_size = compute_total_model_size([param1, param2])
    # 16 elements * 4 bytes per float32 + 6 elements * 4 bytes per float32 = 88 bytes
    assert total_size == 88


@pytest.mark.parametrize(
    "test_case",
    [
        ("shard_attributes", create_test_shard()),
        ("replica_attributes", create_test_replica([create_test_shard()])),
        ("param_attributes", create_test_param()),
    ],
)
def test_meta_attributes(test_case):
    """Test attributes of ParameterShardMeta, ParameterReplicaMeta, and ParameterMeta."""
    test_name, obj = test_case

    if test_name == "shard_attributes":
        assert obj.tp_rank == 0
        assert obj.attn_tp_rank == 0
        assert obj.pp_rank == 0
        assert obj.global_rank == 0
        assert obj.world_size == 1
        assert obj.engine_rank == 0
        assert obj.name == "test.weight"
        assert obj.shape == (4, 4)
        assert obj.numel == 16
        assert obj.dtype == torch.float32
        assert obj.global_offset == (0, 0)
        assert obj.sharding_type == ShardingType.NO_SHARDING
        assert obj.num_shards == 1
        assert obj.sharding_dim == 0

    elif test_name == "replica_attributes":
        assert len(obj.shards) == 1
        assert isinstance(obj.shards[0], ParameterShardMeta)

    elif test_name == "param_attributes":
        assert obj.name == "test.weight"
        assert obj.global_numel == 16
        assert obj.global_shape == (4, 4)
        assert obj.dtype == torch.float32
        assert len(obj.shards) == 1
        assert len(obj.replicas) == 1
        assert isinstance(obj.shards[0], ParameterShardMeta)
        assert isinstance(obj.replicas[0], ParameterReplicaMeta)


def test_dp_tp_sharding_offsets():
    """Test that DP_TP sharding produces correct offsets with duplicates across DP ranks."""
    param_defs = [("attention.weight", (2, 8))]
    ranks = [
        (0, 2, 2, 0, 2, 0, 4, 0, 0, 0, False),
        (1, 2, 2, 1, 2, 0, 4, 1, 1, 1, False),
        (0, 2, 2, 0, 2, 1, 4, 2, 2, 2, False),
        (1, 2, 2, 1, 2, 1, 4, 3, 3, 3, False),
    ]

    backend = create_dummy_engine("dp_tp_sharding", param_defs, ranks)
    resolver = InferParamMetaResolver(backend)
    params_meta = resolver.get_parameters_meta()

    assert len(params_meta) == 1
    param = params_meta[0]
    assert param.name == "attention.weight"
    assert len(param.shards) == 4
    assert len(param.replicas) == 2

    # Check sharding type
    for shard in param.shards:
        assert shard.sharding_type == ShardingType.DP_TP_SHARDING

    # Check offsets - should have 2 unique offsets (one per TP rank)
    offsets = [tuple(s.global_offset) for s in param.shards]
    unique_offsets = set(offsets)
    assert len(unique_offsets) == 2, (
        f"Expected 2 unique offsets for DP_TP sharding, got {len(unique_offsets)}: {unique_offsets}"
    )

    # Verify the offsets are (0, 0) and (2, 0) as expected for TP sharding
    expected_offsets = {(0, 0), (2, 0)}
    assert unique_offsets == expected_offsets, (
        f"Expected offsets {expected_offsets}, got {unique_offsets}"
    )

    # Verify replica structure
    assert len(param.replicas[0].shards) == 2
    assert len(param.replicas[1].shards) == 2

    # Check that replicas have the same TP rank distribution
    replica0_tp_ranks = {s.tp_rank for s in param.replicas[0].shards}
    replica1_tp_ranks = {s.tp_rank for s in param.replicas[1].shards}
    assert replica0_tp_ranks == {0, 1}
    assert replica1_tp_ranks == {0, 1}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real engine test.",
)
def test_meta_resolver_with_real_engine():
    # Check if model exists locally or can be downloaded
    model_path = "Qwen/Qwen2-1.5B"
    config = create_real_engine_config(model_path)
    import sglang as sgl

    from awex.engine.sglang import SGLangEngine, extract_sgl_config

    try:
        sgl_engine = sgl.Engine(**extract_sgl_config(config), random_seed=42)
    except Exception as e:
        msg = str(e)
        # Environment-specific constraints such as unbalanced GPU
        # memory can cause sglang to abort; treat these as
        # non-fatal for the unit test environment.
        if "memory capacity is unbalanced" in msg:
            pytest.skip(f"SGLang Engine unavailable due to GPU memory layout: {e}")
        raise

    engine = SGLangEngine(config, sgl_engine)
    engine.initialize()
    resolver = InferParamMetaResolver(engine)
    params_meta = resolver.get_parameters_meta()
    assert isinstance(params_meta, list)
    assert all(isinstance(p, ParameterMeta) for p in params_meta)
    param_names = [p.name for p in params_meta]
    assert any("embed_tokens" in n for n in param_names)
    assert any("mlp" in n for n in param_names)
    assert any("attention" in n or "attn" in n for n in param_names)

    for p in params_meta:
        assert isinstance(p.dtype, torch.dtype)
        assert isinstance(p.shards, list)
        assert isinstance(p.replicas, list)
        for shard in p.shards:
            assert isinstance(shard, ParameterShardMeta)
            assert isinstance(shard.dtype, torch.dtype)
            assert isinstance(shard.sharding_type, ShardingType)
            assert isinstance(shard.num_shards, int)
        for replica in p.replicas:
            assert isinstance(replica, ParameterReplicaMeta)
            assert all(isinstance(s, ParameterShardMeta) for s in replica.shards)
    sgl_engine.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA are required for lite meta resolver test.",
)
def test_meta_resolver_lite():
    model_path = "Qwen/Qwen2-1.5B"
    # Meta server startup can fail on constrained environments; treat
    # that as a skip rather than a hard failure so tests remain
    # robust when networking or binding is restricted.
    try:
        config = create_real_engine_config(model_path, tp_size=4)
    except RuntimeError as e:
        pytest.skip(f"Meta server unavailable in test environment: {e}")

    import sglang as sgl

    from awex.engine.sglang import SGLangEngine, extract_sgl_config

    try:
        sgl_engine = sgl.Engine(**extract_sgl_config(config), random_seed=42)
    except Exception as e:
        msg = str(e)
        if "memory capacity is unbalanced" in msg:
            pytest.skip(f"SGLang Engine unavailable due to GPU memory layout: {e}")
        raise
    engine = SGLangEngine(config, sgl_engine)
    engine.initialize()
    resolver = InferParamMetaResolver(engine)
    params_meta = resolver.get_parameters_meta()
    print(params_meta)
    data = []
    for p in params_meta:
        data.append(
            to_dict(
                {
                    "name": p.name,
                    "global_numel": p.global_numel,
                    "global_shape": p.global_shape,
                    "dtype": p.dtype,
                    "num_shards": len(p.shards),
                    "num_replicas": len(p.replicas),
                    "replicas": p.replicas,
                }
            )
        )
    print(json.dumps(data, indent=2))
    sgl_engine.shutdown()


if __name__ == "__main__":
    test_meta_resolver_lite()
