# Copyright (c) Ant Group. Licensed under the Apache License, Version 2.0.
"""
Regression test for attention qkv sharding declarations under
**train attn_tp == 1 (pure CP) + infer TP > 1** topologies.

Background:
  With a colocate allocation where the train side uses pure context
  parallelism for attention (e.g. attn=d2p4c8, attn_tp_size == 1),
  inference collapsed at the first post-update step (~98% rejected
  generations), while an otherwise identical run with attn=d1p4t4c2
  (attn_tp_size == 4) was healthy. The only differing variable was
  whether the train attention used TP.

  The sharding-strategy comment for query_key_value in ling.py records
  the same failure mode:
    > Declaring NO_SHARDING ... the transfer plan generated ops for infer
    > tp_rank 0 only and Lightning qkv on tp_rank > 0 stayed all-zero.
  That fix only covers the `attn_tp_size > 1` branch (returns
  TP_SHARDING, aligned with infer TP). When train `attn_tp_size == 1`
  the strategy still falls back to NO_SHARDING -- but the inference side
  runs TP > 1, so the transfer plan again only covers infer tp_rank 0
  and the qkv weights on tp_rank > 0 stay all-zero.

These tests pin down the trigger condition as a regression case:
  - attn_tp == 4 (healthy topology) -> query_key_value declared
    TP_SHARDING, aligned with infer TP
  - attn_tp == 1 (broken topology)  -> query_key_value falls back to
    NO_SHARDING (the trigger condition above)

Note: this test asserts the trigger condition at the sharding-strategy
layer (no TP -> NO_SHARDING declaration). A full end-to-end test that
the transfer plan misses ops for infer tp_rank > 0 requires the
meta-resolver + plan-generation layers (TODO); the runtime failure mode
was cross-validated with an lr=0 run where the post-update importance
weights diverged from 1.
"""

import pytest

from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo


def _make_rank_info(attn_tp_size: int, cp_size: int) -> RankInfo:
    """Build a train-side (mcore) RankInfo.

    attn_tp_size == 1 with cp_size > 1 reproduces the broken pure-CP
    topology (e.g. d2p4c8).
    """
    return RankInfo(
        tp_rank=0,
        tp_size=attn_tp_size,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=attn_tp_size,
        attn_dp_rank=0,
        world_size=max(attn_tp_size, 1) * max(cp_size, 1),
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=False,
        cp_rank=0,
        cp_size=cp_size,
        cp_mode="ring" if cp_size > 1 else "none",
    )


def _make_strategy(attn_tp_size: int, cp_size: int):
    from awex.models.ling_linear import BailingLinearMoeShardingStrategy

    return BailingLinearMoeShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=1,
        tp_size=attn_tp_size,
        ep_size=1,
        ep_tp_size=1,
        rank_info=_make_rank_info(attn_tp_size, cp_size),
    )


QKV_PARAM = "model.layers.0.attention.query_key_value.weight"


def test_qkv_sharding_with_attn_tp4_is_tp_sharded():
    """Healthy topology (attn_tp=4): TP_SHARDING, aligned with infer TP."""
    strat = _make_strategy(attn_tp_size=4, cp_size=2)
    sharding_type, _dim, num_shards = strat.get_sharding_strategy(QKV_PARAM)
    assert sharding_type == ShardingType.TP_SHARDING
    assert num_shards == 4


def test_qkv_sharding_with_attn_tp1_falls_back_to_no_sharding():
    """Broken topology (attn_tp=1, pure CP8): falls back to NO_SHARDING.

    This is the trigger condition: the train side declares a full
    replica (NO_SHARDING) while the inference side runs TP > 1, so the
    transfer plan only generates ops for infer tp_rank 0 and the qkv
    weights on tp_rank > 0 stay all-zero. The TP_SHARDING fix only
    covers the attn_tp > 1 branch, not this attn_tp == 1 + infer_tp > 1
    combination.
    """
    strat = _make_strategy(attn_tp_size=1, cp_size=8)
    sharding_type, _dim, num_shards = strat.get_sharding_strategy(QKV_PARAM)
    # Current awex behavior (the trigger condition): NO_SHARDING fallback.
    assert sharding_type == ShardingType.NO_SHARDING
    assert num_shards == 1


def test_cp_size_does_not_change_qkv_strategy():
    """Control: with attn_tp=1 fixed, cp_size 1 -> 8 does not change the decision.

    The attention sharding decision is entirely CP-agnostic: the CP
    redundant replicas get no special handling at the strategy layer,
    leaving the downstream collection/transfer layers responsible -- and
    those layers under-transfer when attn_tp == 1 and infer_tp > 1.
    """
    s_cp1 = _make_strategy(attn_tp_size=1, cp_size=1).get_sharding_strategy(QKV_PARAM)
    s_cp8 = _make_strategy(attn_tp_size=1, cp_size=8).get_sharding_strategy(QKV_PARAM)
    assert s_cp1 == s_cp8  # cp_size does not affect the attn sharding decision


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
