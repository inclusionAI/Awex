# Copyright (c) Ant Group. Licensed under the Apache License, Version 2.0.
"""
Regression test for the AWEX P73-class bug on **train attn_tp == 1 + infer TP > 1**.

Background (root cause, 2026-06-17 investigation):
  共卡 SWE 实验 (allocation actor attn=d2p4c8, 即 attn_tp_size==1 纯 CP8) 在 step2
  推理崩溃 (reject 98%), 而 zjw / gsm8k 用 attn=d1p4t4c2 (attn_tp_size==4) 正常。
  唯一区分变量收敛到 actor attn 是否开 TP。

  awex `BailingMoeShardingStrategy.get_sharding_strategy` 对 query_key_value 的注释
  (ling.py) 已记录同款失败模式 (P73):
    > Declaring NO_SHARDING ... the transfer plan generated ops for infer
    > tp_rank 0 only and Lightning qkv on tp_rank > 0 stayed all-zero.
  P73 的修复只覆盖 `attn_tp_size > 1` 分支 (返回 TP_SHARDING, 对齐 infer TP)；
  当 train `attn_tp_size == 1` 时仍回落到 NO_SHARDING —— 但推理侧是 TP>1 (sglang
  d16t4)，于是 transfer plan 同样只覆盖 infer tp_rank 0、tp_rank>0 权重保持全零，
  step2 推理生成崩溃。

这个测试把上述「触发条件」固化为回归用例：
  - attn_tp==4 (正常拓扑)  -> query_key_value 走 TP_SHARDING (对齐 infer TP)
  - attn_tp==1 (崩溃拓扑)  -> query_key_value 回落 NO_SHARDING (P73 同款 bug 触发条件)

注意：本测试在 sharding-strategy 层坐实「触发条件」(无 TP -> NO_SHARDING 声明)。
完整的「transfer plan 漏给 infer tp_rank>0 生成 op」需要 meta-resolver + plan 生成
层的端到端测试 (TODO)，并已由 lr=0 运行时实验 (step2 behave_imp_weight 偏离 1) 互证。
"""

import pytest

from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo


def _make_rank_info(attn_tp_size: int, cp_size: int) -> RankInfo:
    """构造 train(mcore)侧 RankInfo。attn_tp_size==1 + cp_size>1 = d2p4c8 崩溃拓扑。"""
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
    """正常拓扑 d1p4t4c2 / d2p4t4c2: attn_tp=4 -> TP_SHARDING, 和 infer TP 对齐, 传对。"""
    strat = _make_strategy(attn_tp_size=4, cp_size=2)
    sharding_type, _dim, num_shards = strat.get_sharding_strategy(QKV_PARAM)
    assert sharding_type == ShardingType.TP_SHARDING
    assert num_shards == 4


def test_qkv_sharding_with_attn_tp1_falls_back_to_no_sharding():
    """崩溃拓扑 d2p4c8: attn_tp=1(纯CP8) -> NO_SHARDING。

    这是 P73 同款 bug 的触发条件: train 声明完整副本(NO_SHARDING), 但推理侧 sglang
    是 TP>1, transfer plan 只给 infer tp_rank 0 生成 op, tp_rank>0 权重全零 -> 崩。
    P73 的修复仅覆盖 attn_tp>1 分支, 未覆盖此处 attn_tp==1 + infer_tp>1 组合。
    """
    strat = _make_strategy(attn_tp_size=1, cp_size=8)
    sharding_type, _dim, num_shards = strat.get_sharding_strategy(QKV_PARAM)
    # 当前 awex 行为(bug 触发条件): 回落 NO_SHARDING。
    assert sharding_type == ShardingType.NO_SHARDING
    assert num_shards == 1


def test_cp_size_does_not_change_qkv_strategy():
    """对照: 固定 attn_tp=1, cp_size 从 1 变到 8, query_key_value 的 sharding 决策不变。

    证明 awex 的 attn sharding 决策完全不感知 CP 维度——CP8 的 8 份冗余副本
    在 strategy 层没有任何特殊处理, 全部依赖下游收集/传输层, 而该层在
    attn_tp==1 + infer_tp>1 时漏传(P73 同款)。
    """
    s_cp1 = _make_strategy(attn_tp_size=1, cp_size=1).get_sharding_strategy(QKV_PARAM)
    s_cp8 = _make_strategy(attn_tp_size=1, cp_size=8).get_sharding_strategy(QKV_PARAM)
    assert s_cp1 == s_cp8  # cp_size 不影响 attn sharding 决策(盲区)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
