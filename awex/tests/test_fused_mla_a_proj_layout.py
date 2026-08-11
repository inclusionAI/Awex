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


"""Layout contract for the fused MLA a_proj under inference tensor parallelism.

SGLang consumes the fused MLA parameter per TP rank as ``[q_i ; kv_i]``. AWEX
chunks the writer-side tensor on dim 0, so a plain ``cat([q_all, kv_all])``
gives rank ``i`` a slice that straddles q and kv instead of the pair it needs.
The rows are the right shape either way, which is why getting this wrong
corrupts attention silently rather than raising.
"""

import pytest
import torch

from awex.converter.mcore_converter import pack_fused_qkv_a_proj_for_tp

Q_ROWS = 8
KV_ROWS = 4
IN_DIM = 3


def _q_kv():
    q = torch.arange(Q_ROWS * IN_DIM, dtype=torch.float32).reshape(Q_ROWS, IN_DIM)
    kv = -torch.arange(KV_ROWS * IN_DIM, dtype=torch.float32).reshape(KV_ROWS, IN_DIM)
    return q, kv


@pytest.mark.parametrize("infer_tp_size", [1, 2, 4])
def test_each_rank_dim0_slice_is_its_own_q_kv_pair(infer_tp_size):
    q, kv = _q_kv()

    packed = pack_fused_qkv_a_proj_for_tp(q, kv, infer_tp_size)

    assert packed.shape == (Q_ROWS + KV_ROWS, IN_DIM)
    q_shards = q.chunk(infer_tp_size, dim=0)
    kv_shards = kv.chunk(infer_tp_size, dim=0)
    for rank, shard in enumerate(packed.chunk(infer_tp_size, dim=0)):
        expected = torch.cat([q_shards[rank], kv_shards[rank]], dim=0)
        assert torch.equal(shard, expected), f"rank {rank} slice is not [q_i; kv_i]"


def test_single_rank_degenerates_to_the_naive_concat():
    q, kv = _q_kv()

    packed = pack_fused_qkv_a_proj_for_tp(q, kv, 1)

    assert torch.equal(packed, torch.cat([q, kv], dim=0))


def test_multi_rank_layout_differs_from_the_naive_concat():
    """The whole point: the naive layout is what corrupts attention."""
    q, kv = _q_kv()

    packed = pack_fused_qkv_a_proj_for_tp(q, kv, 4)

    assert not torch.equal(packed, torch.cat([q, kv], dim=0))


def test_packing_is_order_preserving_and_invertible():
    q, kv = _q_kv()
    tp = 4

    packed = pack_fused_qkv_a_proj_for_tp(q, kv, tp)

    rebuilt_q, rebuilt_kv = [], []
    for shard in packed.chunk(tp, dim=0):
        rebuilt_q.append(shard[: Q_ROWS // tp])
        rebuilt_kv.append(shard[Q_ROWS // tp :])
    assert torch.equal(torch.cat(rebuilt_q), q)
    assert torch.equal(torch.cat(rebuilt_kv), kv)


@pytest.mark.parametrize(
    "bad",
    [
        {"infer_tp_size": 0},
        {"q_rows": 7},
        {"kv_rows": 3},
        {"in_dim_mismatch": True},
        {"one_dim": True},
    ],
)
def test_bad_shapes_raise_instead_of_corrupting(bad):
    q, kv = _q_kv()
    tp = 4
    if bad.get("q_rows"):
        q = torch.zeros(bad["q_rows"], IN_DIM)
    if bad.get("kv_rows"):
        kv = torch.zeros(bad["kv_rows"], IN_DIM)
    if bad.get("in_dim_mismatch"):
        kv = torch.zeros(KV_ROWS, IN_DIM + 1)
    if bad.get("one_dim"):
        q = torch.zeros(Q_ROWS)

    with pytest.raises(ValueError):
        pack_fused_qkv_a_proj_for_tp(q, kv, bad.get("infer_tp_size", tp))
