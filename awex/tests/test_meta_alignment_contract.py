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

import pytest
import torch

from awex.meta.meta_resolver import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.util.common import check_train_infer_params_meta


def _make_param_meta(name: str, shape=(4, 4), global_rank: int = 0) -> ParameterMeta:
    numel = int(shape[0] * shape[1])
    shard = ParameterShardMeta(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=global_rank,
        world_size=1,
        engine_rank=0,
        name=name,
        shape=shape,
        numel=numel,
        dtype=torch.float32,
        global_offset=(0, 0),
        sharding_type=ShardingType.NO_SHARDING,
        num_shards=1,
        sharding_dim=0,
    )
    return ParameterMeta(
        name=name,
        global_numel=numel,
        global_shape=shape,
        dtype=torch.float32,
        shards=[shard],
        replicas=[ParameterReplicaMeta(shards=[shard])],
    )


def test_meta_alignment_allows_infer_extra_in_non_strict_mode():
    train = [_make_param_meta("param1"), _make_param_meta("model.embed_tokens.weight")]
    infer = [
        _make_param_meta("param1"),
        _make_param_meta("model.embed_tokens.weight"),
        _make_param_meta("lm_head.weight"),
    ]
    check_train_infer_params_meta(train, infer, raise_exception=True)


def test_meta_alignment_rejects_unknown_infer_extra_in_non_strict_mode():
    train = [_make_param_meta("param1")]
    infer = [_make_param_meta("param1"), _make_param_meta("param_extra")]
    with pytest.raises(ValueError, match="unsupported_extra_on_infer"):
        check_train_infer_params_meta(train, infer, raise_exception=True)


def test_meta_alignment_fails_when_train_key_missing_on_infer():
    train = [_make_param_meta("param1"), _make_param_meta("param_missing")]
    infer = [_make_param_meta("param1")]
    with pytest.raises(ValueError, match="missing_on_infer"):
        check_train_infer_params_meta(train, infer, raise_exception=True)


def test_meta_alignment_strict_mode_rejects_infer_extra():
    train = [_make_param_meta("param1")]
    infer = [_make_param_meta("param1"), _make_param_meta("param_extra")]
    with pytest.raises(ValueError, match="extra_on_infer"):
        check_train_infer_params_meta(
            train,
            infer,
            raise_exception=True,
            strict_key_match=True,
        )
