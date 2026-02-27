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

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pytest
import torch

from awex.meta.meta_resolver import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlanBuilder


@dataclass(frozen=True)
class _ShardSpec:
    rank: int
    offset: int
    length: int


def _build_param_meta(
    name: str,
    specs: List[_ShardSpec],
    global_numel: int,
    world_size: int,
) -> ParameterMeta:
    shards = []
    for spec in specs:
        shards.append(
            ParameterShardMeta(
                tp_rank=spec.rank,
                attn_tp_rank=spec.rank,
                pp_rank=0,
                ep_rank=0,
                ep_tp_rank=0,
                global_rank=spec.rank,
                world_size=world_size,
                engine_rank=0,
                name=name,
                shape=(spec.length,),
                numel=spec.length,
                dtype=torch.float32,
                global_offset=(spec.offset,),
                sharding_type=(
                    ShardingType.TP_SHARDING
                    if len(specs) > 1
                    else ShardingType.NO_SHARDING
                ),
                num_shards=max(len(specs), 1),
                sharding_dim=0,
            )
        )
    return ParameterMeta(
        name=name,
        global_numel=global_numel,
        global_shape=(global_numel,),
        dtype=torch.float32,
        shards=shards,
        replicas=[ParameterReplicaMeta(shards=shards)],
    )


def _op_signature(op: CommunicationOperation) -> Tuple:
    return (
        op.send_rank,
        op.recv_rank,
        op.send_shard_meta.global_offset,
        op.recv_shard_meta.global_offset,
        op.overlap_shape,
        op.train_slices,
        op.inf_slices,
    )


def _run_inter_cpu_simulation(
    *,
    train_specs: List[_ShardSpec],
    infer_specs: List[_ShardSpec],
    global_numel: int,
) -> None:
    infer_world_size = max(spec.rank for spec in infer_specs) + 1
    train_world_size = max(spec.rank for spec in train_specs) + 1

    train_meta = _build_param_meta(
        name="model.layers.0.mlp.gate_proj.weight",
        specs=train_specs,
        global_numel=global_numel,
        world_size=train_world_size,
    )
    infer_meta = _build_param_meta(
        name="model.layers.0.mlp.gate_proj.weight",
        specs=infer_specs,
        global_numel=global_numel,
        world_size=infer_world_size,
    )

    builder = TransferPlanBuilder(
        infer_world_size=infer_world_size,
        train_world_size=train_world_size,
        num_infer_engines=1,
    )
    ops = builder.build_weights_mapping_operations([infer_meta], [train_meta])
    ops2 = builder.build_weights_mapping_operations([infer_meta], [train_meta])

    assert [_op_signature(op) for op in ops] == [_op_signature(op) for op in ops2]

    # Use non-zero deterministic values to make "unwritten" detection clear.
    ground_truth = torch.arange(1, global_numel + 1, dtype=torch.float32)

    train_tensors: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}
    for spec in train_specs:
        send_rank = infer_world_size + spec.rank
        key = (send_rank, (spec.offset,))
        train_tensors[key] = ground_truth[
            spec.offset : spec.offset + spec.length
        ].clone()

    infer_tensors: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}
    infer_written_masks: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}
    for spec in infer_specs:
        key = (spec.rank, (spec.offset,))
        infer_tensors[key] = torch.zeros(spec.length, dtype=torch.float32)
        infer_written_masks[key] = torch.zeros(spec.length, dtype=torch.bool)

    for op in ops:
        src_key = (op.send_rank, op.send_shard_meta.global_offset)
        dst_key = (op.recv_rank, op.recv_shard_meta.global_offset)
        src_tensor = train_tensors[src_key]
        dst_tensor = infer_tensors[dst_key]
        dst_written = infer_written_masks[dst_key]

        src_view = src_tensor[op.train_slices]
        dst_view = dst_tensor[op.inf_slices]
        dst_written_view = dst_written[op.inf_slices]

        # Every destination slice must be written exactly once.
        assert not dst_written_view.any()
        dst_view.copy_(src_view)
        dst_written_view.fill_(True)

    for key, written in infer_written_masks.items():
        assert written.all(), f"incomplete write for infer shard {key}"

    assembled = torch.zeros(global_numel, dtype=torch.float32)
    for spec in infer_specs:
        key = (spec.rank, (spec.offset,))
        assembled[spec.offset : spec.offset + spec.length] = infer_tensors[key]

    assert torch.equal(assembled, ground_truth)


@pytest.mark.parametrize(
    "train_specs,infer_specs,global_numel",
    [
        # 1 -> 2 split
        (
            [_ShardSpec(rank=0, offset=0, length=8)],
            [
                _ShardSpec(rank=0, offset=0, length=4),
                _ShardSpec(rank=1, offset=4, length=4),
            ],
            8,
        ),
        # 2 -> 1 merge
        (
            [
                _ShardSpec(rank=0, offset=0, length=4),
                _ShardSpec(rank=1, offset=4, length=4),
            ],
            [_ShardSpec(rank=0, offset=0, length=8)],
            8,
        ),
        # 2 -> 2 uneven many-to-many overlap
        (
            [
                _ShardSpec(rank=0, offset=0, length=5),
                _ShardSpec(rank=1, offset=5, length=3),
            ],
            [
                _ShardSpec(rank=0, offset=0, length=3),
                _ShardSpec(rank=1, offset=3, length=5),
            ],
            8,
        ),
    ],
)
def test_cpu_dummy_inter_plan_simulation(
    train_specs: List[_ShardSpec],
    infer_specs: List[_ShardSpec],
    global_numel: int,
):
    _run_inter_cpu_simulation(
        train_specs=train_specs,
        infer_specs=infer_specs,
        global_numel=global_numel,
    )
