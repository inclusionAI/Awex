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

from typing import Tuple

import pytest
import torch

from awex.meta.meta_resolver import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    OverlapRegion,
    ShardOffset,
    TransferPlan,
    TransferPlanBuilder,
    slice_tensor,
)


class TestTransferPlanBuilder:
    """Critical test cases for TransferPlanBuilder class."""

    def test_init_valid_parameters(self):
        """Test TransferPlanBuilder initialization with valid parameters."""
        builder = TransferPlanBuilder(
            infer_world_size=4, train_world_size=2, num_infer_engines=2
        )

        assert builder.train_world_size == 2
        assert builder.infer_world_size == 4
        assert builder.world_size == 6
        assert builder.infer_instance_world_size == 2
        assert builder.num_infer_engines == 2

    def test_init_invalid_num_infer_engines(self):
        """Test TransferPlanBuilder initialization with invalid num_infer_engines."""
        with pytest.raises(ValueError, match="num_infer_engines must be positive"):
            TransferPlanBuilder(
                infer_world_size=4, train_world_size=2, num_infer_engines=0
            )

    def test_build_weights_mapping_operations_empty_meta(self):
        """Test build_weights_mapping_operations with empty metadata."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        result = builder.build_weights_mapping_operations([], [])
        assert result == []

    def test_build_weights_mapping_operations_single_param(self):
        """Test build_weights_mapping_operations with single parameter."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        # Create test metadata with same parameter name
        inference_meta = self._create_test_parameter_meta("param1")
        training_meta = self._create_test_parameter_meta("param1")

        result = builder.build_weights_mapping_operations(
            [inference_meta], [training_meta]
        )
        assert len(result) > 0
        assert all(isinstance(op, CommunicationOperation) for op in result)

    def test_build_parameter_communication_plan_no_replicas(self):
        """Test _build_parameter_communication_plan with no replicas."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        # Create metadata with empty replicas
        inference_meta = self._create_test_parameter_meta("param1")
        inference_meta.replicas = []
        training_meta = self._create_test_parameter_meta("param1")

        with pytest.raises(ValueError, match="No replicas found for parameter param1"):
            builder._build_parameter_communication_plan(
                "param1", inference_meta, training_meta
            )

    def test_create_shard_offset_for_replica_no_shards(self):
        """Test _create_shard_offset_for_replica with no shards."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)
        replica = ParameterReplicaMeta(shards=[])

        with pytest.raises(IndexError):
            builder._create_shard_offset_for_replica(
                replica, engine_rank=0, is_infer=True
            )

    def test_create_shard_offset_for_replica_empty_global_offset(self):
        """Test _create_shard_offset_for_replica with empty global_offset."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)
        shard = self._create_test_shard_meta(global_offset=())
        replica = ParameterReplicaMeta(shards=[shard])

        with pytest.raises(ValueError, match="Shard has empty global_offset"):
            builder._create_shard_offset_for_replica(
                replica, engine_rank=0, is_infer=True
            )

    def test_create_shard_offset_for_replica_dimension_mismatch(self):
        """Test _create_shard_offset_for_replica with dimension mismatch."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)
        shard = self._create_test_shard_meta(
            global_offset=(0, 0),
            shape=(2, 3, 4),  # 3D shape with 2D offset
        )
        replica = ParameterReplicaMeta(shards=[shard])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            builder._create_shard_offset_for_replica(
                replica, engine_rank=0, is_infer=True
            )

    def test_create_shard_offset_for_replica_valid_inference(self):
        """Test _create_shard_offset_for_replica with valid inference shard."""
        builder = TransferPlanBuilder(
            infer_world_size=4, train_world_size=2, num_infer_engines=2
        )
        shard = self._create_test_shard_meta(
            global_offset=(0, 0), shape=(2, 3), global_rank=1
        )
        replica = ParameterReplicaMeta(shards=[shard])

        result = builder._create_shard_offset_for_replica(
            replica, engine_rank=1, is_infer=True
        )

        assert len(result) == 1
        shard_info = result[0]
        assert shard_info.shard == shard
        assert shard_info.start_offset == (0, 0)
        assert shard_info.end_offset == (2, 3)
        assert shard_info.shape == (2, 3)
        # Rank calculation: 1 + (1 * 2) = 3 (global_rank + engine_rank * infer_instance_world_size)
        assert shard_info.rank == 3

    def test_find_overlapping_regions_no_overlap(self):
        """Test _find_overlapping_regions with no overlapping shards."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        # Create non-overlapping shards
        inference_map = [
            ShardOffset(
                shard=self._create_test_shard_meta(),
                start_offset=(0, 0),
                end_offset=(2, 2),
                shape=(2, 2),
                rank=0,
            )
        ]

        training_map = [
            ShardOffset(
                shard=self._create_test_shard_meta(),
                start_offset=(3, 3),
                end_offset=(5, 5),
                shape=(2, 2),
                rank=1,
            )
        ]

        result = builder._find_overlapping_regions(inference_map, training_map)
        assert result == []

    def test_find_overlapping_regions_partial_overlap(self):
        """Test _find_overlapping_regions with partial overlap."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        # Create partially overlapping shards
        inference_map = [
            ShardOffset(
                shard=self._create_test_shard_meta(),
                start_offset=(0, 0),
                end_offset=(3, 3),
                shape=(3, 3),
                rank=0,
            )
        ]

        training_map = [
            ShardOffset(
                shard=self._create_test_shard_meta(),
                start_offset=(1, 1),
                end_offset=(4, 4),
                shape=(3, 3),
                rank=1,
            )
        ]

        result = builder._find_overlapping_regions(inference_map, training_map)
        assert len(result) == 1
        region = result[0]
        assert region.overlap_start == (1, 1)
        assert region.overlap_end == (3, 3)

    def test_create_slice_function(self):
        """Test _create_slice_function creates correct slicing function."""
        source_offset = (1, 2)
        overlap_shape = (2, 3)
        inference_meta = self._create_test_parameter_meta("param1")
        training_meta = self._create_test_parameter_meta("param1")
        slices = tuple(
            slice(source_offset[i], source_offset[i] + overlap_shape[i])
            for i in range(len(source_offset))
        )
        op = CommunicationOperation(
            send_rank=0,
            send_shard_meta=training_meta,
            send_offset=source_offset,
            recv_rank=1,
            recv_shard_meta=inference_meta,
            recv_offset=source_offset,
            overlap_shape=overlap_shape,
            train_slices=slices,
            inf_slices=slices,
        )
        # Create a test tensor
        tensor = torch.randn(5, 5)
        result = slice_tensor(tensor, op, is_train=True)

        # Check that the result has the expected shape
        assert result.shape == overlap_shape
        # Check that the result contains the correct slice
        expected_slice = tensor[1:3, 2:5]
        assert torch.allclose(result, expected_slice)
        # Check that slices are correct
        assert slices == (slice(1, 3), slice(2, 5))

    def test_build_region_communication_plan(self):
        """Test _build_region_communication_plan creates correct operations."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        region = OverlapRegion(
            inference_shard=ShardOffset(
                shard=self._create_test_shard_meta(global_offset=(0, 0), shape=(3, 3)),
                start_offset=(0, 0),
                end_offset=(3, 3),
                shape=(3, 3),
                rank=0,
            ),
            training_shard=ShardOffset(
                shard=self._create_test_shard_meta(global_offset=(1, 1), shape=(3, 3)),
                start_offset=(1, 1),
                end_offset=(4, 4),
                shape=(3, 3),
                rank=2,  # Training rank should be >= infer_world_size (2)
            ),
            overlap_start=(1, 1),
            overlap_end=(3, 3),
        )

        result = builder._build_region_communication_plan(region)

        assert len(result) == 1
        op = result[0]
        assert isinstance(op, CommunicationOperation)
        assert op.send_rank == 2  # Training rank
        assert op.recv_rank == 0  # Inference rank
        assert op.send_offset == (0, 0)  # Relative to training shard
        assert op.recv_offset == (1, 1)  # Relative to inference shard
        assert op.overlap_shape == (2, 2)

    def test_group_operations_by_rank(self):
        """Test _group_operations_by_rank groups operations correctly."""
        builder = TransferPlanBuilder(infer_world_size=2, train_world_size=1)

        # Create test operations with required parameters
        op1 = CommunicationOperation(
            send_rank=0,
            send_shard_meta=self._create_test_shard_meta(),
            send_offset=(0, 0),
            recv_rank=1,
            recv_shard_meta=self._create_test_shard_meta(),
            recv_offset=(0, 0),
            overlap_shape=(2, 2),
            train_slices=(slice(0, 2), slice(0, 2)),
            inf_slices=(slice(0, 2), slice(0, 2)),
        )

        op2 = CommunicationOperation(
            send_rank=0,
            send_shard_meta=self._create_test_shard_meta(),
            send_offset=(1, 1),
            recv_rank=2,
            recv_shard_meta=self._create_test_shard_meta(),
            recv_offset=(1, 1),
            overlap_shape=(2, 2),
            train_slices=(slice(1, 3), slice(1, 3)),
            inf_slices=(slice(1, 3), slice(1, 3)),
        )

        operations = [op1, op2]
        result = builder._group_operations_by_rank(operations, "send_rank")

        assert len(result) == 1
        assert len(result[0]) == 2  # Two operations with send_rank=0

    def test_build_local_transfer_plan_training_rank(self):
        """Test build_local_transfer_plan for training rank."""
        builder = TransferPlanBuilder(
            infer_world_size=4, train_world_size=2, num_infer_engines=2
        )

        inference_meta = self._create_test_parameter_meta("param1")
        training_meta = self._create_test_parameter_meta("param1")

        # Test for training rank 0
        result = builder.build_local_transfer_plan(
            [inference_meta], [training_meta], global_transfer_rank=0
        )

        assert isinstance(result, TransferPlan)
        assert isinstance(result.operations, dict)

    def _create_test_shard_meta(
        self,
        global_offset: Tuple[int, ...] = (0, 0),
        shape: Tuple[int, ...] = (2, 2),
        global_rank: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> ParameterShardMeta:
        """Helper method to create test ParameterShardMeta."""
        # Calculate numel safely
        if len(shape) == 0:
            numel = 1
        elif len(shape) == 1:
            numel = shape[0]
        else:
            numel = shape[0] * shape[1]

        return ParameterShardMeta(
            tp_rank=0,
            attn_tp_rank=0,
            pp_rank=0,
            ep_rank=0,
            ep_tp_rank=0,
            global_rank=global_rank,
            world_size=2,
            engine_rank=0,
            name="test_param",
            shape=shape,
            numel=numel,
            dtype=dtype,
            global_offset=global_offset,
            sharding_type=ShardingType.NO_SHARDING,
            num_shards=1,
            sharding_dim=0,
        )

    def _create_test_parameter_meta(
        self, name: str, global_shape: Tuple[int, ...] = (4, 4)
    ) -> ParameterMeta:
        """Helper method to create test ParameterMeta."""
        shard = self._create_test_shard_meta(
            global_offset=(0, 0), shape=global_shape, global_rank=0
        )

        replica = ParameterReplicaMeta(shards=[shard])

        return ParameterMeta(
            name=name,
            global_numel=global_shape[0] * global_shape[1],
            global_shape=global_shape,
            dtype=torch.float32,
            shards=[shard],
            replicas=[replica],
        )


class TestCommunicationOperation:
    """Test cases for CommunicationOperation dataclass."""

    def test_communication_operation_creation(self):
        """Test CommunicationOperation creation with valid parameters."""
        send_shard = ParameterShardMeta(
            tp_rank=0,
            attn_tp_rank=0,
            pp_rank=0,
            ep_rank=0,
            ep_tp_rank=0,
            global_rank=0,
            world_size=2,
            engine_rank=0,
            name="test_param",
            shape=(2, 2),
            numel=4,
            dtype=torch.float32,
            global_offset=(0, 0),
            sharding_type=ShardingType.NO_SHARDING,
            num_shards=1,
            sharding_dim=0,
        )

        recv_shard = ParameterShardMeta(
            tp_rank=0,
            attn_tp_rank=0,
            pp_rank=0,
            ep_rank=0,
            ep_tp_rank=0,
            global_rank=1,
            world_size=2,
            engine_rank=0,
            name="test_param",
            shape=(2, 2),
            numel=4,
            dtype=torch.float32,
            global_offset=(0, 0),
            sharding_type=ShardingType.NO_SHARDING,
            num_shards=1,
            sharding_dim=0,
        )

        op = CommunicationOperation(
            send_rank=0,
            send_shard_meta=send_shard,
            send_offset=(0, 0),
            recv_rank=1,
            recv_shard_meta=recv_shard,
            recv_offset=(0, 0),
            overlap_shape=(2, 2),
            train_slices=(slice(0, 2), slice(0, 2)),
            inf_slices=(slice(0, 2), slice(0, 2)),
        )

        assert op.send_rank == 0
        assert op.recv_rank == 1
        assert op.send_offset == (0, 0)
        assert op.recv_offset == (0, 0)
        assert op.send_shard_meta == send_shard
        assert op.recv_shard_meta == recv_shard
        assert op.overlap_shape == (2, 2)


class TestTransferPlan:
    """Test cases for TransferPlan dataclass."""

    def test_transfer_plan_creation(self):
        """Test TransferPlan creation with valid operations."""
        operations = {
            0: [
                CommunicationOperation(
                    send_rank=0,
                    send_shard_meta=ParameterShardMeta(
                        tp_rank=0,
                        attn_tp_rank=0,
                        pp_rank=0,
                        ep_rank=0,
                        ep_tp_rank=0,
                        global_rank=0,
                        world_size=2,
                        engine_rank=0,
                        name="test_param",
                        shape=(2, 2),
                        numel=4,
                        dtype=torch.float32,
                        global_offset=(0, 0),
                        sharding_type=ShardingType.NO_SHARDING,
                        num_shards=1,
                        sharding_dim=0,
                    ),
                    send_offset=(0, 0),
                    recv_rank=1,
                    recv_shard_meta=ParameterShardMeta(
                        tp_rank=0,
                        attn_tp_rank=0,
                        pp_rank=0,
                        ep_rank=0,
                        ep_tp_rank=0,
                        global_rank=1,
                        world_size=2,
                        engine_rank=0,
                        name="test_param",
                        shape=(2, 2),
                        numel=4,
                        dtype=torch.float32,
                        global_offset=(0, 0),
                        sharding_type=ShardingType.NO_SHARDING,
                        num_shards=1,
                        sharding_dim=0,
                    ),
                    recv_offset=(0, 0),
                    overlap_shape=(2, 2),
                    train_slices=(slice(0, 2), slice(0, 2)),
                    inf_slices=(slice(0, 2), slice(0, 2)),
                )
            ]
        }

        plan = TransferPlan(operations=operations)
        assert plan.operations == operations
        assert len(plan.operations[0]) == 1

    def test_transfer_plan_empty_operations(self):
        """Test TransferPlan creation with empty operations."""
        operations = {}
        plan = TransferPlan(operations=operations)
        assert plan.operations == operations
        assert len(plan.operations) == 0
