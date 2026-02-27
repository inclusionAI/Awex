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

import json
import os
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Tuple

import torch

from awex import logging
from awex.meta.meta_resolver import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.util.common import _is_allowed_infer_only_alias, to_dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommunicationOperation:
    """Represents a single communication operation in the weights exchange plan."""

    send_rank: int
    send_shard_meta: ParameterShardMeta
    send_offset: Tuple[int, ...]
    recv_rank: int
    recv_shard_meta: ParameterShardMeta
    recv_offset: Tuple[int, ...]
    overlap_shape: Tuple[int, ...]
    train_slices: Tuple[slice, ...]
    inf_slices: Tuple[slice, ...]
    # Destination PP stage index in infer topology.
    pp_rank: int = 0
    # Optional runtime placement version marker for debug/trace.
    placement_version: int = -1
    # Optional parameter class tag (attention/expert/dense_other).
    param_class: str = "dense_other"


@dataclass(slots=True)
class TransferPlan:
    """Represents a transfer plan for a specific rank.

    The operations dictionary maps from the opposite rank (the rank we communicate with)
    to a list of communication operations:
    - For training ranks: maps inference rank -> operations where this training rank sends to that inference rank
    - For inference ranks: maps training rank -> operations where this inference rank receives from that training rank
    """

    # Backward-compatible alias to inter_operations. Existing callers use this.
    operations: Dict[int, List[CommunicationOperation]] = field(default_factory=dict)
    # Explicit inter transfer ops (train <-> infer).
    inter_operations: Dict[int, List[CommunicationOperation]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        # Compatibility bridge: accept old `operations` or new `inter_operations`.
        if self.operations and not self.inter_operations:
            self.inter_operations = self.operations
        elif self.inter_operations and not self.operations:
            self.operations = self.inter_operations
        elif not self.operations and not self.inter_operations:
            self.operations = {}
            self.inter_operations = {}
        elif self.operations != self.inter_operations:
            logger.warning(
                "TransferPlan initialized with both operations and inter_operations; "
                "using inter_operations as source of truth."
            )
            self.operations = self.inter_operations


@dataclass(slots=True)
class ShardOffset:
    shard: Any
    start_offset: tuple
    end_offset: tuple
    shape: tuple
    rank: int


@dataclass(slots=True)
class OverlapRegion:
    inference_shard: ShardOffset
    training_shard: ShardOffset
    overlap_start: tuple
    overlap_end: tuple


@dataclass(slots=True)
class NormalizedAxes:
    """Unified rank-axis view used by planner logic."""

    pp_rank: int
    tp_rank: int
    tp_size: int
    ep_rank: int
    ep_size: int
    ep_tp_rank: int
    ep_tp_size: int
    attn_tp_rank: int
    attn_tp_size: int


def normalize_rank_axes(
    param_class: str, rank_info: Any, shard_meta: Any = None
) -> NormalizedAxes:
    """Map rank axes to a single normalized view.

    `param_class` can be one of: `attention`, `expert`, `dense_other`.
    For attention we prioritize `attn_tp_*`; for expert we preserve both
    `ep_*` and `ep_tp_*`; for dense we use generic `tp_*`.
    """

    def get(obj: Any, key: str, default: Any) -> Any:
        return getattr(obj, key, default) if obj is not None else default

    pp_rank = get(shard_meta, "pp_rank", get(rank_info, "pp_rank", 0))
    if param_class == "attention":
        tp_rank = get(rank_info, "attn_tp_rank", get(rank_info, "tp_rank", 0))
        tp_size = get(rank_info, "attn_tp_size", get(rank_info, "tp_size", 1))
    else:
        tp_rank = get(rank_info, "tp_rank", 0)
        tp_size = get(rank_info, "tp_size", 1)
    return NormalizedAxes(
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=get(rank_info, "ep_rank", 0),
        ep_size=get(rank_info, "ep_size", 1),
        ep_tp_rank=get(rank_info, "ep_tp_rank", 0),
        ep_tp_size=get(rank_info, "ep_tp_size", 1),
        attn_tp_rank=get(rank_info, "attn_tp_rank", tp_rank),
        attn_tp_size=get(rank_info, "attn_tp_size", tp_size),
    )


class TransferPlanBuilder:
    def __init__(
        self,
        infer_world_size: int,
        train_world_size: int,
        num_infer_engines: int = 1,
        enable_debug_mode: bool = False,
        strict_param_key_match: bool = False,
    ):
        if num_infer_engines <= 0:
            raise ValueError("num_infer_engines must be positive")

        self.train_world_size = train_world_size
        self.infer_world_size = infer_world_size
        self.world_size = train_world_size + infer_world_size
        self.infer_instance_world_size = infer_world_size // num_infer_engines
        self.num_infer_engines = num_infer_engines
        self.enable_debug_mode = enable_debug_mode
        self.strict_param_key_match = strict_param_key_match
        logger.info(
            f"TransferPlanBuilder: infer_world_size: {infer_world_size}, train_world_size: {train_world_size}, world_size: {self.world_size}, "
            f"infer_instance_world_size: {self.infer_instance_world_size}, num_infer_engines: {num_infer_engines}, "
            f"enable_debug_mode: {enable_debug_mode}, strict_param_key_match: {strict_param_key_match}"
        )

    def build_weights_mapping_operations(
        self,
        inference_weights_meta: List[ParameterMeta],
        training_weights_meta: List[ParameterMeta],
        global_transfer_rank: int = None,
    ) -> List[CommunicationOperation]:
        """
        Build the weights transfer plan.

        Args:
            inference_weights_meta: List of inference parameter metadata
            training_weights_meta: List of training parameter metadata
            global_transfer_rank: If provided, only build operations for this specific rank
            is_train: If provided, indicates whether this is a training rank
        Returns:
            TransferPlan containing communication operations
        """
        # Validate input parameters
        if not inference_weights_meta or not training_weights_meta:
            logger.warning("Empty weights meta provided, returning empty plan")
            return []

        # Create a mapping from parameter name to ParameterMeta for both inference and training
        inference_meta_dict = {meta.name: meta for meta in inference_weights_meta}
        training_meta_dict = {meta.name: meta for meta in training_weights_meta}

        infer_keys = set(inference_meta_dict.keys())
        train_keys = set(training_meta_dict.keys())
        common_params = infer_keys & train_keys
        missing_on_infer = train_keys - infer_keys
        extra_on_infer = infer_keys - train_keys
        unsupported_extra_on_infer = {
            key
            for key in extra_on_infer
            if not _is_allowed_infer_only_alias(key, infer_keys, train_keys)
        }
        if self.strict_param_key_match:
            key_mismatch = bool(missing_on_infer or extra_on_infer)
        else:
            # Two-stage contract: inter transfer requires train keys to exist
            # in infer canonical-ingress metadata.
            # Infer-only extras are allowed only for known alias coverage.
            key_mismatch = bool(missing_on_infer or unsupported_extra_on_infer)
        if key_mismatch:
            logger.error(
                "Train/infer key mismatch for transfer plan: train=%s infer=%s "
                "missing_on_infer=%s extra_on_infer=%s unsupported_extra_on_infer=%s strict=%s",
                len(train_keys),
                len(infer_keys),
                sorted(missing_on_infer),
                sorted(extra_on_infer),
                sorted(unsupported_extra_on_infer),
                self.strict_param_key_match,
            )
            raise ValueError(
                "Train/infer key mismatch for transfer plan: "
                f"train={len(train_keys)} infer={len(infer_keys)} "
                f"missing_on_infer={sorted(missing_on_infer)} "
                f"extra_on_infer={sorted(extra_on_infer)} "
                f"unsupported_extra_on_infer={sorted(unsupported_extra_on_infer)} "
                f"strict={self.strict_param_key_match}"
            )
        if extra_on_infer:
            logger.info(
                "Infer metadata has extra keys (ignored for inter plan in non-strict mode), "
                "count=%s sample=%s",
                len(extra_on_infer),
                sorted(extra_on_infer)[:8],
            )

        communication_plan = []

        for param_name in sorted(common_params):
            inference_meta = inference_meta_dict[param_name]
            training_meta = training_meta_dict[param_name]

            # Build communication plan for this parameter
            param_plan = self._build_parameter_communication_plan(
                param_name,
                inference_meta,
                training_meta,
                global_transfer_rank=global_transfer_rank,
            )
            communication_plan.extend(param_plan)

        logger.info(
            f"Built communication plan with {len(communication_plan)} operations "
            f"for {len(common_params)} common parameters"
        )
        return communication_plan

    def _build_parameter_communication_plan(
        self,
        param_name: str,
        inference_meta: ParameterMeta,
        training_meta: ParameterMeta,
        global_transfer_rank: int = None,
    ) -> List[CommunicationOperation]:
        """
        Build communication plan for a single parameter for a specific rank.

        This method handles multiple replicas by:
        1. Getting all replicas from both inference and training ParameterMeta
        2. Assigning training replicas to inference replicas evenly using round-robin
        3. For each replica pair, finding overlapping regions and building communication operations
        4. Each inference replica receives tensors from one assigned training replica

        Args:
            param_name: Name of the parameter
            inference_meta: Inference parameter metadata with replicas
            training_meta: Training parameter metadata with replicas

        Returns:
            List of communication operations for this parameter
        """
        plan = []

        # Get the replicas for both inference and training
        inference_replicas = []
        for engine_rank in range(self.num_infer_engines):
            inference_replicas.extend(
                (engine_rank, replica) for replica in inference_meta.replicas
            )
        training_replicas = training_meta.replicas

        if not inference_replicas or not training_replicas:
            raise ValueError(f"No replicas found for parameter {param_name}")

        # Assign training replicas to inference replicas using proper distribution
        num_inference_replicas = len(inference_replicas)
        num_training_replicas = len(training_replicas)

        # Ensure each inference replica is assigned to exactly one training replica
        # and each training replica gets an equal number of inference replicas
        replica_assignments = []

        # Use round-robin assignment to distribute inference replicas evenly
        for inf_replica_idx in range(num_inference_replicas):
            # Assign each inference replica to a training replica using modulo
            train_replica_idx = inf_replica_idx % num_training_replicas
            replica_assignments.append((inf_replica_idx, train_replica_idx))

        logger.debug(
            f"Parameter {param_name}: Assigned {num_inference_replicas} inference replicas "
            f"to {num_training_replicas} training replicas: {replica_assignments}"
        )

        # Build communication plan for each replica pair
        for inf_replica_idx, train_replica_idx in replica_assignments:
            engine_rank, inference_replica = inference_replicas[inf_replica_idx]
            training_replica = training_replicas[train_replica_idx]
            is_infer_rank = (
                global_transfer_rank is not None
                and global_transfer_rank < self.infer_world_size
            )
            is_train_rank = (
                global_transfer_rank is not None
                and global_transfer_rank >= self.infer_world_size
            )
            reused_shape_obj = [None] * len(training_replica.shards[0].shape)
            # Create a mapping from global offset ranges to shards for both inference and training replicas
            inference_shard_offsets = self._create_shard_offset_for_replica(
                inference_replica,
                engine_rank,
                is_infer=True,
                global_transfer_rank=global_transfer_rank if is_infer_rank else None,
                reused_shape_obj=reused_shape_obj,
            )
            training_shard_offsets = self._create_shard_offset_for_replica(
                training_replica,
                0,
                is_infer=False,
                global_transfer_rank=global_transfer_rank if is_train_rank else None,
                reused_shape_obj=reused_shape_obj,
            )

            # Find overlapping regions between inference and training shards
            overlapping_regions = self._find_overlapping_regions(
                inference_shard_offsets,
                training_shard_offsets,
            )
            for region in overlapping_regions:
                region_plan = self._build_region_communication_plan(
                    region, reused_shape_obj
                )
                plan.extend(region_plan)

        return plan

    def _create_shard_offset_for_replica(
        self,
        replica: ParameterReplicaMeta,
        engine_rank: int,
        is_infer: bool,
        global_transfer_rank: int = None,
        reused_shape_obj=None,
    ) -> List[ShardOffset]:
        """
        Create a mapping from offset ranges to shards for a specific replica.

        Args:
            replica: ParameterReplicaMeta containing a list of shards
            is_infer: Whether this is for inference replicas

        Returns:
            List of dictionaries containing shard information with offset ranges
        """
        assert is_infer is not None, (
            "is_infer must be provided if global_transfer_rank is provided"
        )
        shard_map = []
        reused_shape_obj = reused_shape_obj or [None] * len(replica.shards[0].shape)

        if not replica.shards:
            raise ValueError("No shards found for replica")

        for shard in replica.shards:
            # Validate that global_offset is not empty
            if not shard.global_offset:
                raise ValueError("Shard has empty global_offset")

            # Validate that global_offset and shape have the same number of dimensions
            if len(shard.global_offset) != len(shard.shape):
                raise ValueError(
                    f"Dimension mismatch for shard in replica: "
                    f"global_offset has {len(shard.global_offset)} dims, shape has {len(shard.shape)} dims"
                )
            shard_transfer_rank = self._compute_shard_transfer_rank(
                shard, engine_rank, is_infer
            )
            if (
                global_transfer_rank is not None
                and shard_transfer_rank != global_transfer_rank
            ):
                continue

            # Calculate the end offset for this shard
            for i in range(len(shard.global_offset)):
                reused_shape_obj[i] = shard.global_offset[i] + shard.shape[i]
            end_offset = tuple(reused_shape_obj)
            shard_map.append(
                ShardOffset(
                    shard=shard,
                    start_offset=shard.global_offset,
                    end_offset=end_offset,
                    shape=shard.shape,
                    rank=shard_transfer_rank,
                )
            )

        return shard_map

    def _compute_shard_transfer_rank(
        self, shard: ParameterShardMeta, engine_rank: int, is_infer: bool
    ) -> int:
        if is_infer:
            return shard.global_rank + engine_rank * self.infer_instance_world_size
        else:
            assert engine_rank == 0, "Training only has one engine instance"
            return shard.global_rank + self.infer_world_size

    def _find_overlapping_regions(
        self,
        inference_map: List[ShardOffset],
        training_map: List[ShardOffset],
    ) -> List[OverlapRegion]:
        """Find overlapping regions between inference and training shards."""
        overlapping_regions = []

        for inf_shard_offset in inference_map:
            for train_shard_offset in training_map:
                # Check if shards overlap in all dimensions
                overlap_start = []
                overlap_end = []
                has_overlap = True

                for dim in range(len(inf_shard_offset.start_offset)):
                    start = max(
                        inf_shard_offset.start_offset[dim],
                        train_shard_offset.start_offset[dim],
                    )
                    end = min(
                        inf_shard_offset.end_offset[dim],
                        train_shard_offset.end_offset[dim],
                    )

                    if start >= end:
                        has_overlap = False
                        break

                    overlap_start.append(start)
                    overlap_end.append(end)

                if has_overlap:
                    overlapping_regions.append(
                        OverlapRegion(
                            inference_shard=inf_shard_offset,
                            training_shard=train_shard_offset,
                            overlap_start=tuple(overlap_start),
                            overlap_end=tuple(overlap_end),
                        )
                    )

        return overlapping_regions

    def _build_region_communication_plan(
        self,
        region: OverlapRegion,
        reused_shape_obj: List = None,
    ) -> List[CommunicationOperation]:
        """Build communication plan for an overlapping region."""
        plan = []

        inf_shard_offset = region.inference_shard
        train_shard_offset = region.training_shard
        overlap_start = region.overlap_start
        overlap_end = region.overlap_end
        reused_shape_obj = reused_shape_obj or [None] * len(
            region.inference_shard.shape
        )

        # Calculate the shape of the overlapping region
        overlap_shape = tuple(
            overlap_end[i] - overlap_start[i] for i in range(len(overlap_start))
        )

        # Calculate relative offsets within each shard
        for i in range(len(overlap_start)):
            reused_shape_obj[i] = overlap_start[i] - inf_shard_offset.start_offset[i]
        inf_relative_offset = tuple(reused_shape_obj)

        for i in range(len(overlap_start)):
            reused_shape_obj[i] = overlap_start[i] - train_shard_offset.start_offset[i]
        train_relative_offset = tuple(reused_shape_obj)

        # Create conversion functions for both training (sender) and inference (receiver)
        # Each side may need to slice if their shard is larger than the overlapping region
        for i in range(len(reused_shape_obj)):
            reused_shape_obj[i] = slice(
                train_relative_offset[i], train_relative_offset[i] + overlap_shape[i]
            )
        train_slices = tuple(reused_shape_obj)
        for i in range(len(reused_shape_obj)):
            reused_shape_obj[i] = slice(
                inf_relative_offset[i], inf_relative_offset[i] + overlap_shape[i]
            )
        infer_slices = tuple(reused_shape_obj)

        # Add to communication plan
        # Training is sender, inference is receiver
        # Only transfer the overlapping region
        send_rank = train_shard_offset.rank
        recv_rank = inf_shard_offset.rank
        assert send_rank != recv_rank, "Send and recv rank cannot be the same"
        assert send_rank >= self.infer_world_size, (
            f"Send rank {send_rank} is not in training world {self.infer_world_size, self.world_size}"
        )
        assert recv_rank < self.infer_world_size, (
            f"Recv rank {recv_rank} is not in inference world {0, self.infer_world_size}"
        )
        plan.append(
            CommunicationOperation(
                send_rank=send_rank,
                send_shard_meta=train_shard_offset.shard,
                send_offset=train_relative_offset,
                recv_rank=recv_rank,
                recv_shard_meta=inf_shard_offset.shard,
                recv_offset=inf_relative_offset,
                overlap_shape=overlap_shape,
                train_slices=train_slices,
                inf_slices=infer_slices,
            )
        )

        return plan

    def _group_operations_by_rank(
        self, operations: List[CommunicationOperation], key_name: str
    ) -> Dict[int, List[CommunicationOperation]]:
        grouped_operations = {}
        for operation in operations:
            key = getattr(operation, key_name)
            if key not in grouped_operations:
                grouped_operations[key] = []
            grouped_operations[key].append(operation)

        # Sort operations within each group to ensure consistent ordering
        for key in grouped_operations:
            grouped_operations[key].sort(
                key=lambda op: (
                    op.send_shard_meta.name,  # Sort by parameter name first
                    op.send_offset,  # Then by send offset
                    op.recv_offset,  # Then by recv offset
                )
            )
        # Sort the keys to ensure consistent iteration order
        sorted_grouped_operations = {}
        for key in sorted(grouped_operations.keys()):
            sorted_grouped_operations[key] = grouped_operations[key]
        return sorted_grouped_operations

    def build_local_transfer_plan(
        self,
        inference_weights_meta: List[ParameterMeta],
        training_weights_meta: List[ParameterMeta],
        global_transfer_rank: int,
    ) -> TransferPlan:
        is_train = global_transfer_rank >= self.infer_world_size
        name = "train" if is_train else "infer"
        start_time = time.time()
        logger.info(
            f"Rank[{global_transfer_rank}] Building local transfer plan for {name}"
        )
        num_infer_shards = (
            sum(
                len(replica.shards)
                for param in inference_weights_meta
                for replica in param.replicas
            )
            * self.num_infer_engines
        )
        num_train_shards = sum(
            len(replica.shards)
            for param in training_weights_meta
            for replica in param.replicas
        )
        logger.info(
            f"Rank[{global_transfer_rank}] Number of inference shards: {num_infer_shards}, number of training shards: {num_train_shards}"
        )
        total_shards = num_infer_shards + num_train_shards
        prune_threshold = 10000
        if total_shards > prune_threshold:
            logger.info(
                f"Rank[{global_transfer_rank}] Pruning global transfer plan for {name} because number of shards is too large: {total_shards}"
            )
            operations = self.build_weights_mapping_operations(
                inference_weights_meta,
                training_weights_meta,
                global_transfer_rank=global_transfer_rank,
            )
        else:
            logger.info(
                f"Rank[{global_transfer_rank}] Building global transfer plan for {name} because number of shards is small: {total_shards}"
            )
            operations = self.build_weights_mapping_operations(
                inference_weights_meta, training_weights_meta
            )
        if self.enable_debug_mode:
            data = to_dict(operations)
            file_name = f"global_communication_plan_{name}_{global_transfer_rank}_{os.getpid()}.json"
            with open(file_name, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Rank[{global_transfer_rank}] Saved communication plan to {os.path.abspath(file_name)}"
            )

        # Filter operations for this specific rank
        new_operations = []
        for operation in operations:
            if is_train:
                # For training ranks, only include operations where this rank is the sender
                if operation.send_rank == global_transfer_rank:
                    new_operations.append(operation)
            else:
                # For inference ranks, only include operations where this rank is the receiver
                if operation.recv_rank == global_transfer_rank:
                    new_operations.append(operation)

        # Group operations by the opposite rank (the rank we communicate with)
        # For training ranks: group by recv_rank (the inference rank we send to)
        # For inference ranks: group by send_rank (the training rank we receive from)
        key = "recv_rank" if is_train else "send_rank"
        grouped_operations = self._group_operations_by_rank(new_operations, key)

        # Validate that all operations in the grouped plan are for this specific rank
        for ops in grouped_operations.values():
            for op in ops:
                if is_train:
                    assert op.send_rank == global_transfer_rank, (
                        f"Training rank {global_transfer_rank} local plan contains operation "
                        f"with send_rank {op.send_rank} instead of {global_transfer_rank}"
                    )
                else:
                    assert op.recv_rank == global_transfer_rank, (
                        f"Inference rank {global_transfer_rank} local plan contains operation "
                        f"with recv_rank {op.recv_rank} instead of {global_transfer_rank}"
                    )
        duration = time.time() - start_time
        logger.info(
            f"Rank[{global_transfer_rank}] ({'training' if is_train else 'inference'}) "
            f"local plan has {len(new_operations)} operations grouped by {len(grouped_operations)} "
            f"opposite ranks: {list(grouped_operations.keys())}, "
            f"global operations across all ranks: {len(operations)}, "
            f"took time: {duration:.2f} seconds"
        )

        if self.enable_debug_mode:
            data = to_dict(grouped_operations)
            file_name = f"local_communication_plan_{name}_{global_transfer_rank}_{os.getpid()}.json"
            with open(file_name, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Rank[{global_transfer_rank}] Saved communication plan to {os.path.abspath(file_name)}"
            )
        return TransferPlan(inter_operations=grouped_operations)


@torch.no_grad()
def slice_tensor(
    tensor: torch.Tensor, op: CommunicationOperation, is_train, **kwargs
) -> torch.Tensor:
    """
    Slice the overlapping region from the source tensor.

    Args:
        tensor: The source tensor to slice

    Returns:
        The sliced tensor containing only the overlapping region
    """
    slices = op.train_slices if is_train else op.inf_slices
    sliced_tensor = tensor[slices]
    if not sliced_tensor.is_contiguous():
        param_name = op.send_shard_meta.name
        source_offset = op.send_offset if is_train else op.recv_offset
        if is_train:
            slice_context = kwargs.get("slice_context", {})
            key = f"{param_name}-{slices}"
            sliced = slice_context.get(key)
            if sliced is not None:
                return sliced
            sliced = sliced_tensor.contiguous()
            slice_context[key] = sliced
            return sliced
        else:
            msg = (
                f"Sliced tensor is not contiguous, param_name: {param_name}, "
                f"inference_meta: {op.recv_shard_meta}, training_meta: {op.send_shard_meta}, "
                f"source_offset: {source_offset}, overlap_shape: {op.overlap_shape}, "
                f"tensor shape: {tensor.shape}, slices: {slices}, contiguous: {tensor.is_contiguous()}"
            )
            raise ValueError(msg)
    return sliced_tensor


def compute_transfer_plan_hash(
    transfer_plan: TransferPlan,
) -> str:
    """Create a deterministic hash for plan observability/debugging."""

    def _slice_tuple(slices: Iterable[slice]) -> Tuple[Tuple[int, int, int], ...]:
        return tuple((s.start, s.stop, s.step or 1) for s in slices)

    payload = {"inter": []}
    for peer_rank in sorted(transfer_plan.inter_operations.keys()):
        for op in transfer_plan.inter_operations[peer_rank]:
            payload["inter"].append(
                (
                    peer_rank,
                    op.send_rank,
                    op.recv_rank,
                    op.send_shard_meta.name,
                    op.recv_shard_meta.name,
                    tuple(op.send_offset),
                    tuple(op.recv_offset),
                    tuple(op.overlap_shape),
                    _slice_tuple(op.train_slices),
                    _slice_tuple(op.inf_slices),
                    op.pp_rank,
                    op.placement_version,
                    op.param_class,
                )
            )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _shape_numel(shape: Iterable[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _dtype_element_size(dtype: Any) -> int:
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    if isinstance(dtype, str):
        normalized = dtype.replace("torch.", "")
        maybe_dtype = getattr(torch, normalized, None)
        if isinstance(maybe_dtype, torch.dtype):
            return torch.empty((), dtype=maybe_dtype).element_size()
    # Keep a conservative default for unknown dtypes.
    return 4


def compute_transfer_plan_stats(
    transfer_plan: TransferPlan,
) -> Dict[str, Any]:
    """Return structured observability stats for a transfer plan."""
    inter_ops = 0
    inter_numel = 0
    inter_bytes = 0
    for ops in transfer_plan.inter_operations.values():
        inter_ops += len(ops)
        for op in ops:
            op_numel = _shape_numel(op.overlap_shape)
            inter_numel += op_numel
            inter_bytes += op_numel * _dtype_element_size(op.send_shard_meta.dtype)

    total_ops = inter_ops
    total_numel = inter_numel
    total_bytes = inter_bytes
    return {
        "inter_hash": compute_transfer_plan_hash(transfer_plan),
        "full_hash": compute_transfer_plan_hash(transfer_plan),
        "inter": {
            "peer_count": len(transfer_plan.inter_operations),
            "op_count": inter_ops,
            "numel": inter_numel,
            "bytes": inter_bytes,
        },
        "total": {
            "op_count": total_ops,
            "numel": total_numel,
            "bytes": total_bytes,
        },
    }
