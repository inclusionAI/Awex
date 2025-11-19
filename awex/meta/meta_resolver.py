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

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from awex import logging
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import (
    ShardingType,
)
from awex.sharding.rank_info import RankInfo

logger = logging.getLogger(__name__)


class ParamMetaResolver(ABC):
    """
    Resolves and reconstructs parameter metadata for a distributed model, including sharding and replica information.
    """

    def __init__(self, hf_config):
        self.hf_config = hf_config
        self.num_hidden_layers = hf_config.num_hidden_layers

    @abstractmethod
    def get_model_arch_name(self) -> str:
        """
        Returns the name of the model architecture.
        """
        pass

    @abstractmethod
    def get_parameters_meta(self) -> List[ParameterMeta]:
        """
        Returns the list of ParameterMeta objects for all parameters in the model.
        """
        pass

    @abstractmethod
    def _get_params_raw_meta(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def _get_sharding_info(
        self, name: str, rank_info: RankInfo, param_meta: Dict[str, Any]
    ) -> Tuple[ShardingType, int, int]:
        """
        Get the sharding information for a parameter.
        Returns (ShardingType, sharding_dim, num_shards).
        """
        pass

    def _build_params_meta(self) -> List[ParameterMeta]:
        """
        Build and return a list of ParameterMeta objects with global (unsharded) size, shape, and relative offset
        for each parameter in the model.

        This function processes the parameter metadata collected from all model workers (shards).
        For each parameter, it reconstructs the global shape and size by aggregating sharded information,
        and computes the offset of each shard within the global parameter.

        Returns:
            List[ParameterMeta]: A list of ParameterMeta objects, each containing all replicas and shards for a parameter.
        """
        all_params_raw_meta = (
            self._get_params_raw_meta()
        )  # List of meta dicts, one per rank
        param_shards = defaultdict(list)
        param_sharding_info = {}
        for rank_meta in all_params_raw_meta:
            rank_info: RankInfo = rank_meta["rank_info"]
            for param_meta in rank_meta["params_meta"]:
                name = param_meta["name"]
                sharding_type, sharding_dim, num_shards = self._get_sharding_info(
                    name, rank_info, param_meta
                )
                shard_info = ParameterShardMeta(
                    name=name,
                    tp_rank=rank_info.tp_rank,
                    attn_tp_rank=rank_info.attn_tp_rank,
                    pp_rank=rank_info.pp_rank,
                    ep_rank=rank_info.ep_rank,
                    ep_tp_rank=rank_info.ep_tp_rank,
                    global_rank=rank_info.global_rank,
                    engine_rank=rank_info.engine_rank,
                    world_size=rank_info.world_size,
                    shape=param_meta["shape"],
                    numel=param_meta["numel"],
                    dtype=param_meta["dtype"],
                    sharding_type=sharding_type,
                    num_shards=num_shards,
                    sharding_dim=sharding_dim,
                )
                param_shards[name].append(shard_info)
                param_sharding_info[name] = (sharding_type, sharding_dim, num_shards)
        replicas = {}
        for name, shards in param_shards.items():
            sharding_type, sharding_dim, num_shards = param_sharding_info[name]

            if sharding_type == ShardingType.NO_SHARDING:
                # FIX: For NO_SHARDING, each rank should be its own replica with one shard
                replicas[name] = [[shard] for shard in shards]
            elif num_shards == 1:
                # For single shard parameters, put all shards in one replica
                replicas[name] = [shards]
            else:
                # For sharded parameters, group by rank
                if sharding_type == ShardingType.TP_SHARDING:
                    rank_key = "tp_rank"
                elif sharding_type == ShardingType.DP_TP_SHARDING:
                    rank_key = "attn_tp_rank"
                elif sharding_type == ShardingType.EP_TP_SHARDING:
                    rank_key = "ep_tp_rank"
                elif sharding_type == ShardingType.EP_SHARDING:
                    rank_key = "ep_rank"
                else:
                    raise ValueError(f"Unknown sharding_type: {sharding_type}")

                rank_to_shards = defaultdict(list)
                for shard in shards:
                    rank_to_shards[getattr(shard, rank_key)].append(shard)

                replica_counts = [
                    len(shard_list) for shard_list in rank_to_shards.values()
                ]
                assert all(count == replica_counts[0] for count in replica_counts), (
                    f"Inconsistent number of replicas across {rank_key}s for param {name}: {replica_counts}"
                )

                num_replicas = replica_counts[0]
                replica_groups = []
                for replica_idx in range(num_replicas):
                    replica = []
                    for rk in sorted(rank_to_shards.keys()):
                        shard_list = rank_to_shards[rk]
                        assert replica_idx < len(shard_list), (
                            f"{replica_idx} {len(shard_list)}"
                        )
                        replica.append(shard_list[replica_idx])
                    replica.sort(key=lambda x: x.tp_rank)
                    replica_groups.append(replica)
                replicas[name] = replica_groups
        params_meta = []
        for name, shards in param_shards.items():
            dtype = shards[0].dtype
            num_dims = len(shards[0].shape)
            first_replica = replicas[name][0]
            sharding_dim = shards[0].sharding_dim
            # only sum up the sharding dimension, other dimensions are the same as the first replica
            global_shape = tuple(
                (
                    sum(shard.shape[i] for shard in first_replica)
                    if i == sharding_dim
                    else first_replica[0].shape[i]
                )
                for i in range(num_dims)
            )
            global_numel = sum(shard.numel for shard in first_replica)
            # Compute global offsets per replica, not across all shards
            for replica in replicas[name]:
                prev_offsets = [0] * num_dims
                for shard in replica:
                    shard.global_offset = tuple(prev_offsets)
                    for i in range(num_dims):
                        if i == sharding_dim:
                            prev_offsets[i] += shard.shape[i]
                        else:
                            prev_offsets[i] = shard.global_offset[i]
            param_meta = ParameterMeta(
                name=name,
                global_numel=global_numel,
                global_shape=global_shape,
                dtype=dtype,
                shards=shards,
                replicas=[
                    ParameterReplicaMeta(shards=replica) for replica in replicas[name]
                ],
            )
            params_meta.append(param_meta)
        num_shards = sum(
            len(replica.shards) for param in params_meta for replica in param.replicas
        )
        logger.info(f"Number of shards: {num_shards}")
        return params_meta
