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

from awex.sharding.param_sharding import ShardingType, get_default_sharding_dim


class LinearMLAShardingMixin:
    def get_sharding_strategy(self, parameter_name, **kwargs):
        if any(
            key in parameter_name
            for key in (
                "attention.kv_a_proj_with_mqa.weight",
                "attention.q_a_proj.weight",
                "attention.fused_qkv_a_proj_with_mqa.weight",
                "attention.kv_a_layernorm.weight",
                "attention.q_a_layernorm.weight",
            )
        ):
            sharding_dim = get_default_sharding_dim(parameter_name)
            return ShardingType.NO_SHARDING, sharding_dim, 1

        if "attention.g_norm.weight" in parameter_name:
            sharding_dim = get_default_sharding_dim(parameter_name)
            tp_size = self.rank_info.tp_size
            if tp_size > 1:
                return ShardingType.TP_SHARDING, sharding_dim, tp_size
            return ShardingType.NO_SHARDING, sharding_dim, 1

        return super().get_sharding_strategy(parameter_name, **kwargs)
