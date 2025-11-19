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

import torch
import torch.distributed as dist

from awex.sharding.rank_info import RankInfo


def get_mcore_sharding_strategy(model_name: str, rank_info: RankInfo, **kwargs):
    from awex.models import get_sharding_strategy

    cls = get_sharding_strategy(model_name)
    return cls(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=rank_info.tp_size,
        tp_size=rank_info.tp_size,
        ep_size=rank_info.ep_size,
        ep_tp_size=rank_info.ep_tp_size,
        rank_info=rank_info,
        **kwargs,
    )


def get_mcore_rank_info() -> RankInfo:
    from megatron.core import parallel_state as mpu

    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    ep_rank = mpu.get_expert_model_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_tp_size = mpu.get_expert_tensor_parallel_world_size()
    ep_tp_rank = mpu.get_expert_tensor_parallel_rank()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_gpu_per_node = torch.cuda.device_count()
    local_rank = dist.get_rank() % num_gpu_per_node
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_size=dp_size,
        dp_rank=dp_rank,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_dp_rank=dp_rank,
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        engine_rank=0,
        is_infer=False,
    )
