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

from awex.sharding.rank_info import RankInfo


def get_sglang_sharding_strategy(
    model_name: str, infer_engine_config, rank_info: RankInfo, **kwargs
):
    """
    Get the sharding strategy class for a given model architecture name.
    """
    from awex.models import get_sharding_strategy

    cls = get_sharding_strategy(model_name)
    return cls(
        engine_name="sglang",
        enable_dp_attention=infer_engine_config.enable_dp_attention,
        enable_dp_lm_head=infer_engine_config.enable_dp_lm_head,
        moe_dense_tp_size=infer_engine_config.moe_dense_tp_size,
        tp_size=rank_info.tp_size,
        ep_size=infer_engine_config.ep_size,
        ep_tp_size=rank_info.ep_tp_size,
        rank_info=rank_info,
        **kwargs,
    )


def get_sglang_rank_info(model_context, engine_rank) -> RankInfo:
    scheduler = model_context["scheduler"]
    infer_engine_config = scheduler.server_args
    if infer_engine_config.dp_size != 1 and not infer_engine_config.enable_dp_attention:
        raise ValueError(
            f"DP size is not 1, but {infer_engine_config.dp_size}. This is not supported yet."
        )
    tp_size = model_context["tp_size"]
    tp_rank = model_context["tp_rank"]
    ep_size = infer_engine_config.ep_size
    if ep_size > 1:
        ep_tp_size = tp_size // ep_size
        ep_tp_rank = tp_rank % ep_tp_size
        ep_rank = tp_rank // ep_tp_size
    else:
        assert ep_size == 1, "ep_size must be 1"
        ep_rank = 0
        ep_tp_size = 1
        ep_tp_rank = 0
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=model_context["pp_rank"],
        pp_size=model_context["pp_size"],
        dp_size=model_context["dp_size"],
        dp_rank=0,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=model_context["attn_tp_rank"],
        attn_tp_size=model_context["attn_tp_size"],
        attn_dp_rank=model_context["attn_dp_rank"],
        world_size=model_context["world_size"],
        global_rank=model_context["global_rank"],
        engine_rank=engine_rank,
        local_rank=model_context["local_rank"],
        is_infer=True,
    )
