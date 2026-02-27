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

from types import SimpleNamespace

from transformers import PretrainedConfig

from awex.models.registry import get_train_weights_converter
from awex.sharding.rank_info import RankInfo


def _make_rank_info() -> RankInfo:
    return RankInfo(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=1,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=False,
    )


def test_bailing_moe_train_converter_accepts_tf_config():
    cfg = PretrainedConfig()
    cfg.quantization_config = {}
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace()

    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    assert converter.tf_config is tf_config
