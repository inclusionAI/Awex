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


@dataclass(slots=True)
class RankInfo:
    """
    Holds information about the distributed training ranks and sizes for a model worker.

    Attributes:
        tp_rank (int): Tensor parallel rank.
        tp_size (int): Tensor parallel size.
        pp_rank (int): Pipeline parallel rank.
        pp_size (int): Pipeline parallel size.
        dp_size (int): Data parallel size.
        attn_tp_rank (int): Attention tensor parallel rank.
        attn_tp_size (int): Attention tensor parallel size.
        attn_dp_rank (int): Attention data parallel rank.
        world_size (int): Total world size.
        global_rank (int): Global rank of the worker.
        local_rank (int): Local rank of the worker.
        engine_rank (int): Engine rank of the worker.
        is_infer (bool): Whether the worker is an inference worker.
    """

    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    dp_size: int
    dp_rank: int
    ep_rank: int
    ep_size: int
    ep_tp_rank: int
    ep_tp_size: int
    attn_tp_rank: int
    attn_tp_size: int
    attn_dp_rank: int
    world_size: int
    global_rank: int
    local_rank: int
    engine_rank: int
    is_infer: bool
