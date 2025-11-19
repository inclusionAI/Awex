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

from typing import List

from awex import logging
from awex.writer.weights_writer import WeightsExchangeShardingWriter

logger = logging.getLogger(__name__)


class AStateWeightsWriter(WeightsExchangeShardingWriter):
    def _initialize(self):
        super()._initialize()
        logger.info(
            f"Start to initialize AStateWeightsWriter for rank {self.rank_info.global_rank}"
        )
        from astate import ParallelConfig, create_remote_table

        config = ParallelConfig.create_training_config(
            role_size=self.rank_info.world_size,
            role_rank=self.rank_info.global_rank,
            dp_size=self.rank_info.dp_size,
            dp_rank=self.rank_info.dp_rank,
            tp_size=self.rank_info.tp_size,
            tp_rank=self.rank_info.tp_rank,
            pp_size=self.rank_info.pp_size,
            pp_rank=self.rank_info.pp_rank,
            cp_size=1,
            cp_rank=0,
            ep_size=self.rank_info.ep_size,
            ep_rank=self.rank_info.ep_rank,
            etp_size=self.rank_info.ep_tp_size,
            etp_rank=self.rank_info.ep_tp_rank,
        )
        self.table = create_remote_table("weights_exchange", parallel_config=config)
        logger.info(
            f"Finished initializing AStateWeightsWriter for rank {self.rank_info.global_rank}"
        )

    def write_tensors(self, step_id, tensor_pairs: List, **kwargs):
        from astate.utils import create_sharded_key

        astate_tensor_pairs = []
        for name, parameter, shard_meta, meta in tensor_pairs:
            global_shape = meta.global_shape
            global_offset = shard_meta.global_offset
            shard_key = create_sharded_key(name, global_shape, global_offset)
            astate_tensor_pairs.append((shard_key, parameter))
        self.table.multi_put(step_id, astate_tensor_pairs)

    def finish_step(self, step_id):
        logger.info(f"Start to complete table for step {step_id}")
        self.table.complete(step_id)
        logger.info(f"Finished completing table for step {step_id}")
