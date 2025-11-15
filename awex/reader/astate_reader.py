from awex from awex import logging
from typing import List

from awex.reader.weights_reader import WorkerWeightsReader
from awex.sharding.param_sharding import (
    get_sglang_rank_info,
)

logger = logging.getLogger(__name__)

class AStateWorkerWeightsReader(WorkerWeightsReader):
    def initialize(self):
        super().initialize()
        coordinate = f"{self.engine_rank}-{self.rank_info.global_rank}"
        logger.info(
            f"Start to initialize AStateWorkerWeightsReader for rank {coordinate}"
        )
        world_size = self.rank_info.world_size * self.num_engines
        role_rank = self.rank_info.global_rank + (
            self.rank_info.world_size * self.engine_rank
        )
        from astate import ParallelConfig, create_remote_table

        config = ParallelConfig.create_inference_config(
            role_size=world_size,
            role_rank=role_rank,
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
            f"Finished initializing AStateWorkerWeightsReader for rank {coordinate}"
        )

    def read_tensors(
        self,
        step_id: int,
        tensor_pairs: List,
        **kwargs,
    ):
        from astate.utils import create_sharded_key

        state_tensor_pairs = []
        for name, parameter, shard_meta, meta in tensor_pairs:
            global_shape = meta.global_shape
            global_offset = shard_meta.global_offset
            shard_key = create_sharded_key(name, global_shape, global_offset)
            state_tensor_pairs.append((shard_key, parameter))
        # inplace update weights
        self.table.multi_get(step_id, state_tensor_pairs)

    def finish_step(self, step_id):
        logger.info(f"Start to complete table for step {step_id}")
        self.table.complete(step_id)
        logger.info(f"Finished completing table for step {step_id}")
