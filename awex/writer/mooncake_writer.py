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

"""Training-side Mooncake weight exchange."""

from awex.transfer.mooncake_comm import MooncakeTransport
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.writer.weights_writer import WeightsExchangeShardingWriter


class MooncakeWeightsWriter(WeightsExchangeShardingWriter):
    def _initialize(self):
        super()._initialize()
        self.transfer_plan = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_infer_engines,
            self.enable_debug_mode,
        ).build_local_transfer_plan(
            self.infer_params_meta, self.parameters_meta, self.transfer_rank
        )
        self.required_param_names = {
            op.send_shard_meta.name
            for operations in self.transfer_plan.operations.values()
            for op in operations
        }
        self.transport = MooncakeTransport(
            self.meta_server_client, self.transfer_rank, self.timeout
        )

    def _write_weights(self, step_id, **kwargs):
        if self.enable_colocate_mode:
            self.train_engine.release_grad_memory()
        parameters = (
            self.convert_parameters(required_names=self.required_param_names)
            if self.required_param_names
            else {}
        )
        self.transport.send(
            parameters,
            self.transfer_plan,
            step_id,
            offload=self.enable_colocate_mode,
            on_ready=self._offload_weights if self.enable_colocate_mode else None,
        )

    def _offload_weights(self):
        self.train_engine.release_memory_occupation("weights")
        self.meta_server_client.add_object_to_set(
            "all_training_offloaded_weights", self.transfer_rank
        )
