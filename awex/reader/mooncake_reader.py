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

"""Inference-side Mooncake weight exchange."""

from awex.reader.weights_reader import WorkerWeightsReader
from awex.transfer.mooncake_comm import MooncakeTransport
from awex.transfer.transfer_plan import TransferPlanBuilder


class MooncakeWorkerWeightsReader(WorkerWeightsReader):
    def initialize(self):
        super().initialize()
        self.transfer_plan = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        ).build_local_transfer_plan(
            self.parameters_meta, self.training_params_meta, self.transfer_rank
        )
        self.transport = MooncakeTransport(
            self.meta_server_client, self.transfer_rank, self.timeout
        )

    def _update_weights(self, step_id, **kwargs):
        # Offload/resume can replace parameter allocations. Rebuild converter
        # views for every exchange rather than keeping stale device pointers.
        super().initialize()
        self.transport.receive(self.parameters, self.transfer_plan, step_id)
