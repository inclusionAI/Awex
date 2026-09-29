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

from awex.config import InferenceConfig as InferenceConfig

__all__ = [
    "InferenceConfig",
    "NCCLWeightsWriter",
    "WeightsReader",
    "NCCLWorkerWeightsReader",
]


def __getattr__(name):
    # Config-only callers and packaging tools do not need runtime backends.
    from importlib import import_module

    modules = {
        "NCCLWeightsWriter": "awex.writer.nccl_writer",
        "WeightsReader": "awex.reader.weights_reader",
        "NCCLWorkerWeightsReader": "awex.reader.nccl_reader",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
