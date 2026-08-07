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

"""Registration for dense Qwen3 (``Qwen3ForCausalLM``).

Dense and MoE Qwen3 share the layout both converters target: canonical
``self_attn.{q,k,v,o}_proj`` with per-head ``self_attn.{q,k}_norm``, GQA head
grouping in the fused Megatron ``linear_qkv``, and gate/up projections that
sglang serves fused while the train side reports them split. The dense model
simply has no expert parameters, so the shared converters need no change; only
the architecture name has to be registered.
"""

from awex.models.qwen3_moe import (
    SGlangToHFWeightConverterQwen3Moe,
    _build_mcore_converter_qwen3_moe,
)

CONFIG = {
    "model_name": "Qwen3ForCausalLM",
    "mcore_converter": _build_mcore_converter_qwen3_moe,
    "sglang_converter": SGlangToHFWeightConverterQwen3Moe,
}
