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

"""Converter for vLLM parameter names to HF/Megatron-friendly names.

We normalize vLLM self-attention names to the HF-style projection naming
used by Awex converters (e.g., qkv/qkv_proj -> query_key_value_proj).
"""

from awex.converter.sglang_converter import SGlangToHFWeightConverter


class VLLMToHFWeightConverter(SGlangToHFWeightConverter):
    def _normalize_name(self, name: str) -> str:
        replacements = [
            (".self_attn.attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.attn.o_proj", ".attention.dense"),
            (".self_attn.o_proj", ".attention.dense"),
            (".self_attn.proj", ".attention.dense"),
            (".self_attn.q_norm", ".attention.query_layernorm"),
            (".self_attn.k_norm", ".attention.key_layernorm"),
        ]
        for old, new in replacements:
            if old in name:
                name = name.replace(old, new)
        # Guard against double normalization.
        name = name.replace("query_key_value_proj_proj", "query_key_value_proj")
        return name

    def convert_param(self, name, parameter):
        return super().convert_param(self._normalize_name(name), parameter)
