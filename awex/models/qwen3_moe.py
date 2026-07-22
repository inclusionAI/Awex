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

from typing import List, Tuple

import torch

from awex import logging
from awex.converter.sglang_converter import SGlangToHFWeightConverter

logger = logging.getLogger(__name__)


class SGlangToHFWeightConverterQwen3Moe(SGlangToHFWeightConverter):
    """SGLang -> HF converter for Qwen3-MoE.

    Splits the SGLang fused qkv_proj into canonical q/k/v projections
    (GQA-aware split in the base class) so that inference-side weight
    metadata matches the per-parameter HF names emitted by the Megatron
    train-side converter. Expert parameters (experts.w13_weight /
    experts.w2_weight) are expanded per expert by the base class.
    """

    def _fuse_qkv(self, name: str) -> bool:
        # Train side reports canonical self_attn.{q,k,v}_proj names, so the
        # inference side must unfuse qkv_proj for transfer-plan matching.
        return False

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        # Qwen3 uses self_attn.{q,k}_norm; the base class only recognizes
        # the bailing-style {query,key}_layernorm names.
        if "q_norm" in name or "k_norm" in name:
            return [(name, parameter)]
        return super()._convert_layer_norm_param(name, parameter, layer_number)


def _build_mcore_converter_qwen3_moe():
    # Lazily import Megatron converter to avoid MindSpeed patching in
    # vLLM-only paths (same pattern as ling.py).
    from awex.converter.mcore_converter import McoreToHFWeightConverter

    class McoreToHFWeightConverterQwen3Moe(McoreToHFWeightConverter):
        """Stock HF/sglang qwen3_moe serves canonical attention names
        (self_attn.{q,k,v,o}_proj + self_attn.{q,k}_norm), so keep them
        verbatim instead of the bailing-flavored renames, and split the
        Megatron fused linear_qkv with GQA-aware group strides — the base
        equal-thirds split only holds when q/k/v head counts match.
        """

        def _fuse_qkv(self, name: str) -> bool:
            return False

        @staticmethod
        def _normalize_attn_name(name: str) -> str:
            return name

        def _gqa_head_dim(self) -> int:
            head_dim = getattr(self.hf_config, "head_dim", None)
            if head_dim:
                return int(head_dim)
            return int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)

        def _split_gqa_qkv(
            self, parameter: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hf = self.hf_config
            head_dim = self._gqa_head_dim()
            attn_tp = max(1, int(getattr(self.rank_info, "attn_tp_size", 1)))
            if hf.num_key_value_heads % attn_tp != 0:
                raise ValueError(
                    f"num_key_value_heads ({hf.num_key_value_heads}) must be "
                    f"divisible by attn_tp_size ({attn_tp})"
                )
            if hf.num_attention_heads % hf.num_key_value_heads != 0:
                raise ValueError(
                    f"num_attention_heads ({hf.num_attention_heads}) must be "
                    f"divisible by num_key_value_heads "
                    f"({hf.num_key_value_heads})"
                )
            num_groups = hf.num_key_value_heads // attn_tp
            q_per_group = hf.num_attention_heads // hf.num_key_value_heads
            group_rows = (q_per_group + 2) * head_dim
            expected_rows = num_groups * group_rows
            if parameter.shape[0] != expected_rows:
                raise ValueError(
                    "Unexpected linear_qkv rows for GQA split: "
                    f"got {parameter.shape[0]}, expected {expected_rows} "
                    f"(kv_heads={hf.num_key_value_heads}, attn_tp={attn_tp}, "
                    f"q_per_group={q_per_group}, head_dim={head_dim})"
                )
            blocks = parameter.reshape(num_groups, group_rows, *parameter.shape[1:])
            q_rows = q_per_group * head_dim
            q = blocks[:, :q_rows].reshape(num_groups * q_rows, *parameter.shape[1:])
            k = blocks[:, q_rows : q_rows + head_dim].reshape(
                num_groups * head_dim, *parameter.shape[1:]
            )
            v = blocks[:, q_rows + head_dim :].reshape(
                num_groups * head_dim, *parameter.shape[1:]
            )
            return q.contiguous(), k.contiguous(), v.contiguous()

        def _convert_attention_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> List[Tuple[str, torch.Tensor]]:
            if "self_attention.linear_qkv.weight" in name or (
                "self_attention.linear_qkv.bias" in name
            ):
                suffix = "weight" if name.endswith("weight") else "bias"
                q, k, v = self._split_gqa_qkv(parameter)
                return [
                    (f"self_attn.q_proj.{suffix}", q),
                    (f"self_attn.k_proj.{suffix}", k),
                    (f"self_attn.v_proj.{suffix}", v),
                ]
            return super()._convert_attention_param(name, parameter, layer_number)

    return McoreToHFWeightConverterQwen3Moe


CONFIG = {
    "model_name": "Qwen3MoeForCausalLM",
    "mcore_converter": _build_mcore_converter_qwen3_moe,
    "sglang_converter": SGlangToHFWeightConverterQwen3Moe,
}
