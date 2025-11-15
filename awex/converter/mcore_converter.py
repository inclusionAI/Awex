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

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import distributed as dist
from transformers import PretrainedConfig

from awex.sharding.rank_info import RankInfo
from awex.util.common import divide


def _process_mcore_pp_name(
    name: str, rank_info: RankInfo, hf_config: PretrainedConfig
) -> str:
    """
    Process the name of a parameter to remove the pipeline parallel rank.
    """
    if "layers." in name:
        from megatron.training import get_args

        args = get_args()

        # model.layers.0.attention.dense.weight
        left, remains = name.rsplit(".layers.", 1)
        splits = remains.split(".")
        local_layer_id = splits[0]
        num_hidden_layers = hf_config.num_hidden_layers
        pp_rank_start_ids = [0] * rank_info.pp_size
        if args.decoder_first_pipeline_num_layers is not None:
            pp_rank_start_ids[0] = args.decoder_first_pipeline_num_layers
        if args.decoder_last_pipeline_num_layers is not None:
            pp_rank_start_ids[-1] = args.decoder_last_pipeline_num_layers
        intermediate_pp_size = len([item for item in pp_rank_start_ids if item == 0])
        local_pp_size = (
            num_hidden_layers - sum(pp_rank_start_ids)
        ) // intermediate_pp_size
        for index, start in enumerate(pp_rank_start_ids):
            if start != 0:
                continue
            pp_rank_start_ids[index] = local_pp_size
        pp_rank_start_ids.insert(0, 0)
        start_offsets = list(np.cumsum(pp_rank_start_ids))
        start_offsets.pop(-1)
        current_start_id = start_offsets[rank_info.pp_rank]
        global_layer_id = int(local_layer_id) + current_start_id
        # Reconstruct the remaining part after the layer ID
        remaining_parts = ".".join(splits[1:])
        return f"{left}.layers.{global_layer_id}.{remaining_parts}"
    else:
        return name


class McoreToHFWeightConverter:
    def __init__(
        self, hf_config: PretrainedConfig, rank_info: RankInfo, infer_conf: Dict
    ):
        self.hf_config = hf_config
        self.rank_info = rank_info
        self.infer_conf = infer_conf
        self.router_dtype = infer_conf.get("router_dtype", "bf16")
        if self.router_dtype == "bf16":
            self.router_dtype = torch.bfloat16
        elif self.router_dtype == "fp16":
            self.router_dtype = torch.float16
        elif self.router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported router dtype: {self.router_dtype}")
        if "infer_atten_tp_size" not in infer_conf:
            raise ValueError("infer_atten_tp_size must be specified")
        self.infer_atten_tp_size = infer_conf["infer_atten_tp_size"]

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if "self_attention.linear_qkv.weight" in name:
            if self._fuse_qkv(name):
                # Keep fused format
                parameter = convert_qkv_weight_along_tp_attention(
                    parameter, self.infer_atten_tp_size
                )
                return [("attention.query_key_value.weight", parameter)]
            else:
                # Split into separate Q, K, V projections
                shape0 = parameter.shape[0]
                stride = shape0 // 3
                return [
                    ("attention.q_proj.weight", parameter.narrow(0, 0, stride)),
                    ("attention.k_proj.weight", parameter.narrow(0, stride, stride)),
                    (
                        "attention.v_proj.weight",
                        parameter.narrow(0, 2 * stride, stride),
                    ),
                ]
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            return [("input_layernorm.weight", parameter)]
        elif "self_attention.linear_proj.weight" in name:
            return [("attention.dense.weight", parameter)]
        elif "self_attention.q_layernorm.weight" in name:
            return [("attention.query_layernorm.weight", parameter)]
        elif "self_attention.k_layernorm.weight" in name:
            return [("attention.key_layernorm.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_linear(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        if "linear_fc1.weight" in name:
            if self._fuse_gate_up_proj(name):
                return [("gate_up_proj.weight", parameter)]
            else:
                # split gate_proj and up_proj
                return [
                    (
                        "gate_proj.weight",
                        parameter.narrow(0, 0, parameter.shape[0] // 2),
                    ),
                    (
                        "up_proj.weight",
                        parameter.narrow(
                            0, parameter.shape[0] // 2, parameter.shape[0] // 2
                        ),
                    ),
                ]
        elif "linear_fc2.weight" in name:
            return [("down_proj.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_gate(
        self, name: str, parameter: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        assert "router" in name or "gate." in name, (
            f"Unsupported parameter name: {name}"
        )
        return ("mlp.gate.weight", parameter.to(self.router_dtype))

    def _convert_mlp_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        assert "attention" not in name, (
            f"{name} shouble be hanled in _convert_attention_param"
        )
        if "pre_mlp_layernorm" in name or "linear_fc1.layer_norm_weight" in name:
            return [("post_attention_layernorm.weight", parameter)]
        if "input_layernorm.weight" in name:
            return [("input_layernorm.weight", parameter)]
        elif "shared_experts.gate_weight" in name:
            return [("mlp.shared_expert_gate.weight", parameter)]
        elif "shared_experts.linear_fc1.weight" in name:
            return [
                (f"mlp.shared_experts.{name}", param)
                for name, param in self._convert_linear(name, parameter)
            ]
        elif "shared_experts.linear_fc2.weight" in name:
            return [
                (f"mlp.shared_experts.{name}", param)
                for name, param in self._convert_linear(name, parameter)
            ]
        elif "mlp.experts." in name:
            if "local_experts" in name:
                # mlp.experts.local_experts.0.linear_fc1.weight
                local_expert_id = int(name.rsplit(".", 3)[-3])
            else:
                # mlp.experts.linear_fc1.weight0
                local_expert_id = int(name.rsplit("weight", 1)[-1])
            num_experts = self.hf_config.num_experts
            num_experts_per_partition = num_experts // self.rank_info.ep_size
            expert_id = (
                local_expert_id + self.rank_info.ep_rank * num_experts_per_partition
            )
            return [
                (f"mlp.experts.{expert_id}.{name}", param)
                for name, param in self._convert_linear(name, parameter)
            ]
        elif "linear_fc1.weight" in name or "linear_fc2.weight" in name:
            return [
                (f"mlp.{name}", param)
                for name, param in self._convert_linear(name, parameter)
            ]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_expert_bias_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> Tuple[str, torch.Tensor]:
        """Convert bias parameters"""
        if "expert_bias" in name:
            return ("mlp.gate.expert_bias", parameter.to(torch.bfloat16))
        else:
            raise NotImplementedError(f"Unsupported bias parameter name: {name}")

    def _fuse_qkv(self, name: str) -> bool:
        """Override this method to control QKV fusion behavior"""
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_lm_head_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        if self.hf_config.norm_head:
            import torch.nn.functional as F

            parameter = F.normalize(parameter, dim=0, p=2, eps=1e-7)
        return [("lm_head.weight", parameter)]

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        name = name.replace("module.", "")
        name = _process_mcore_pp_name(name, self.rank_info, self.hf_config)
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.word_embeddings.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        if name in direct_name_mapping:
            return [(direct_name_mapping[name], parameter)]
        if "output_layer.weight" in name:
            return self._convert_lm_head_param(name, parameter)
        name = name.replace("decoder.layers.", "")
        layer_number, remaining_name = name.split(".", 1)
        if "self_attention" in remaining_name:
            return [
                (f"model.layers.{layer_number}.{name}", param)
                for name, param in self._convert_attention_param(
                    remaining_name, parameter, layer_number
                )
            ]
        elif "mlp" in remaining_name:
            if "mlp.gate.weight" in name or "mlp.router.weight" in name:
                name, param = self._convert_gate(name, parameter)
                return [(f"model.layers.{layer_number}.{name}", param)]
            elif "expert_bias" in name:
                name, param = self._convert_expert_bias_param(
                    name, parameter, layer_number
                )
                return [(f"model.layers.{layer_number}.{name}", param)]
            return [
                (f"model.layers.{layer_number}.{name}", param)
                for name, param in self._convert_mlp_param(
                    remaining_name, parameter, layer_number
                )
            ]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")


def get_full_tensor(weight: torch.Tensor, dim: int = 0):
    from megatron.core import parallel_state as mpu

    # TODO: support ep_tp
    train_tp_size = mpu.get_tensor_model_parallel_world_size()
    if train_tp_size != 1:
        # this is rare: bailing moe don't use tensor model parallel
        tp_group = mpu.get_tensor_model_parallel_group()
        new_v = [torch.zeros_like(weight) for i in range(train_tp_size)]
        # async_op must be False ?
        dist.all_gather(new_v, weight, group=tp_group, async_op=False)
        weight = torch.cat(new_v, dim=dim)
    return weight


def transform_mcore_qkv_weight(weight: torch.Tensor):
    """
    Megatron QKV is alternately packed, SGlangQKV is packed consecutively.
                    tp0                tp1
                               │
               ┌───────┬───┬───│───────┬───┬───┐
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
    mcore      │   Q   │ K │ V │   Q   │ K │ V │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               └───┬───┴─┬─┴─┬─│───┬───┴─┬─┴──┬┘
                   │     │   │ │   │     │    │
                   │     │   └─┴───┼─────┼┐   │
                   │     │┌────────┘     ││   │
                   │     ││              ││   │
                   │     └┼─────┐   ┌────┘│   │
                   │      │     │   │     │   │
               ┌───▼──────▼────┬▼───▼──┬──▼───▼┐
               │  Q0      Q1   │K0  K1 │  V0 V1│
               │               │       │       │
               │               │       │       │
    bailing    │       Q       │   K   │   V   │
               │               │       │       │
               │               │       │       │
               │               │       │       │
               └───────────────┴───────┴───────┘
    """
    from megatron.training import get_args

    args = get_args()
    weight = get_full_tensor(weight, dim=0)
    hidden_size = args.hidden_size
    total_num_heads = args.num_attention_heads
    total_num_kv_heads = args.num_query_groups
    head_size = divide(hidden_size, total_num_heads)

    each_kv_size = head_size
    each_query_size = head_size * divide(total_num_heads, total_num_kv_heads)
    query_list = []
    key_list = []
    value_list = []
    for qkv in torch.chunk(weight, total_num_kv_heads, dim=0):
        q, k, v = qkv.split([each_query_size, each_kv_size, each_kv_size], dim=0)
        query_list.append(q)
        key_list.append(k)
        value_list.append(v)
    # concat the query, key, value
    all_query = torch.cat(query_list, dim=0)
    all_key = torch.cat(key_list, dim=0)
    all_value = torch.cat(value_list, dim=0)
    return all_query, all_key, all_value


def convert_qkv_weight_along_tp_attention(
    weight: torch.Tensor, infer_atten_tp_size: int
):
    """
    SGlang QKV: The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """
    from megatron.training import get_args

    args = get_args()
    total_num_kv_heads = args.num_query_groups
    # Divide the weight matrix along the last dimension.
    if infer_atten_tp_size >= total_num_kv_heads:
        num_kv_head_replicas = divide(infer_atten_tp_size, total_num_kv_heads)
    else:
        num_kv_head_replicas = 1
    query, key, value = transform_mcore_qkv_weight(weight)
    query_shards = query.chunk(infer_atten_tp_size, dim=0)
    if infer_atten_tp_size >= total_num_kv_heads:
        key_chunks = key.chunk(total_num_kv_heads, dim=0)
        key_shards = [k for k in key_chunks for _ in range(num_kv_head_replicas)]
        value_chunks = value.chunk(total_num_kv_heads, dim=0)
        value_shards = [v for v in value_chunks for _ in range(num_kv_head_replicas)]
    else:
        key_shards = key.chunk(infer_atten_tp_size, dim=0)
        value_shards = value.chunk(infer_atten_tp_size, dim=0)
    qkv_tp_groups = []
    for query_shard, key_shard, value_shard in zip(
        query_shards, key_shards, value_shards
    ):
        qkv_tp_groups.append(query_shard)
        qkv_tp_groups.append(key_shard)
        qkv_tp_groups.append(value_shard)
    return torch.cat(qkv_tp_groups, dim=0)


def get_mcore_model_parameters(model) -> Dict[str, torch.Tensor]:
    params_dict = dict(model.named_parameters())
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        # there is a bug in megatron GPTModel: decoder.layers[n].mlp.router.expert_bias" in GPTModel
        # is not registered in named_parameter, but in state_dict().
        if "expert_bias" in name:
            params_dict[name] = param
    return params_dict
