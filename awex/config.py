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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class InferenceConfig:
    """
    Configuration for inference.
    """

    model_path: Optional[str] = None
    # Other runtime options
    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    # Data parallelism
    dp_size: Optional[int] = None
    load_balance_method: Optional[str] = None
    # Expert parallelism
    ep_size: Optional[int] = None
    enable_dp_attention: Optional[bool] = None
    enable_dp_lm_head: Optional[bool] = None
    deepep_mode: Optional[Literal["auto", "normal", "low_latency"]] = None
    ep_num_redundant_experts: Optional[int] = None
    enable_eplb: Optional[bool] = None
    enable_memory_saver: Optional[bool] = None
    moe_dense_tp_size: Optional[int] = None
    n_share_experts_fusion: Optional[int] = None
    nnodes: Optional[int] = None
    node_rank: Optional[int] = None

    local_rank: Optional[int] = None
    # awex specific config
    # the number of all sglang engines in the cluster
    num_engines: int = 1
    # the rank of the current engine
    engine_rank: int = 0
    # the address of the meta server: `ip:port`
    meta_server_addr: Optional[str] = None
    # weights exchange communication backend
    comm_backend: str = "file"
    # how much steps with weights validation, if enabled, weights update will use both file and transfer and
    # compare the weights
    weights_validation_steps: int = 0
    # validate weights every n steps, if enabled, weights update will use both file and transfer and compare the weights
    validate_weights_every_n_steps: int = 1
    # the list of weights to be validated
    dump_weights_list_for_validation: List[str] = field(default_factory=list)
    # the directory to dump weights for validation
    dump_weights_dir_for_validation: Optional[str] = None
    # disable the pipeline of weights exchange
    disable_weights_exchange_pipeline: bool = False
    # enable debug mode
    enable_debug_mode: bool = False
    # debug mode config, e.g. "enable_nccl_debug_mode=1"
    debug_mode_config: Dict = field(
        default_factory=dict, metadata={"help": "Debug mode configuration"}
    )
    # enable training and inference share same gpus
    enable_colocate_mode: bool = False
    # the ipc backend of weights exchange, can be "cpu" or "cuda"
    weights_exchange_ipc_backend: str = "cuda"
    weights_comm_nccl_group_size: int = 1

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "InferenceConfig":
        # remove all keys that are not fields of InferenceConfig
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k in InferenceConfig.__dataclass_fields__
        }
        return InferenceConfig(**config_dict)

    @staticmethod
    def from_sgl_engine(sgl_engine, **extra_config) -> "InferenceConfig":
        return InferenceConfig.from_sgl_server_args(
            sgl_engine.server_args, **extra_config
        )

    @staticmethod
    def from_sgl_server_args(server_args, **extra_config) -> "InferenceConfig":
        config = {}
        for k in InferenceConfig.__dataclass_fields__:
            value = getattr(server_args, k, None)
            if value is not None:
                config[k] = value
        config.update(**extra_config)
        return InferenceConfig(**config)
