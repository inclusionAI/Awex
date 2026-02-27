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

"""Helpers for loading HF weights into Megatron modules via mbridge mappings."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Callable, Iterable

import torch

TensorReader = Callable[[str, str], torch.Tensor]


def _default_tensor_reader(weights_path: str, name: str) -> torch.Tensor:
    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(
            "safetensors is required for loading HF weights without AReaL. "
            "Install with: pip install safetensors"
        ) from e

    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        filename = os.path.join(weights_path, weight_map[name])
        with safe_open(filename, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    safetensor_files = glob(os.path.join(weights_path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found under {weights_path}.")
    for path in safetensor_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            if name in f.keys():
                return f.get_tensor(name)
    raise KeyError(f"Weight {name} not found in safetensors under {weights_path}.")


def _coerce_names(names: Iterable[str] | str) -> list[str]:
    if isinstance(names, str):
        return [names]
    return list(names)


def _concat_along_matching_dim(
    tensors: list[torch.Tensor], target_shape: torch.Size
) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    for dim in range(len(target_shape)):
        if all(t.ndim == len(target_shape) for t in tensors):
            other_dims_match = all(
                all(
                    t.shape[d] == tensors[0].shape[d]
                    for d in range(len(target_shape))
                    if d != dim
                )
                for t in tensors
            )
            if not other_dims_match:
                continue
            if sum(t.shape[dim] for t in tensors) == target_shape[dim]:
                return torch.cat(tensors, dim=dim)
    raise ValueError(
        f"Cannot concatenate tensors to match target shape {tuple(target_shape)}."
    )


def load_weights_from_hf_with_mbridge(
    bridge,
    models: list[torch.nn.Module],
    weights_path: str,
    reader: TensorReader | None = None,
    is_critic: bool = False,
) -> None:
    if reader is None:
        reader = _default_tensor_reader

    if hasattr(bridge, "_get_actual_hf_path"):
        weights_path = bridge._get_actual_hf_path(weights_path)

    for model in models:
        state_dict = model.state_dict()
        if hasattr(bridge, "_weight_name_mapping_mcore_local_to_global"):
            local_to_global = bridge._weight_name_mapping_mcore_local_to_global(model)
        else:
            local_to_global = {name: name for name in state_dict}

        for local_name, global_name in local_to_global.items():
            if is_critic and "output_layer" in local_name:
                continue
            if local_name not in state_dict:
                continue
            if hasattr(bridge, "_weight_name_mapping_mcore_to_hf"):
                hf_names = bridge._weight_name_mapping_mcore_to_hf(global_name)
            else:
                hf_names = [global_name]
            hf_names = _coerce_names(hf_names)

            loaded_tensors = [reader(weights_path, name) for name in hf_names]
            target_param = state_dict[local_name]
            merged = _concat_along_matching_dim(loaded_tensors, target_param.shape)
            target_param.copy_(
                merged.to(device=target_param.device, dtype=target_param.dtype)
            )
