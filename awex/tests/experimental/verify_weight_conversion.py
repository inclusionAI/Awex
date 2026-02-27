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

import argparse
import json
import os
import re
from glob import glob

import torch
from transformers import AutoConfig

from awex.config import InferenceConfig
from awex.converter.vllm_converter import VLLMToHFWeightConverter
from awex.sharding.rank_info import RankInfo
from awex.util import device as device_util


def _load_hf_weight_index(model_path: str) -> set[str]:
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as f:
            return set(json.load(f)["weight_map"].keys())
    safetensor_files = glob(os.path.join(model_path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors found under {model_path}.")
    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(
            "safetensors is required to inspect HF weights. Install with: pip install safetensors"
        ) from e
    keys = set()
    for path in safetensor_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys.update(f.keys())
    return keys


def _make_converter(hf_config):
    infer_config = InferenceConfig(tp_size=1, ep_size=1)
    rank_info = RankInfo(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=1,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=True,
    )
    return VLLMToHFWeightConverter(hf_config, infer_config, rank_info)


def _candidate_vllm_names(model_type: str) -> list[str]:
    if model_type in {"glm4v", "glm4v_moe"}:
        return [
            "model.layers.0.self_attn.qkv.weight",
            "model.layers.0.self_attn.proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
        ]
    return [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    ]


def _qwen3_alt_hf_keys(layer_prefix: str, missing_key: str) -> list[str]:
    if "query_key_value_proj" in missing_key:
        return [
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",
        ]
    if "attention.dense" in missing_key:
        return [f"{layer_prefix}.self_attn.o_proj.weight"]
    return []


def verify_conversion(model_path: str) -> int:
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_keys = _load_hf_weight_index(model_path)
    converter = _make_converter(hf_config)
    model_type = getattr(hf_config, "model_type", "unknown")

    failures = 0
    for vllm_name in _candidate_vllm_names(model_type):
        dummy = torch.zeros((4, 4), device=device_util.get_torch_device())
        try:
            converted = converter.convert_param(vllm_name, dummy)
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {vllm_name}: converter failed: {exc}")
            continue
        missing = [name for name, _ in converted if name not in hf_keys]
        if missing:
            if str(model_type).startswith("qwen3"):
                layer_prefix = "model.layers.0"
                for name in missing:
                    match = re.search(r"(model\\.layers\\.\\d+)", name)
                    if match:
                        layer_prefix = match.group(1)
                    alt_keys = _qwen3_alt_hf_keys(layer_prefix, name)
                    if alt_keys and all(key in hf_keys for key in alt_keys):
                        print(
                            f"[OK] {vllm_name} -> HF uses q/k/v (or o_proj) keys: {alt_keys}"
                        )
                    else:
                        failures += 1
                        print(f"[MISMATCH] {vllm_name} -> missing HF keys: {missing}")
                        break
            else:
                failures += 1
                print(f"[MISMATCH] {vllm_name} -> missing HF keys: {missing}")
        else:
            print(f"[OK] {vllm_name} -> {[name for name, _ in converted]}")

    if failures:
        print(f"Found {failures} conversion issues for {model_type}.")
    else:
        print(f"All checks passed for {model_type}.")
    return failures


def main(args):
    failures = verify_conversion(args.model_path)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify vLLM -> HF conversion mapping")
    parser.add_argument(
        "--model-path", required=True, help="HF model path or cache dir"
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )
    args = parser.parse_args()
    if args.device_backend and args.device_backend != "auto":
        os.environ["AWEX_DEVICE_TYPE"] = args.device_backend
    main(args)
