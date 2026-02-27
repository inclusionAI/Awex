#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _apply_device_backend(device_backend: str) -> str:
    from awex.util import device as device_util

    if device_backend and device_backend != "auto":
        os.environ["AWEX_DEVICE_TYPE"] = device_backend
    return device_util.get_device_type()


def _dist_backend_for(device_type: str) -> str:
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def _hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


_LAYER_ID_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.")


def _layer_id_from_name(name: str) -> int | None:
    match = _LAYER_ID_RE.search(name)
    if not match:
        return None
    return int(match.group(1))


def _should_include_name(
    name: str, max_layers: int | None, include_non_layer: bool
) -> bool:
    if max_layers is None:
        return True
    layer_id = _layer_id_from_name(name)
    if layer_id is None:
        return include_non_layer
    return layer_id < max_layers


def _save_manifest(manifest_path: Path, entries: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["name"]: entry for entry in entries}


def _maybe_get_tf_config(model):
    for attr in ("transformer_config", "config"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            return cfg
    return None


def _dump_megatron_hf_weights(args: argparse.Namespace) -> None:
    import torch

    from awex.converter.mcore_converter import McoreToHFWeightConverter
    from awex.sharding.rank_info import RankInfo
    from awex.tests.test_utils import megatron_model_from_hf
    from awex.util import device as device_util

    out_dir = Path(args.out_dir).resolve()
    weights_dir = out_dir / "megatron_hf_tensors"
    weights_dir.mkdir(parents=True, exist_ok=True)

    models, hf_config = megatron_model_from_hf(
        model_path=args.model_path,
        use_mbridge=not args.no_mbridge,
    )

    infer_conf = {
        "infer_atten_tp_size": 1,
        "num_query_groups": getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
    }

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

    entries: list[dict] = []
    name_to_tensor: dict[str, torch.Tensor] = {}
    duplicate_conflicts: list[str] = []

    with torch.no_grad():
        for model in models:
            tf_config = _maybe_get_tf_config(model)
            converter = McoreToHFWeightConverter(
                hf_config,
                rank_info,
                infer_conf=infer_conf,
                tf_config=tf_config,
            )
            for name, param in model.named_parameters():
                converted = converter.convert_param(name, param.detach())
                for hf_name, hf_tensor in converted:
                    if not _should_include_name(
                        hf_name, args.max_layers, args.include_non_layer
                    ):
                        continue
                    existing = name_to_tensor.get(hf_name)
                    if existing is not None:
                        if existing.shape != hf_tensor.shape or not torch.allclose(
                            existing, hf_tensor
                        ):
                            duplicate_conflicts.append(hf_name)
                        continue
                    name_to_tensor[hf_name] = hf_tensor.detach().cpu()

    for hf_name, tensor in name_to_tensor.items():
        file_name = f"{_hash_name(hf_name)}.pt"
        file_path = weights_dir / file_name
        torch.save(tensor, file_path)
        entries.append(
            {
                "name": hf_name,
                "file": str(file_path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
        )

    _save_manifest(out_dir / "megatron_hf_manifest.json", entries)

    if duplicate_conflicts:
        conflict_path = out_dir / "megatron_duplicate_conflicts.json"
        with conflict_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(set(duplicate_conflicts)), f, indent=2)

    if device_util.get_device_type() == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_util.get_device_type() == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "empty_cache"):
            npu_mod.empty_cache()
    try:
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _compare_with_vllm(args: argparse.Namespace) -> None:
    import tempfile

    import torch
    from transformers import AutoConfig
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.model_executor.model_loader import get_model

    from awex.config import InferenceConfig
    from awex.converter.vllm_converter import VLLMToHFWeightConverter
    from awex.sharding.rank_info import RankInfo
    from awex.util import device as device_util

    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "megatron_hf_manifest.json"
    manifest = _load_manifest(manifest_path)

    hf_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
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
    converter = VLLMToHFWeightConverter(hf_config, infer_config, rank_info)

    if not model_parallel_is_initialized():
        temp_file = tempfile.mkstemp()[1]
        backend = _dist_backend_for(device_util.get_device_type())
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend=backend,
        )
        initialize_model_parallel(1, 1)

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            enforce_eager=True,
        ),
        load_config=LoadConfig(
            load_format=args.vllm_load_format,
            download_dir=args.download_dir,
        ),
    )

    model = get_model(vllm_config=vllm_config)

    vllm_hf_weights: dict[str, torch.Tensor] = {}
    vllm_conflicts: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            for hf_name, hf_tensor in converter.convert_param(name, param.detach()):
                if not _should_include_name(
                    hf_name, args.max_layers, args.include_non_layer
                ):
                    continue
                existing = vllm_hf_weights.get(hf_name)
                if existing is not None:
                    if existing.shape != hf_tensor.shape or not torch.allclose(
                        existing, hf_tensor
                    ):
                        vllm_conflicts.append(hf_name)
                    continue
                vllm_hf_weights[hf_name] = hf_tensor.detach().cpu()

    missing_in_megatron: list[str] = []
    missing_in_vllm: list[str] = []
    shape_mismatch: list[dict] = []
    dtype_mismatch: list[dict] = []
    value_mismatch: list[dict] = []

    for name, vllm_tensor in vllm_hf_weights.items():
        entry = manifest.get(name)
        if entry is None:
            missing_in_megatron.append(name)
            continue
        megatron_tensor = torch.load(entry["file"], map_location="cpu")
        if list(megatron_tensor.shape) != list(vllm_tensor.shape):
            shape_mismatch.append(
                {
                    "name": name,
                    "megatron_shape": list(megatron_tensor.shape),
                    "vllm_shape": list(vllm_tensor.shape),
                }
            )
            continue
        if str(megatron_tensor.dtype) != str(vllm_tensor.dtype):
            dtype_mismatch.append(
                {
                    "name": name,
                    "megatron_dtype": str(megatron_tensor.dtype),
                    "vllm_dtype": str(vllm_tensor.dtype),
                }
            )
            if args.compare_values_strict:
                vllm_tensor = vllm_tensor.to(megatron_tensor.dtype)
            else:
                continue
        if not torch.allclose(
            megatron_tensor,
            vllm_tensor,
            rtol=args.rtol,
            atol=args.atol,
        ):
            diff = (megatron_tensor - vllm_tensor).abs()
            value_mismatch.append(
                {
                    "name": name,
                    "max_abs_diff": diff.max().item(),
                    "mean_abs_diff": diff.mean().item(),
                }
            )

    for name in manifest.keys():
        if name not in vllm_hf_weights:
            missing_in_vllm.append(name)

    report = {
        "missing_in_megatron": missing_in_megatron,
        "missing_in_vllm": missing_in_vllm,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "value_mismatch": value_mismatch,
        "vllm_duplicate_conflicts": sorted(set(vllm_conflicts)),
    }
    report_path = out_dir / "megatron_vllm_compare_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Comparison complete.")
    print(f"Missing in Megatron: {len(missing_in_megatron)}")
    print(f"Missing in vLLM: {len(missing_in_vllm)}")
    print(f"Shape mismatch: {len(shape_mismatch)}")
    print(f"Dtype mismatch: {len(dtype_mismatch)}")
    print(f"Value mismatch: {len(value_mismatch)}")
    print(f"Report: {report_path}")

    del model
    if device_util.get_device_type() == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_util.get_device_type() == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "empty_cache"):
            npu_mod.empty_cache()
    try:
        cleanup_dist_env_and_memory()
    except Exception:
        pass


def _run_subprocess(stage: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--model-path",
        args.model_path,
        "--out-dir",
        args.out_dir,
        "--dtype",
        args.dtype,
        "--vllm-load-format",
        args.vllm_load_format,
        "--device-backend",
        args.device_backend,
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.no_mbridge:
        cmd.append("--no-mbridge")
    if args.compare_values_strict:
        cmd.append("--compare-values-strict")
    if args.max_layers is not None:
        cmd.extend(["--max-layers", str(args.max_layers)])
    if args.include_non_layer:
        cmd.append("--include-non-layer")
    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])

    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Megatron->HF converted weights against vLLM->HF converted weights."
        )
    )
    parser.add_argument("--model-path", required=True, help="HF model path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--dtype", default="auto", help="Weight dtype")
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Only compare layers [0, max_layers).",
    )
    parser.add_argument(
        "--include-non-layer",
        action="store_true",
        help="Include non-layer weights when max-layers is set.",
    )
    parser.add_argument(
        "--vllm-load-format",
        default="auto",
        help="vLLM load format (auto, safetensors, pt, etc)",
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional HF download cache dir",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HF models",
    )
    parser.add_argument(
        "--no-mbridge",
        action="store_true",
        help="Use Megatron DCP conversion instead of mbridge loading",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--compare-values-strict",
        action="store_true",
        help="Compare values even if dtype mismatches",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "megatron_dump", "vllm_compare"],
        default="all",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    _apply_device_backend(args.device_backend)

    if args.stage == "megatron_dump":
        _dump_megatron_hf_weights(args)
        return
    if args.stage == "vllm_compare":
        _compare_with_vllm(args)
        return

    _run_subprocess("megatron_dump", args)
    _run_subprocess("vllm_compare", args)


if __name__ == "__main__":
    main()
