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


def _dump_vllm_weights(args: argparse.Namespace) -> None:
    import tempfile

    import torch
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.model_executor.model_loader import get_model

    from awex.util import device as device_util

    out_dir = Path(args.out_dir).resolve()
    weights_dir = out_dir / "vllm_tensors"
    weights_dir.mkdir(parents=True, exist_ok=True)

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

    model = get_model(vllm_config=vllm_config)
    entries: list[dict] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _should_include_name(name, args.max_layers, args.include_non_layer):
                continue
            tensor = param.detach().cpu()
            file_name = f"{_hash_name(name)}.pt"
            file_path = weights_dir / file_name
            torch.save(tensor, file_path)
            entries.append(
                {
                    "name": name,
                    "file": str(file_path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
            )

    _save_manifest(out_dir / "vllm_manifest.json", entries)

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


def _compare_with_sglang(args: argparse.Namespace) -> None:
    import tempfile

    import torch
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.model_loader import get_model
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
        set_global_server_args_for_tokenizer,
    )

    from awex.util import device as device_util

    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "vllm_manifest.json"
    manifest = _load_manifest(manifest_path)
    seen_vllm_names: set[str] = set()

    sglang_model_config = ModelConfig(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )
    sglang_load_config = LoadConfig(
        load_format=args.sglang_load_format,
        download_dir=args.download_dir,
    )
    sglang_device_config = DeviceConfig(device=args.device, gpu_id=args.gpu_id)

    server_args = ServerArgs(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device=args.device,
        base_gpu_id=args.gpu_id,
    )
    set_global_server_args_for_scheduler(server_args)
    set_global_server_args_for_tokenizer(server_args)

    if not model_parallel_is_initialized():
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
        elif args.device == "npu":
            npu_mod = getattr(torch, "npu", None)
            if npu_mod is None:
                raise RuntimeError("torch.npu is not available for NPU backend.")
            npu_mod.set_device(args.gpu_id)
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

    initialize_dp_attention(server_args, sglang_model_config)

    model = get_model(
        model_config=sglang_model_config,
        load_config=sglang_load_config,
        device_config=sglang_device_config,
    )

    missing_in_vllm: list[str] = []
    missing_in_sglang: list[str] = []
    shape_mismatch: list[dict] = []
    dtype_mismatch: list[dict] = []
    value_mismatch: list[dict] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _should_include_name(name, args.max_layers, args.include_non_layer):
                continue
            entry = manifest.get(name)
            if entry is None:
                missing_in_vllm.append(name)
                continue

            seen_vllm_names.add(name)
            vllm_tensor = torch.load(entry["file"], map_location="cpu")
            sglang_tensor = param.detach().cpu()

            if list(vllm_tensor.shape) != list(sglang_tensor.shape):
                shape_mismatch.append(
                    {
                        "name": name,
                        "vllm_shape": list(vllm_tensor.shape),
                        "sglang_shape": list(sglang_tensor.shape),
                    }
                )
                continue

            if str(vllm_tensor.dtype) != str(sglang_tensor.dtype):
                dtype_mismatch.append(
                    {
                        "name": name,
                        "vllm_dtype": str(vllm_tensor.dtype),
                        "sglang_dtype": str(sglang_tensor.dtype),
                    }
                )
                if args.compare_values_strict:
                    vllm_tensor = vllm_tensor.to(sglang_tensor.dtype)
                else:
                    continue

            if not torch.allclose(
                vllm_tensor,
                sglang_tensor,
                rtol=args.rtol,
                atol=args.atol,
            ):
                diff = (vllm_tensor - sglang_tensor).abs()
                value_mismatch.append(
                    {
                        "name": name,
                        "max_abs_diff": diff.max().item(),
                        "mean_abs_diff": diff.mean().item(),
                    }
                )

    for name in manifest.keys():
        if name not in seen_vllm_names:
            missing_in_sglang.append(name)

    report = {
        "missing_in_vllm": missing_in_vllm,
        "missing_in_sglang": missing_in_sglang,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "value_mismatch": value_mismatch,
    }
    report_path = out_dir / "compare_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Comparison complete.")
    print(f"Missing in vLLM: {len(missing_in_vllm)}")
    print(f"Missing in SGLang: {len(missing_in_sglang)}")
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
        "--device",
        args.device,
        "--gpu-id",
        str(args.gpu_id),
        "--dtype",
        args.dtype,
        "--vllm-load-format",
        args.vllm_load_format,
        "--sglang-load-format",
        args.sglang_load_format,
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--device-backend",
        args.device_backend,
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
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
        description="Compare vLLM and SGLang loaded weights for the same HF model."
    )
    parser.add_argument("--model-path", required=True, help="HF model path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device for SGLang")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for SGLang")
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
        "--sglang-load-format",
        default="auto",
        help="SGLang load format (auto, safetensors, pt, etc)",
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
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--compare-values-strict",
        action="store_true",
        help="Compare values even if dtype mismatches",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "vllm_dump", "sglang_compare"],
        default="all",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )

    args = parser.parse_args()
    device_backend = _apply_device_backend(args.device_backend)
    if args.device == "cuda" and device_backend == "npu":
        args.device = "npu"

    if args.stage == "vllm_dump":
        _dump_vllm_weights(args)
        return
    if args.stage == "sglang_compare":
        _compare_with_sglang(args)
        return

    _run_subprocess("vllm_dump", args)
    _run_subprocess("sglang_compare", args)


if __name__ == "__main__":
    main()
