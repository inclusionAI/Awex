#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


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


def _parse_devices(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    return [int(x) for x in arg.split(",") if x.strip()]


def _maybe_get_tf_config(model):
    for attr in ("transformer_config", "config"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            return cfg
    return None


def _setup_distributed():
    import torch.distributed as dist

    from awex.util import device as device_util

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = _dist_backend_for(device_util.get_device_type())
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def _pick_train_devices(
    world_size: int, visible_devices: List[int], train_devices_arg: Optional[str]
) -> List[int]:
    train_devices = _parse_devices(train_devices_arg)
    if not train_devices:
        if len(visible_devices) < world_size:
            raise RuntimeError(
                f"Need at least {world_size} visible devices for training, "
                f"but only see {len(visible_devices)}."
            )
        train_devices = visible_devices[:world_size]
    if len(train_devices) != world_size:
        raise RuntimeError(
            f"train-devices length ({len(train_devices)}) must equal "
            f"WORLD_SIZE ({world_size})."
        )
    for device_id in train_devices:
        if device_id not in visible_devices:
            raise RuntimeError(
                f"Training device id {device_id} is not visible in this process. "
                "If you need physical IDs outside visible devices env, launch without "
                "restricting CUDA_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES."
            )
    return train_devices


def _dump_megatron_hf_weights_multi(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    from awex.converter.mcore_converter import McoreToHFWeightConverter
    from awex.sharding.mcore_sharding import get_mcore_rank_info
    from awex.tests.test_utils import megatron_model_from_hf
    from awex.util import device as device_util

    rank, local_rank, world_size = _setup_distributed()
    visible_devices = list(range(device_util.device_count()))
    train_devices = _pick_train_devices(
        world_size, visible_devices, args.train_cuda_devices
    )
    device_util.set_device(train_devices[local_rank])

    expert_tp_size = (
        args.train_expert_tp_size
        if args.train_expert_tp_size is not None
        else args.train_tp_size
    )
    dense_parallel = args.train_tp_size * args.train_pp_size
    expert_parallel = expert_tp_size * args.train_ep_size * args.train_pp_size
    required_world = math.lcm(dense_parallel, expert_parallel)
    if world_size % dense_parallel != 0 or world_size % expert_parallel != 0:
        raise RuntimeError(
            "Invalid train parallel config for WORLD_SIZE. "
            f"dense(tp*pp)={dense_parallel}, "
            f"expert(expert_tp*ep*pp)={expert_parallel}, "
            f"WORLD_SIZE={world_size}, required multiple={required_world}. "
            "Try adjusting --train-expert-tp-size / --train-ep-size / --train-tp-size."
        )

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.train_tp_size,
        pipeline_model_parallel_size=args.train_pp_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=args.train_ep_size,
        expert_tensor_parallel_size=expert_tp_size,
    )
    model_parallel_cuda_manual_seed(0)

    models, hf_config = megatron_model_from_hf(
        model_path=args.model_path,
        use_mbridge=not args.no_mbridge,
    )

    rank_info = get_mcore_rank_info()
    infer_conf = {
        "infer_atten_tp_size": rank_info.attn_tp_size,
        "num_query_groups": getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
    }

    out_dir = Path(args.out_dir).resolve()
    rank_dir = out_dir / f"megatron_rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

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
        file_path = rank_dir / file_name
        torch.save(tensor, file_path)
        entries.append(
            {
                "name": hf_name,
                "file": str(file_path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "rank": rank,
            }
        )

    _save_manifest(out_dir / f"megatron_rank_{rank}_manifest.json", entries)
    if duplicate_conflicts:
        conflict_path = out_dir / f"megatron_rank_{rank}_conflicts.json"
        with conflict_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(set(duplicate_conflicts)), f, indent=2)

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged: dict[str, dict] = {}
        merge_conflicts: list[dict] = []
        for r in range(world_size):
            manifest_path = out_dir / f"megatron_rank_{r}_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(f"Missing manifest for rank {r}: {manifest_path}")
            for name, entry in _load_manifest(manifest_path).items():
                existing = merged.get(name)
                if existing is None:
                    merged[name] = entry
                    continue
                if (
                    existing["shape"] != entry["shape"]
                    or existing["dtype"] != entry["dtype"]
                ):
                    merge_conflicts.append(
                        {
                            "name": name,
                            "first_rank": existing.get("rank"),
                            "second_rank": entry.get("rank"),
                            "first_shape": existing["shape"],
                            "second_shape": entry["shape"],
                            "first_dtype": existing["dtype"],
                            "second_dtype": entry["dtype"],
                        }
                    )
        merged_entries = list(merged.values())
        _save_manifest(out_dir / "megatron_hf_manifest.json", merged_entries)
        if merge_conflicts:
            conflict_path = out_dir / "megatron_merge_conflicts.json"
            with conflict_path.open("w", encoding="utf-8") as f:
                json.dump(merge_conflicts, f, indent=2)


def _compare_with_vllm(args: argparse.Namespace) -> None:
    import tempfile

    import torch
    from transformers import AutoConfig
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import (
        get_dp_group,
        get_ep_group,
        get_pp_group,
        get_tp_group,
        model_parallel_is_initialized,
    )
    from vllm.model_executor.model_loader import get_model

    from awex.config import InferenceConfig
    from awex.converter.vllm_converter import VLLMToHFWeightConverter
    from awex.sharding.rank_info import RankInfo
    from awex.util import device as device_util

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        world_size = 1

    out_dir = Path(args.out_dir).resolve()
    rank_manifest_path = out_dir / f"megatron_rank_{rank}_manifest.json"
    merged_manifest_path = out_dir / "megatron_hf_manifest.json"
    if rank_manifest_path.exists():
        manifest_path = rank_manifest_path
    elif merged_manifest_path.exists():
        manifest_path = merged_manifest_path
    else:
        raise RuntimeError(
            f"No Megatron manifest found for rank {rank}. Checked: "
            f"{rank_manifest_path} and {merged_manifest_path}"
        )
    manifest = _load_manifest(manifest_path)

    hf_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    tp_size = (
        args.vllm_tp_size
        if args.vllm_tp_size is not None
        else (args.train_tp_size if world_size > 1 else 1)
    )
    pp_size = (
        args.vllm_pp_size
        if args.vllm_pp_size is not None
        else (args.train_pp_size if world_size > 1 else 1)
    )
    dp_size = args.vllm_data_parallel_size
    expected_world_size = tp_size * pp_size * dp_size
    if world_size > 1 and world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE ({world_size}) must match vLLM tp*pp*dp "
            f"({tp_size} * {pp_size} * {dp_size} = {expected_world_size})."
        )

    if not model_parallel_is_initialized():
        if world_size > 1:
            init_file = out_dir / "vllm_init"
            if rank == 0:
                init_file.touch(exist_ok=True)
            temp_file = str(init_file)
        else:
            temp_file = tempfile.mkstemp()[1]
        backend = _dist_backend_for(device_util.get_device_type())
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=local_rank,
            backend=backend,
        )
        initialize_model_parallel(tp_size, pp_size)

    tp_group = get_tp_group()
    pp_group = get_pp_group()
    dp_group = get_dp_group()
    ep_group = get_ep_group() if args.vllm_enable_expert_parallel else None
    ep_size = ep_group.world_size if ep_group is not None else 1
    ep_rank = ep_group.rank_in_group if ep_group is not None else 0

    infer_config = InferenceConfig(tp_size=tp_size, ep_size=ep_size)
    rank_info = RankInfo(
        tp_rank=tp_group.rank_in_group,
        tp_size=tp_group.world_size,
        pp_rank=pp_group.rank_in_group,
        pp_size=pp_group.world_size,
        dp_size=dp_group.world_size,
        dp_rank=dp_group.rank_in_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=tp_group.rank_in_group,
        attn_tp_size=tp_group.world_size,
        attn_dp_rank=0,
        world_size=world_size,
        global_rank=rank,
        local_rank=local_rank,
        engine_rank=rank,
        is_infer=True,
    )
    converter = VLLMToHFWeightConverter(hf_config, infer_config, rank_info)

    if device_util.device_count() > 0:
        try:
            device_util.set_device(local_rank)
        except Exception:
            pass

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            enforce_eager=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            enable_expert_parallel=args.vllm_enable_expert_parallel,
        ),
        load_config=LoadConfig(
            load_format=args.vllm_load_format,
            download_dir=args.download_dir,
        ),
    )

    model = get_model(vllm_config=vllm_config)

    if os.environ.get("AWEX_DEBUG_VLLM_PARAMS", "0") == "1":
        raw_names = [name for name, _ in model.named_parameters()]
        has_lm_head = (
            "lm_head.weight" in raw_names or "model.lm_head.weight" in raw_names
        )
        has_embed = (
            "model.embed_tokens.weight" in raw_names
            or "embed_tokens.weight" in raw_names
        )
        print(
            "[vLLM params] "
            f"pp_rank={rank_info.pp_rank}/{rank_info.pp_size} "
            f"tp_rank={rank_info.tp_rank}/{rank_info.tp_size} "
            f"has_lm_head={has_lm_head} has_embed={has_embed} total_params={len(raw_names)}"
        )

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

    if (
        getattr(hf_config, "tie_word_embeddings", False)
        and rank_info.pp_rank == rank_info.pp_size - 1
        and "lm_head.weight" not in vllm_hf_weights
        and "model.embed_tokens.weight" in vllm_hf_weights
    ):
        vllm_hf_weights["lm_head.weight"] = vllm_hf_weights["model.embed_tokens.weight"]

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

    if rank_info.pp_size > 1:
        # In pipeline parallelism, embeddings are typically on pp_rank 0 and lm_head on last rank.
        if (
            rank_info.pp_rank != 0
            and "model.embed_tokens.weight" in missing_in_megatron
        ):
            missing_in_megatron = [
                n for n in missing_in_megatron if n != "model.embed_tokens.weight"
            ]
        if (
            rank_info.pp_rank != rank_info.pp_size - 1
            and "lm_head.weight" in missing_in_megatron
        ):
            missing_in_megatron = [
                n for n in missing_in_megatron if n != "lm_head.weight"
            ]

    report = {
        "missing_in_megatron": missing_in_megatron,
        "missing_in_vllm": missing_in_vllm,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "value_mismatch": value_mismatch,
        "vllm_duplicate_conflicts": sorted(set(vllm_conflicts)),
    }
    if world_size > 1:
        report_path = out_dir / f"megatron_vllm_compare_report_rank_{rank}.json"
    else:
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

    if world_size > 1:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        if rank == 0:
            summary = {
                "ranks": world_size,
                "reports": [],
                "missing_in_megatron": 0,
                "missing_in_vllm": 0,
                "shape_mismatch": 0,
                "dtype_mismatch": 0,
                "value_mismatch": 0,
                "vllm_duplicate_conflicts": 0,
            }
            for r in range(world_size):
                per_rank_path = out_dir / f"megatron_vllm_compare_report_rank_{r}.json"
                if not per_rank_path.exists():
                    continue
                with per_rank_path.open("r", encoding="utf-8") as f:
                    per_rank = json.load(f)
                summary["reports"].append(str(per_rank_path))
                summary["missing_in_megatron"] += len(per_rank["missing_in_megatron"])
                summary["missing_in_vllm"] += len(per_rank["missing_in_vllm"])
                summary["shape_mismatch"] += len(per_rank["shape_mismatch"])
                summary["dtype_mismatch"] += len(per_rank["dtype_mismatch"])
                summary["value_mismatch"] += len(per_rank["value_mismatch"])
                summary["vllm_duplicate_conflicts"] += len(
                    per_rank["vllm_duplicate_conflicts"]
                )
            summary_path = out_dir / "megatron_vllm_compare_report.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"Merged report: {summary_path}")

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
    if args.vllm_tp_size is not None:
        cmd.extend(["--vllm-tp-size", str(args.vllm_tp_size)])
    if args.vllm_pp_size is not None:
        cmd.extend(["--vllm-pp-size", str(args.vllm_pp_size)])
    cmd.extend(["--vllm-data-parallel-size", str(args.vllm_data_parallel_size)])
    if args.vllm_enable_expert_parallel:
        cmd.append("--vllm-enable-expert-parallel")
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
            "Compare Megatron->HF converted weights against vLLM->HF converted weights "
            "using torchrun to dump Megatron weights in multi-GPU mode."
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
    parser.add_argument("--train-tp-size", type=int, default=1, help="Megatron TP size")
    parser.add_argument("--train-pp-size", type=int, default=1, help="Megatron PP size")
    parser.add_argument("--train-ep-size", type=int, default=1, help="Megatron EP size")
    parser.add_argument(
        "--train-expert-tp-size",
        type=int,
        default=None,
        help=(
            "Megatron expert tensor-parallel size. Default: same as --train-tp-size."
        ),
    )
    parser.add_argument(
        "--train-cuda-devices",
        default="",
        help="Comma-separated list of train GPU ids (visible indices). "
        "Length must equal WORLD_SIZE.",
    )
    parser.add_argument(
        "--vllm-tp-size",
        type=int,
        default=None,
        help="vLLM TP size for compare stage (default: train_tp_size when WORLD_SIZE>1).",
    )
    parser.add_argument(
        "--vllm-pp-size",
        type=int,
        default=None,
        help="vLLM PP size for compare stage (default: train_pp_size when WORLD_SIZE>1).",
    )
    parser.add_argument(
        "--vllm-data-parallel-size",
        type=int,
        default=1,
        help="vLLM DP size for compare stage.",
    )
    parser.add_argument(
        "--vllm-enable-expert-parallel",
        action="store_true",
        help="Enable vLLM expert parallel mode when loading MoE weights.",
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
        _dump_megatron_hf_weights_multi(args)
        return
    if args.stage == "vllm_compare":
        _compare_with_vllm(args)
        return

    # If run under torchrun, do both stages on all ranks.
    # vLLM compare needs all ranks participating when WORLD_SIZE > 1.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _dump_megatron_hf_weights_multi(args)
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        _compare_with_vllm(args)
        return

    raise RuntimeError(
        "Stage 'all' must be launched with torchrun. "
        "Use:\n"
        "  torchrun --nproc_per_node=N compare_megatron_vllm_weights_multi.py --stage megatron_dump ...\n"
        "then:\n"
        "  torchrun --nproc_per_node=M compare_megatron_vllm_weights_multi.py --stage vllm_compare ..."
    )


if __name__ == "__main__":
    main()
