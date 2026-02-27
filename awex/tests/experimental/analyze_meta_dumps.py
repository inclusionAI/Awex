#!/usr/bin/env python3
"""Analyze AWEX train/infer parameter meta dumps.

This script is intended for large-model debugging where manually scanning
`train_params_meta_*.json` / `infer_params_meta_*.json` is too expensive.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any


def _newest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_json_list(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")
    return data


def _allowed_infer_only_alias(
    extra_key: str, infer_keys: set[str], train_keys: set[str]
) -> bool:
    # Keep behavior aligned with awex.util.common._is_allowed_infer_only_alias.
    if extra_key == "lm_head.weight":
        return (
            "model.embed_tokens.weight" in infer_keys
            and "model.embed_tokens.weight" in train_keys
        )
    if extra_key == "model.embed_tokens.weight":
        return "lm_head.weight" in infer_keys and "lm_head.weight" in train_keys
    return False


def _tp_size(param_meta: dict[str, Any]) -> int:
    replicas = param_meta.get("replicas") or []
    if not replicas:
        return 0
    shards = replicas[0].get("shards") or []
    return len(shards)


def _sharding_types(param_meta: dict[str, Any]) -> list[str]:
    out: set[str] = set()
    for replica in param_meta.get("replicas") or []:
        for shard in replica.get("shards") or []:
            sharding_type = shard.get("sharding_type")
            if isinstance(sharding_type, str):
                out.add(sharding_type)
    return sorted(out)


def analyze_meta(
    train_meta: list[dict[str, Any]],
    infer_meta: list[dict[str, Any]],
    strict_key_match: bool,
) -> dict[str, Any]:
    train = {item["name"]: item for item in train_meta}
    infer = {item["name"]: item for item in infer_meta}
    train_keys = set(train.keys())
    infer_keys = set(infer.keys())
    common = sorted(train_keys & infer_keys)

    missing_on_infer = sorted(train_keys - infer_keys)
    extra_on_infer = sorted(infer_keys - train_keys)
    unsupported_extra = sorted(
        key
        for key in extra_on_infer
        if not _allowed_infer_only_alias(key, infer_keys, train_keys)
    )

    shape_mismatch: list[tuple[str, Any, Any]] = []
    dtype_mismatch: list[tuple[str, Any, Any]] = []
    numel_mismatch: list[tuple[str, Any, Any]] = []
    tp_rule_violations: list[tuple[str, int, int]] = []
    sharding_type_diff: list[tuple[str, list[str], list[str]]] = []

    for name in common:
        t = train[name]
        i = infer[name]
        if t.get("global_shape") != i.get("global_shape"):
            shape_mismatch.append((name, t.get("global_shape"), i.get("global_shape")))
        if t.get("dtype") != i.get("dtype"):
            dtype_mismatch.append((name, t.get("dtype"), i.get("dtype")))
        if t.get("global_numel") != i.get("global_numel"):
            numel_mismatch.append((name, t.get("global_numel"), i.get("global_numel")))

        train_tp = _tp_size(t)
        infer_tp = _tp_size(i)
        if train_tp > 0 and (infer_tp < train_tp or infer_tp % train_tp != 0):
            tp_rule_violations.append((name, train_tp, infer_tp))

        ts = _sharding_types(t)
        is_ = _sharding_types(i)
        if ts != is_:
            sharding_type_diff.append((name, ts, is_))

    key_errors = bool(
        missing_on_infer or (extra_on_infer if strict_key_match else unsupported_extra)
    )
    critical_errors = (
        int(key_errors)
        + len(shape_mismatch)
        + len(dtype_mismatch)
        + len(numel_mismatch)
        + len(tp_rule_violations)
    )

    return {
        "summary": {
            "train_params": len(train),
            "infer_params": len(infer),
            "common_params": len(common),
            "missing_on_infer_count": len(missing_on_infer),
            "extra_on_infer_count": len(extra_on_infer),
            "unsupported_extra_on_infer_count": len(unsupported_extra),
            "shape_mismatch_count": len(shape_mismatch),
            "dtype_mismatch_count": len(dtype_mismatch),
            "numel_mismatch_count": len(numel_mismatch),
            "tp_rule_violation_count": len(tp_rule_violations),
            "sharding_type_diff_count": len(sharding_type_diff),
            "critical_error_count": critical_errors,
            "strict_key_match": strict_key_match,
        },
        "details": {
            "missing_on_infer": missing_on_infer,
            "extra_on_infer": extra_on_infer,
            "unsupported_extra_on_infer": unsupported_extra,
            "shape_mismatch": shape_mismatch,
            "dtype_mismatch": dtype_mismatch,
            "numel_mismatch": numel_mismatch,
            "tp_rule_violations": tp_rule_violations,
            "sharding_type_diff": sharding_type_diff,
        },
    }


def _print_report(report: dict[str, Any], max_items: int) -> None:
    s = report["summary"]
    d = report["details"]
    print("== Meta Analysis ==")
    for k in [
        "train_params",
        "infer_params",
        "common_params",
        "missing_on_infer_count",
        "extra_on_infer_count",
        "unsupported_extra_on_infer_count",
        "shape_mismatch_count",
        "dtype_mismatch_count",
        "numel_mismatch_count",
        "tp_rule_violation_count",
        "sharding_type_diff_count",
        "critical_error_count",
    ]:
        print(f"{k}: {s[k]}")

    def show_list(title: str, items: list[Any]) -> None:
        if not items:
            return
        print(f"\n[{title}] showing {min(len(items), max_items)} / {len(items)}")
        for item in items[:max_items]:
            print(item)

    show_list("missing_on_infer", d["missing_on_infer"])
    show_list("unsupported_extra_on_infer", d["unsupported_extra_on_infer"])
    show_list("shape_mismatch", d["shape_mismatch"])
    show_list("dtype_mismatch", d["dtype_mismatch"])
    show_list("numel_mismatch", d["numel_mismatch"])
    show_list("tp_rule_violations", d["tp_rule_violations"])
    show_list("sharding_type_diff", d["sharding_type_diff"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze AWEX train/infer meta dump files."
    )
    parser.add_argument(
        "--workspace", default=".", help="Directory to search dump files."
    )
    parser.add_argument(
        "--train-meta", default="", help="Path to train_params_meta*.json."
    )
    parser.add_argument(
        "--infer-meta", default="", help="Path to infer_params_meta*.json."
    )
    parser.add_argument(
        "--strict-key-match",
        action="store_true",
        help="Treat any infer-only keys as error (strict mode).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="Max number of items shown per mismatch bucket.",
    )
    parser.add_argument(
        "--json-out", default="", help="Write full report json to this path."
    )
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    train_meta_path = args.train_meta or _newest_file(
        os.path.join(workspace, "train_params_meta_*.json")
    )
    infer_meta_path = args.infer_meta or _newest_file(
        os.path.join(workspace, "infer_params_meta_*.json")
    )
    if not train_meta_path or not infer_meta_path:
        print(
            "ERROR: could not find both train/infer meta files. "
            "Use --train-meta and --infer-meta explicitly.",
            file=sys.stderr,
        )
        return 2

    print(f"train_meta: {train_meta_path}")
    print(f"infer_meta: {infer_meta_path}")
    train_meta = _load_json_list(train_meta_path)
    infer_meta = _load_json_list(infer_meta_path)

    report = analyze_meta(train_meta, infer_meta, args.strict_key_match)
    _print_report(report, args.max_items)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nWrote report: {args.json_out}")

    return 1 if report["summary"]["critical_error_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
