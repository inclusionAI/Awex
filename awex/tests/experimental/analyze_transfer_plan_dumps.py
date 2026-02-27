#!/usr/bin/env python3
"""Analyze AWEX transfer-plan dump files (global/local communication plan)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Iterable, Sequence

_PLAN_NAME_RE = re.compile(
    r"(?P<kind>global|local)_communication_plan_(?P<role>train|infer)_(?P<rank>\d+)_(?P<pid>\d+)\.json$"
)


def _newest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _parse_plan_filename(path: str) -> dict[str, Any]:
    name = os.path.basename(path)
    m = _PLAN_NAME_RE.match(name)
    if not m:
        return {"kind": None, "role": None, "rank": None, "pid": None, "name": name}
    return {
        "kind": m.group("kind"),
        "role": m.group("role"),
        "rank": int(m.group("rank")),
        "pid": int(m.group("pid")),
        "name": name,
    }


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten_local_plan(plan_dict: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for _, op_list in plan_dict.items():
        if not isinstance(op_list, list):
            continue
        for op in op_list:
            if isinstance(op, dict):
                ops.append(op)
    return ops


def _prod(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _op_key(op: dict[str, Any]) -> tuple[Any, ...]:
    return (
        op.get("send_rank"),
        op.get("recv_rank"),
        op.get("send_shard_meta", {}).get("name"),
        tuple(op.get("send_offset", [])),
        tuple(op.get("recv_offset", [])),
        tuple(op.get("overlap_shape", [])),
    )


def _slice_len(dim_slice: Any) -> int | None:
    if isinstance(dim_slice, str):
        # Handles stringified slice formats if they appear.
        m = re.match(r"slice\(([-\d]+),\s*([-\d]+),\s*([-\d]+|None)\)", dim_slice)
        if not m:
            return None
        start = int(m.group(1))
        stop = int(m.group(2))
        step_raw = m.group(3)
        step = 1 if step_raw == "None" else int(step_raw)
        if step == 0:
            return None
        return (stop - start) // step
    if isinstance(dim_slice, dict):
        # to_dict usually stores slices as string, but support dict format anyway.
        start = int(dim_slice.get("start", 0))
        stop = int(dim_slice.get("stop", 0))
        step = int(dim_slice.get("step", 1) or 1)
        if step == 0:
            return None
        return (stop - start) // step
    return None


def _check_operation(op: dict[str, Any], idx: int) -> list[str]:
    errors: list[str] = []
    send_meta = op.get("send_shard_meta", {})
    recv_meta = op.get("recv_shard_meta", {})
    send_name = send_meta.get("name")
    recv_name = recv_meta.get("name")
    if send_name != recv_name:
        errors.append(
            f"op#{idx}: send/recv name mismatch: {send_name!r} vs {recv_name!r}"
        )

    overlap_shape = op.get("overlap_shape", [])
    if not overlap_shape or any(int(x) <= 0 for x in overlap_shape):
        errors.append(f"op#{idx}: invalid overlap_shape={overlap_shape}")
        return errors

    for key in ("send_offset", "recv_offset"):
        offset = op.get(key, [])
        if len(offset) != len(overlap_shape):
            errors.append(
                f"op#{idx}: {key} rank mismatch. len(offset)={len(offset)} len(shape)={len(overlap_shape)}"
            )

    for key, shape_key in (
        ("send_offset", "send_shard_meta"),
        ("recv_offset", "recv_shard_meta"),
    ):
        offset = op.get(key, [])
        shard_shape = op.get(shape_key, {}).get("shape", [])
        if len(offset) != len(shard_shape):
            continue
        for dim, (off, ov, dim_size) in enumerate(
            zip(offset, overlap_shape, shard_shape)
        ):
            if int(off) < 0:
                errors.append(f"op#{idx}: {key}[{dim}]={off} < 0")
            if int(off) + int(ov) > int(dim_size):
                errors.append(
                    f"op#{idx}: {key}[{dim}] + overlap[{dim}] exceeds shard dim: "
                    f"{off}+{ov}>{dim_size}"
                )

    # Best-effort slice sanity checks.
    for slice_key, _ov_key in (
        ("train_slices", "overlap_shape"),
        ("inf_slices", "overlap_shape"),
    ):
        slices = op.get(slice_key, [])
        if len(slices) != len(overlap_shape):
            errors.append(
                f"op#{idx}: {slice_key} rank mismatch. len={len(slices)} expected={len(overlap_shape)}"
            )
            continue
        for dim, (s, ov) in enumerate(zip(slices, overlap_shape)):
            slen = _slice_len(s)
            if slen is not None and int(slen) != int(ov):
                errors.append(
                    f"op#{idx}: {slice_key}[{dim}] length={slen} mismatch overlap={ov}"
                )
    return errors


def _coverage_check(ops: Iterable[dict[str, Any]]) -> list[str]:
    # For each receive shard, total received overlap should match shard numel.
    acc = defaultdict(int)
    expected = {}
    for op in ops:
        recv_meta = op.get("recv_shard_meta", {})
        key = (
            op.get("recv_rank"),
            recv_meta.get("name"),
            tuple(recv_meta.get("global_offset", [])),
            tuple(recv_meta.get("shape", [])),
        )
        overlap_numel = _prod(op.get("overlap_shape", []))
        acc[key] += overlap_numel
        expected[key] = int(recv_meta.get("numel", overlap_numel))

    errors: list[str] = []
    for key, recv_total in acc.items():
        recv_expected = expected[key]
        if recv_total != recv_expected:
            errors.append(
                f"recv coverage mismatch key={key}: received_numel={recv_total} expected_numel={recv_expected}"
            )
    return errors


def analyze_transfer_plans(
    global_ops: list[dict[str, Any]],
    local_ops: list[dict[str, Any]] | None,
    local_grouped: dict[str, Any] | None,
    role: str | None,
    rank: int | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    # Per-op validation + duplicate detection.
    seen: set[tuple[Any, ...]] = set()
    dup_count = 0
    for idx, op in enumerate(global_ops):
        errors.extend(_check_operation(op, idx))
        key = _op_key(op)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    if dup_count:
        warnings.append(f"global plan has duplicated ops: {dup_count}")

    errors.extend(_coverage_check(global_ops))

    if local_ops is not None and local_grouped is not None:
        local_seen: set[tuple[Any, ...]] = set()
        for idx, op in enumerate(local_ops):
            errors.extend(_check_operation(op, idx))
            local_seen.add(_op_key(op))

        missing_in_global = local_seen - seen
        if missing_in_global:
            errors.append(
                f"local plan has ops not in global plan: {len(missing_in_global)} entries"
            )

        # Group key consistency for local plan.
        for peer_rank_str, grouped_ops in local_grouped.items():
            try:
                peer_rank = int(peer_rank_str)
            except Exception:
                continue
            for op in grouped_ops:
                if role == "train":
                    if rank is not None and int(op.get("send_rank")) != rank:
                        errors.append(
                            f"local(train) op send_rank mismatch, expected {rank}, got {op.get('send_rank')}"
                        )
                    if int(op.get("recv_rank")) != peer_rank:
                        errors.append(
                            f"local(train) group key mismatch: peer={peer_rank}, op.recv_rank={op.get('recv_rank')}"
                        )
                elif role == "infer":
                    if rank is not None and int(op.get("recv_rank")) != rank:
                        errors.append(
                            f"local(infer) op recv_rank mismatch, expected {rank}, got {op.get('recv_rank')}"
                        )
                    if int(op.get("send_rank")) != peer_rank:
                        errors.append(
                            f"local(infer) group key mismatch: peer={peer_rank}, op.send_rank={op.get('send_rank')}"
                        )

    return {
        "summary": {
            "global_op_count": len(global_ops),
            "local_op_count": 0 if local_ops is None else len(local_ops),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "role": role,
            "rank": rank,
        },
        "errors": errors,
        "warnings": warnings,
    }


def _print_report(report: dict[str, Any], max_items: int) -> None:
    s = report["summary"]
    print("== Transfer Plan Analysis ==")
    for k in [
        "role",
        "rank",
        "global_op_count",
        "local_op_count",
        "error_count",
        "warning_count",
    ]:
        print(f"{k}: {s[k]}")
    if report["warnings"]:
        print(
            f"\n[WARN] showing {min(len(report['warnings']), max_items)} / "
            f"{len(report['warnings'])}"
        )
        for item in report["warnings"][:max_items]:
            print(item)
    if report["errors"]:
        print(
            f"\n[ERROR] showing {min(len(report['errors']), max_items)} / "
            f"{len(report['errors'])}"
        )
        for item in report["errors"][:max_items]:
            print(item)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze AWEX transfer plan dump files."
    )
    parser.add_argument(
        "--workspace", default=".", help="Directory to search dump files."
    )
    parser.add_argument(
        "--global-plan", default="", help="Path to global_communication_plan_*.json"
    )
    parser.add_argument(
        "--local-plan", default="", help="Path to local_communication_plan_*.json"
    )
    parser.add_argument(
        "--allow-local-missing",
        action="store_true",
        help="Allow running with only global plan.",
    )
    parser.add_argument(
        "--max-items", type=int, default=30, help="Max printed error/warn items."
    )
    parser.add_argument(
        "--json-out", default="", help="Write full report json to this path."
    )
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    global_path = args.global_plan or _newest_file(
        os.path.join(workspace, "global_communication_plan_*.json")
    )
    local_path = args.local_plan or _newest_file(
        os.path.join(workspace, "local_communication_plan_*.json")
    )

    if not global_path:
        print("ERROR: no global plan file found.", file=sys.stderr)
        return 2
    if not local_path and not args.allow_local_missing:
        print(
            "ERROR: no local plan file found. Use --allow-local-missing to skip.",
            file=sys.stderr,
        )
        return 2

    global_info = _parse_plan_filename(global_path)
    local_info = _parse_plan_filename(local_path) if local_path else {}
    role = local_info.get("role") or global_info.get("role")
    rank = local_info.get("rank")

    print(f"global_plan: {global_path}")
    if local_path:
        print(f"local_plan: {local_path}")

    global_data = _load_json(global_path)
    if not isinstance(global_data, list):
        raise ValueError(f"{global_path} must be a list of operations.")

    local_grouped = None
    local_ops = None
    if local_path:
        local_data = _load_json(local_path)
        if not isinstance(local_data, dict):
            raise ValueError(f"{local_path} must be a dict of peer_rank -> op list.")
        local_grouped = local_data
        local_ops = _flatten_local_plan(local_data)

    report = analyze_transfer_plans(
        global_ops=global_data,
        local_ops=local_ops,
        local_grouped=local_grouped,
        role=role,
        rank=rank,
    )
    _print_report(report, args.max_items)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nWrote report: {args.json_out}")

    return 1 if report["summary"]["error_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
