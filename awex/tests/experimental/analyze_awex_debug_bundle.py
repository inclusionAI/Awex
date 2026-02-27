#!/usr/bin/env python3
"""Run both AWEX meta and transfer-plan analyzers on one workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AWEX debug dump bundle.")
    parser.add_argument(
        "--workspace", default=".", help="Directory containing AWEX dump json files."
    )
    parser.add_argument(
        "--strict-key-match",
        action="store_true",
        help="Pass strict key check to meta analyzer.",
    )
    parser.add_argument(
        "--allow-local-missing",
        action="store_true",
        help="Allow transfer analyzer to run without a local plan file.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_script = os.path.join(script_dir, "analyze_meta_dumps.py")
    plan_script = os.path.join(script_dir, "analyze_transfer_plan_dumps.py")

    meta_cmd = [sys.executable, meta_script, "--workspace", args.workspace]
    if args.strict_key_match:
        meta_cmd.append("--strict-key-match")

    plan_cmd = [sys.executable, plan_script, "--workspace", args.workspace]
    if args.allow_local_missing:
        plan_cmd.append("--allow-local-missing")

    meta_rc = _run(meta_cmd)
    plan_rc = _run(plan_cmd)

    print("\n== Bundle Summary ==")
    print(f"meta_analyzer_rc: {meta_rc}")
    print(f"plan_analyzer_rc: {plan_rc}")

    # Return non-zero if either analyzer detected errors.
    return 0 if (meta_rc == 0 and plan_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
