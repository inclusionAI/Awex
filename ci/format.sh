#!/usr/bin/env bash
set -euo pipefail

case "${1:---write}" in
  --check)
    ruff format --check .
    ruff check .
    git ls-files -z -- '*.md' | xargs -0 prettier --check
    ;;
  --write)
    ruff format .
    ruff check --fix .
    git ls-files -z -- '*.md' | xargs -0 prettier --write
    ;;
  *)
    echo "Usage: bash ci/format.sh [--check|--write]" >&2
    exit 2
    ;;
esac
