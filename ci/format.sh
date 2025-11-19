#!/usr/bin/env bash
set -euo pipefail

if ! command -v ruff >/dev/null 2>&1; then
  pip install ruff
fi

if ! command -v prettier >/dev/null 2>&1; then
  npm install -g prettier
fi

ruff format .
ruff check --fix .
prettier --write "**/*.md"
