# Repository Guidelines

## Project Structure & Module Organization

The Python package lives under `awex/`, with submodules for `engine/` (training/inference orchestration), `transfer/` and `sharding/` (weight movement), `meta/` and `models/` (model-specific metadata), plus `util/` helpers. Integration tests are in `awex/tests/` and mirror the package layout; shared assets such as sample configs land in `docs/` or module-specific `meta/` folders. Keep new datasets or fixtures close to the code that exercises them, and gate experimental tools behind clearly named subpackages.

## Build, Test, and Development Commands

- `pip install -e .` — local editable install for the core library.
- `pip install -e ".[dev]"` — adds lint/test dependencies used by CI.
- `pytest -v -s .` — runs the full suite; narrow scope with paths like `pytest awex/tests/test_meta_resolver.py`.
- `ruff format . && ruff check --fix .` — standard Python formatting/linting pass.
- `prettier --write "**/*.md"` — fixes Markdown in `README.md`, `docs/`, etc.

## Coding Style & Naming Conventions

Python code follows 4-space indentation, type-hinted public APIs, and snake_case for functions, methods, and module names. Classes use CapWords, configuration constants stay upper snake case, and loggers are retrieved via `awex.logging`. Run Ruff before opening a PR; it enforces import ordering, docstring style, and basic safety rules. Files requiring license headers should be updated with `docker run ... hawkeye-native format` as shown in `CONTRIBUTING.md`.

## Testing Guidelines

Pytest drives all regression coverage. Place new tests beside the feature under `awex/tests/<module>/test_<feature>.py` so the intent is obvious. Prefer descriptive test names (`test_shard_plan_handles_gap`) and include synthetic weight files in `meta/` or temporary fixtures rather than relying on external assets. CI expects `pytest -v -s .` to pass locally; run with `-k` filters when iterating.

## Commit & Pull Request Guidelines

Commits and PR titles follow Conventional Commits because merges are squashed: `feat:`, `fix:`, `docs:`, `chore:` etc. Reference the impacted module in the summary (`feat: add RDMA shard planner`). PRs should include: brief context, validation notes (`pytest -v -s awex/tests/test_transfer.py`), linked issues, and any screenshots or tensor dumps that help reviewers. Keep changes small, document new configuration knobs in `docs/` or inline docstrings, and ensure formatting/lint passes before requesting review.
