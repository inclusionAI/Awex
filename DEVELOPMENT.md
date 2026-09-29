## How to build Awex

Please checkout the source tree from https://github.com/inclusionAI/awex.

### Build

### Build Awex Python

```bash
pip install -v -e ".[dev,mcore]"
```

#### Environment Requirements

- Python 3.10 or higher
- Install the PyTorch build required by your device before installing AWEX.
- CPU CI currently validates Python 3.10/3.12 with PyTorch 2.6.0,
  Transformers 4.57.1 and Megatron Core 0.16.1. Framework integration tests must
  use the version set deployed by the job.

### CPU and integration tests

```bash
bash ci/test_cpu.sh
```

This runs local metadata-server tests, reader/writer contracts, sharding and
converter tests without model downloads or GPU allocation. The native Mooncake
TCP test is skipped unless the optional engine is installed. `pytest --timeout`
limits test hangs; this does not certify GPU/NPU or complete model updates.

The `*_it.py` scripts are explicit integration entry points, outside pytest's
default collection. See the [README](README.md) for SGLang worker and
Megatron-to-vLLM commands, device requirements and model-path setup. Run those
when changing framework/GPU integration boundaries.

Validate release artifacts and the rendered package description with:

```bash
python -m build
python -m twine check --strict dist/*
```

CI also installs the wheel in a fresh environment and verifies inference imports
without Megatron. README images use absolute HTTPS URLs so they resolve on PyPI.

### Lint Markdown Docs

```bash
# Install prettier globally
npm install -g prettier@3.9.6

# Check Python and tracked Markdown without rewriting files (used by CI)
bash ci/format.sh --check

# Apply fixes locally
bash ci/format.sh --write
```

#### Environment Requirements

- node 14+
- npm 8+

## Contributing

For more information, please refer to [How to contribute to Awex](CONTRIBUTING.md).
