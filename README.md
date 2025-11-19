# Awex

**Awex** is a high-performance RL training-inference **weight synchronization** framework,
designed to enable **second-level parameter updates** from training to inference in RL workflows.
It minimizes iteration latency, ensuring rollout phases consistently use the latest model.

## 🚀 Key Features

- **Extreme Sync Speed**: **Trillion-parameter models** fully synchronized within **10 seconds**; validated on thousand-GPU
  clusters with industry-leading performance.
- **Unified Weight Adaptation Layer**: Automatically **handles tensor format/layout differences** across parallel strategies
  and engine frameworks, supporting any model architecture.
- **Zero-Redundancy Transfer & In-Place Update**: Transfers only necessary shards; supports in-place GPU memory updates
  on inference, avoiding costly allocation and copying.
- **Multi-Mode Transfer Support**: Support NCCL, RDMA, and shared memory transfer mode to leverage NVLink/NVSwitch/RDMA
  bandwidth and reduce long-tail latency.
- **Heterogeneous Deployment Compatibility**: Fully supports co-location and separation modes, make RL sync/async
  algorithms runs seamlessly.
- Extensibility: Easily extends to support new training and inference engines.

## Architecture

<div align="center">
  <img width="85%" alt="Apache Fory logo" src="docs/images/awex_arch.png"><br>
</div>

## 📦 Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0.0 or higher (for GPU support)

### Basic Installation

Install awex using pip:

```bash
pip install awex
```

### Build from Source

Clone the repository and install in development mode:

```bash
git clone git@github.com:inclusionAI/awex.git
cd awex
pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

Awex is a pure Python library that can be installed and used with one command, supporting Python 3.8 and above.

```bash
pip install awex
```

Megatron training engine weight sending example:

```python
from awex import NCCLWeightsWriter
from awex.engine.mcore import MegatronEngine

# init
train_engine = MegatronEngine(awex_config, hf_config, mcore_model)
writer = NCCLWeightsWriter(train_engine)
writer.initialize()

# write weights
writer.write_weights(step_id=1)
```

SGLang inference engine weight update example:

```python
from awex import WeightsReader
from awex.engine.sglang import SGLangEngine
import sglang as sgl

sgl_engine = sgl.Engine(model_path="xxx", tp_size=2, random_seed=42)
inference_engine = SGLangEngine(awex_confg, sgl_engine)
reader = WeightsReader(inference_engine)
reader.initialize()

# update weights
reader.update_weights(step_id=1)
```

## 🤝 Contributing

Awex is an open-source project. We welcome all forms of contributions:

### How to Contribute

1. **Report Issues**: Found a bug? [Open an issue](https://github.com/inclusionAI/awex/issues)
2. **Suggest Features**: Have an idea? Start a discussion
3. **Improve Docs**: Documentation improvements are always welcome
4. **Submit Code**: See our [Contributing Guide](https://github.com/inclusionAI/awex/blob/main/CONTRIBUTING.md)
5. **Agent Workflows**: Read the [Repository Guidelines](AGENTS.md) for structure, testing, and PR expectations.

### Development Setup

```bash
git clone https://github.com/inclusionAI/awex.git
cd awex

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v -s .

# Run specific test
pytest -v -s awex/tests/test_meta_resolver.py

# Format code
ruff format .
ruff check --fix .
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed build instructions.

## 📄 License

Apache License 2.0. See [LICENSE](https://github.com/inclusionAI/awex/blob/main/LICENSE) for details.

---

**Awex** - high-performance RL training-inference **weight synchronization** framework with **second-level parameter updates**

## 🌟 Community

We welcome contributions! Whether it's bug reports, feature requests, documentation improvements, or code contributions,
we appreciate your help.

- Star the project on [GitHub](https://github.com/inclusionAI/awex) ⭐
