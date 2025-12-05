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

The Awex weight exchange framework consists primarily of three components:

- **WeightWriter**: Runs within each training process, responsible for metadata collection and reporting of weight shards for the current training process, weight convert, resharding transfer plan construction, weight transmission, and other functions;
- **WeightReader**: Runs on the control process of each inference instance, which starts a WorkerWeightsReader on each GPU managed by the inference instance, corresponding to the WeightWriter of the training process. Responsible for metadata collection and reporting of weight shards for each inference process, weight convert, resharding transfer plan construction, weight reception, and other functions;
- **MetaServer**: Job-level global server for service discovery and weight metadata exchange between training and inference engines, as well as event notification functions in co-located scenarios;

<div align="center">
  <img width="85%" alt="Apache Fory logo" src="docs/images/awex_arch.png"><br>
</div>

The core modules of weight exchange consist mainly of 5 parts:

- **Unified training-inference weight convert**: Responsible for converting weights from training and inference engines with **different parallelism strategies and tensor layouts** into a **unified format** for subsequent weight metadata calculation and weight transmission;
- **Global weight metadata calculation and exchange**: After converting training and inference weights into a unified format, collects all weight shard metadata from each worker and reports to Meta Server for subsequent weight transmission plan construction;
- **P2P weight transmission execution plan**: Training and inference engines obtain global weight shard metadata from all workers, then separately construct peer-to-peer deterministic transfer plan for sending and receiving;
- **NCCL weight transmission**: Uses NCCL's send/recv API for peer-to-peer weight transmission based on the constructed transmission plan;
- **RDMA weight transmission**: Uses NUMA affinity and RDMA communication for globally load-balanced transfer plan for weight updates;

Awex also supports tensor-level validation of weights, comparing weights loaded through file system mode with those loaded through transmission mode at the tensor level for fine-grained comparison, ensuring the correctness of the transmission mode.

See more details on our [Document](docs).

For comprehensive introduction about awex, see the [medium article](https://medium.com/@shawn.ck.yang/awex-an-ultra-fast-weight-sync-framework-powering-trillion-scale-reinforcement-learning-766ebc79f58b)

## Performance Benchmarks

On thousand-GPU scale clusters, Awex using NCCL transmission can **exchange 10B-scale model weights within one second**, and **exchange 1T-scale model weights within twenty seconds**. Using RDMA for transmission, **1T model weight exchange time** can be further **reduced to six seconds**.

| Weight Parameter Scale | Weight Data Size | Verl Time | Awex NCCL Transmission Time | Awex RDMA Transmission Time |
| ---------------------- | ---------------- | --------- | --------------------------- | --------------------------- |
| 10B                    | 31GB             | 3.5S      | 0.8S                        | 0.5S                        |
| 100B                   | 191GB            | 35S       | 9S                          | 3.2S                        |
| 1000B                  | 1000GB (FP8)     | /         | 20S                         | 6S                          |

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
from awex import WeightsReader, InferenceConfig
from awex.engine.sglang import SGLangEngine
import sglang as sgl

sgl_engine = sgl.Engine(model_path="xxx", tp_size=2, random_seed=42)
awex_config = InferenceConfig.from_sgl_engine(sgl_engine, comm_backend="nccl")
# for sglang support, you must ensure https://github.com/sgl-project/sglang/pull/13595
# is included in your sglang version
inference_engine = SGLangEngine(awex_config, sgl_engine)
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
