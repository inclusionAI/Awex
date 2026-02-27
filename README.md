# Awex

[![Build Status](https://img.shields.io/github/actions/workflow/status/inclusionAI/asystem-awex/ci.yml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://github.com/inclusionAI/asystem-awex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/awex.svg?style=for-the-badge&logo=PyPI)](https://pypi.org/project/awex/)
[![Python Versions](https://img.shields.io/pypi/pyversions/awex.svg?style=for-the-badge&logo=python)](https://pypi.org/project/awex/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)


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

## Weight Conversion Tests

These scripts compare weight formats across Megatron, vLLM, and SGLang by
converting all parameters into HF-style names and then diffing tensors.

**Intended use** (for new model bring‑up):
- These scripts primarily validate Awex converter coverage. They help answer:
  “Does the current converter support this new model, or do we need mapping fixes?”
- If your target stack is **Megatron → vLLM**, usually running
  `verify_weight_conversion.py` + `compare_megatron_vllm_weights.py` is sufficient.
- Use `compare_vllm_sglang_weights.py` only if you also care about **vLLM ↔ SGLang**
  parity (or you’re adding SGLang support for a new model).

**GPU/NPU notes**
- All compare/verify scripts accept `--device-backend` (auto/cuda/npu/cpu), but
  they are **CUDA-only today** because vLLM/SGLang backends require CUDA.
  Use `--device-backend cuda` explicitly if auto-detection picks the wrong device.
- For NPU, use these scripts on CUDA to validate **converter coverage**, then
  validate the **runtime weight update** path on NPU with the integration tests.

### Naming normalization (why `self_attn.qkv_proj` becomes `attention.query_key_value_proj`)
Awex normalizes parameter names from different backends into a single canonical
HF-style naming scheme so Megatron, vLLM, and SGLang can be compared directly.
There are three “namespaces” involved:

1) **Megatron (mcore) names** – e.g. `decoder.layers.0.self_attention.linear_qkv.weight`  
2) **vLLM/SGLang names** – e.g. `model.layers.0.self_attn.qkv_proj.weight`  
3) **Awex canonical HF-style names** – e.g. `model.layers.0.attention.query_key_value_proj.weight`

Example for **QKV** conversion:
- Megatron `self_attention.linear_qkv.weight`  
  → (mcore converter) `self_attn.qkv_proj.weight`  
  → (normalize) `attention.query_key_value_proj.weight`
- vLLM `self_attn.qkv_proj.weight`  
  → (normalize) `attention.query_key_value_proj.weight`

So `self_attn.qkv_proj` is **not** the canonical HF name; it is a vLLM name (and
also an intermediate name in the Megatron converter). The canonical name used
for comparison is `attention.query_key_value_proj`.

**Qwen3 note:** HF checkpoints store unfused `q_proj/k_proj/v_proj` weights. The
verifier treats those as valid matches for the canonical `query_key_value_proj`.

- Compare vLLM vs SGLang HF-loaded weights:
  - Script: `awex/tests/experimental/compare_vllm_sglang_weights.py`
  - Example:
    ```bash
    python awex/tests/experimental/compare_vllm_sglang_weights.py \
      --model-path /path/to/hf/model \
      --out-dir /tmp/vllm_sglang_compare \
      --device-backend cuda \
      --trust-remote-code \
      --max-layers 4 \
      --include-non-layer
    ```
- Compare Megatron vs vLLM (via converters to HF naming):
  - Script: `awex/tests/experimental/compare_megatron_vllm_weights.py`
  - Note: We default to mbridge for all models. Use `--no-mbridge` to force the
    Megatron convert.py path (Qwen3 will still fall back to mbridge).
  - Example:
    ```bash
    python awex/tests/experimental/compare_megatron_vllm_weights.py \
      --model-path /path/to/hf/model \
      --out-dir /tmp/megatron_vllm_compare \
      --device-backend cuda \
      --trust-remote-code \
      --max-layers 4 \
      --include-non-layer
    ```
  - Multi-GPU (torchrun) variant:
    - Script: `awex/tests/experimental/compare_megatron_vllm_weights_multi.py`
    - Example:
      ```bash
      torchrun --nproc_per_node=2 awex/tests/experimental/compare_megatron_vllm_weights_multi.py \
        --stage megatron_dump \
        --model-path /path/to/hf/model \
        --out-dir /tmp/megatron_vllm_compare \
        --device-backend cuda \
        --train-tp-size 2 \
        --train-pp-size 1 \
        --train-ep-size 1 \
        --train-cuda-devices 0,1
      python awex/tests/experimental/compare_megatron_vllm_weights_multi.py \
        --stage vllm_compare \
        --model-path /path/to/hf/model \
        --out-dir /tmp/megatron_vllm_compare
      ```

Both scripts produce a JSON report with missing keys, shape/dtype mismatches,
and value diffs. You can limit comparison to the first N layers with
`--max-layers N`. For large models, expect heavy disk usage because each tensor
is saved to disk for comparison.

- Verify HF weight conversion coverage:
  - Script: `awex/tests/experimental/verify_weight_conversion.py`
  - Note: Qwen3 HF checkpoints store unfused q/k/v (and o_proj) weights, so the
    verifier treats those as valid matches for vLLM qkv/o_proj names.
  - Example:
    ```bash
    python awex/tests/experimental/verify_weight_conversion.py \
      --model-path /path/to/hf/model \
      --device-backend cuda
    ```

## Integration Tests (Awex ↔ vLLM)

- Megatron → vLLM weight exchange (requires 2 GPUs and Awex vLLM plugin):
  - Script: `awex/tests/weights_exchange_vllm_it.py`
  - Example:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python awex/tests/weights_exchange_vllm_it.py \
      --comm_backend nccl \
      --model-path /path/to/hf/model \
      --device-backend cuda \
      --validate
    ```
  - Optional: add `--validate` to run a consistency check and print
    "weights are consistent" logs (supported for NCCL or file backend).
  - `--model-path` defaults to `vllm_inference_config["model_path"]` inside the
    script. Set it explicitly for your local model directory.
  - NPU (experimental, requires vllm-ascend + MindSpeed + Megatron):
    ```bash
    ASCEND_RT_VISIBLE_DEVICES=0,1 AWEX_USE_MINDSPEED=1 \
      python awex/tests/weights_exchange_vllm_it.py \
      --comm_backend hccl \
      --device-backend npu
    ```
  - Multi-process (`torchrun`) integration is currently excluded because startup
    is not stable in our test environment. Use the single-process script above
    as the baseline validation path.

## NPU / MindSpeed Notes (Experimental)

Awex includes **experimental** NPU support for the weight-exchange runtime path
(training ↔ inference). This path is intended for **MindSpeed + Megatron** on
Ascend and **vllm-ascend** on the inference side.

- **Device backend**: set `AWEX_DEVICE_TYPE=npu` to switch the internal device
  helpers to NPU semantics. For communication, use `comm_backend=hccl` and
  `weights_exchange_ipc_backend=cpu` (CUDA IPC is not supported on NPU).
- **MindSpeed patching**: set `AWEX_USE_MINDSPEED=1` **before** importing
  `megatron` / `megatron.core` so MindSpeed can patch Megatron internals.
- **Inference**: requires `vllm-ascend` with the Awex plugin enabled. This
  integration has been validated in our environment.
- **Memory debug logging**: set `AWEX_MEM_DEBUG=1` to emit additional memory
  diagnostics during weight conversion and NCCL send-op construction. This is
  intended for debugging memory pressure or unexpected tensor retention and
  should remain disabled in normal runs.

**What is NOT NPU-ready yet**
- `compare_megatron_vllm_weights.py`, `verify_weight_conversion.py`, and
  `compare_vllm_sglang_weights.py` are **CUDA-only** (they rely on vLLM CUDA
  kernels and torch.cuda).
- If you target NPU, use these scripts on CUDA to validate **converter
  coverage**, then validate the **runtime weight update** path on NPU.

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

# Run heavy GPU integration tests (requires Megatron-LM and 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 pytest -v -s awex/tests/test_weights_writer.py

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
