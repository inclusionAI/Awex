#!/usr/bin/env bash
set -euo pipefail

# No model downloads or accelerator allocation. Install .[dev,mcore] first.
export AWEX_DEVICE_TYPE=cpu
python -m pytest --timeout=120 \
  awex/tests/test_imports.py \
  awex/tests/test_runtime_contracts.py \
  awex/tests/test_weights_reader_init.py \
  awex/tests/test_meta_server.py \
  awex/tests/test_meta_server_deadlines.py \
  awex/tests/test_transfer_plan_cpu_sim.py \
  awex/tests/test_transfer_plan.py \
  awex/tests/test_meta_alignment_contract.py \
  awex/tests/test_model_registry.py \
  awex/tests/test_vllm_sharding.py \
  awex/tests/test_sglang_worker_hooks.py \
  awex/tests/test_nccl_bounded_stream.py \
  awex/tests/test_nccl_stream_batch.py \
  awex/tests/test_mooncake_transfer.py \
  awex/tests/test_qwen3_dense_sglang_converter.py \
  awex/tests/test_qwen3_moe_sglang_converter.py \
  awex/tests/test_qwen3_5.py \
  awex/tests/test_qwen3_vl.py \
  awex/tests/test_qwen4_exp.py \
  awex/tests/test_fused_mla_a_proj_layout.py \
  awex/tests/test_infer_meta_tied_embeddings.py \
  awex/tests/test_mcore_pp_mapping.py \
  awex/tests/test_vllm_converter.py "$@"
