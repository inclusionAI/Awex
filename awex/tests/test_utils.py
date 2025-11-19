# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoConfig, PretrainedConfig


def is_huggingface_available() -> bool:
    """
    Check if HuggingFace is accessible.

    Returns:
        True if HuggingFace is accessible, False otherwise
    """
    try:
        import socket
        import urllib.request

        # Set a short timeout to quickly detect network issues
        socket.setdefaulttimeout(5)

        # Try to access HuggingFace
        urllib.request.urlopen("https://huggingface.co", timeout=1)
        return True
    except Exception:
        return False


def setup_modelscope_cache():
    """
    Setup ModelScope cache directory and environment.
    """
    try:
        import modelscope  # noqa: F401

        return True
    except ImportError:
        print(
            "Warning: modelscope is not installed. Install it with: pip install modelscope"
        )
        return False


def _resolve_local_model_dir_and_config(
    model_path: str,
) -> Tuple[str, PretrainedConfig]:
    """Resolve a model path to a local directory and HF config.

    This helper encapsulates the download logic shared between tests
    that need a local model directory (for Megatron or sglang) and
    supports both HuggingFace and ModelScope backends.
    """
    # Detect network and select source
    use_modelscope = False
    model_path_for_download = model_path
    if not is_huggingface_available():
        print("HuggingFace is not accessible, trying ModelScope...")
        if setup_modelscope_cache():
            use_modelscope = True
            # Map HuggingFace model names to ModelScope equivalents
            modelscope_map = {
                "Qwen/Qwen2-1.5B": "qwen/Qwen2-1.5B",
                "Qwen/Qwen2-7B": "qwen/Qwen2-7B",
                "Qwen/Qwen2.5-1.5B": "qwen/Qwen2.5-1.5B",
                "Qwen/Qwen2.5-7B": "qwen/Qwen2.5-7B",
            }
            model_path_for_download = modelscope_map.get(
                model_path, model_path.replace("Qwen/", "qwen/")
            )
        else:
            print(
                "Warning: Neither HuggingFace nor ModelScope is available. "
                "Attempting to load from local cache..."
            )

    print(
        f"Loading model from {'ModelScope' if use_modelscope else 'HuggingFace'}: {model_path}"
    )

    if use_modelscope:
        # Download model from ModelScope
        try:
            from modelscope import snapshot_download

            local_model_path = snapshot_download(
                model_path_for_download,
                cache_dir=os.path.expanduser("~/.cache/modelscope"),
            )
            print(f"Model downloaded to: {local_model_path}")
            hf_model_dir = local_model_path
        except Exception as e:
            print(f"Failed to download from ModelScope: {e}")
            print("Falling back to HuggingFace (may fail if not accessible)...")
            hf_model_dir = model_path
    else:
        # Download from HuggingFace (or use local cache if already present)
        from transformers import AutoModelForCausalLM

        print(f"Downloading {model_path} from HuggingFace...")
        AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        hf_model_dir = model_path

    hf_config = AutoConfig.from_pretrained(
        hf_model_dir,
        trust_remote_code=True,
    )

    return hf_model_dir, hf_config


def get_local_model_dir(model_path: str = "Qwen/Qwen2-1.5B") -> str:
    """Ensure model weights/config are available locally and return directory.

    This is used by tests that need a local model path (e.g., sglang
    backend) without hard-coding HuggingFace or ModelScope behavior.
    """
    hf_model_dir, _ = _resolve_local_model_dir_and_config(model_path)
    return hf_model_dir


def megatron_model_from_hf(
    model_path: str = "Qwen/Qwen2-1.5B",
) -> Tuple[list, PretrainedConfig]:
    """Convert HF/ModelScope model to DCP format and load into Megatron.

    This function:
    1. Ensures the model is available locally (via HF or ModelScope)
    2. Converts HF weights to Megatron DCP format using convert.py
    3. Initializes Megatron model with TP=PP=DP=EP=CP=1
    4. Loads the DCP checkpoint into Megatron model
    5. Returns Megatron model list and HF config

    Note:
        This creates a temporary DCP checkpoint in /tmp/megatron_dcp_<model_name>
    """
    import subprocess
    import sys

    hf_model_dir, hf_config = _resolve_local_model_dir_and_config(model_path)

    print("HF Config loaded:")
    print(f"  Model type: {hf_config.model_type}")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num attention heads: {hf_config.num_attention_heads}")
    print(
        f"  Num KV heads: {getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)}"
    )
    print(f"  Vocab size: {hf_config.vocab_size}")

    # Create temporary directory for DCP checkpoint
    model_name = model_path.split("/")[-1]
    dcp_dir = f"/tmp/megatron_dcp_{model_name}"
    os.makedirs(dcp_dir, exist_ok=True)

    print("\nConverting HF weights to Megatron DCP format...")
    print(f"  Source: {hf_model_dir}")
    print(f"  Target: {dcp_dir}")

    # Check if checkpoint already exists to skip conversion
    if os.path.exists(f"{dcp_dir}/iter_0000001") or os.path.exists(
        f"{dcp_dir}/latest_checkpointed_iteration.txt"
    ):
        print(f"DCP checkpoint already exists at {dcp_dir}, skipping conversion")
    else:
        # Find convert.py in Megatron-LM (assume it's on Python path)
        try:
            import megatron.training

            # Try to get the path from a submodule that has __file__
            megatron_module_path = megatron.training.__file__
            if megatron_module_path:
                # Go up from megatron/training/__init__.py to Megatron-LM root
                megatron_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(megatron_module_path))
                )
            else:
                raise RuntimeError("Cannot determine Megatron-LM path from module")
        except Exception as e:
            raise RuntimeError(
                f"Cannot find Megatron-LM installation: {e}. "
                "Please ensure Megatron-LM is properly installed and on PYTHONPATH."
            ) from e

        convert_script = f"{megatron_root}/tools/checkpoint/convert.py"

        if not os.path.exists(convert_script):
            raise RuntimeError(
                f"Cannot find Megatron conversion script at {convert_script}. "
                "Please ensure Megatron-LM is properly installed."
            )

        print(f"Using Megatron-LM from: {megatron_root}")

        # Determine tokenizer model path
        tokenizer_model = (
            f"{hf_model_dir}/tokenizer.model"
            if os.path.exists(f"{hf_model_dir}/tokenizer.model")
            else hf_model_dir
        )

        convert_cmd = [
            sys.executable,
            convert_script,
            "--model-type",
            "GPT",
            "--loader",
            "llama_mistral",
            "--saver",
            "core",
            "--model-size",
            "qwen2.5",
            "--checkpoint-type",
            "hf",
            "--load-dir",
            hf_model_dir,
            "--save-dir",
            dcp_dir,
            "--tokenizer-model",
            tokenizer_model,
            "--target-tensor-parallel-size",
            "1",
            "--target-pipeline-parallel-size",
            "1",
            "--bf16",
        ]

        print(f"Running conversion command: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Conversion failed with return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError("Failed to convert HF model to DCP format")

        print(f"Conversion stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Conversion stderr:\n{result.stderr}")

        print("Conversion completed successfully!")

    # Now initialize Megatron and load the checkpoint
    print("\nInitializing Megatron model...")
    model = initialize_megatron_and_load_checkpoint(dcp_dir, hf_config, hf_model_dir)

    # Return as list (Megatron expects a list for virtual pipeline parallelism support)
    return [model], hf_config


def initialize_megatron_and_load_checkpoint(dcp_dir, hf_config, hf_model_dir):
    """
    Initialize Megatron with all parallel sizes = 1 and load DCP checkpoint.

    Args:
        dcp_dir: Directory containing the DCP checkpoint
        hf_config: HuggingFace config
        hf_model_dir: Directory containing the HuggingFace model (for tokenizer)

    Returns:
        Megatron GPTModel instance
    """
    import sys

    # Add Megatron-LM root to path for model_provider and gpt_builders imports
    import megatron.training

    megatron_module_path = megatron.training.__file__
    megatron_root = os.path.dirname(
        os.path.dirname(os.path.dirname(megatron_module_path))
    )
    if megatron_root not in sys.path:
        sys.path.insert(0, megatron_root)

    from megatron.core import mpu
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import set_global_variables

    # Honor DEVICE env before any Megatron CUDA work so that Megatron
    # initializes directly on the requested GPU (e.g., device 1 for
    # training tests) rather than defaulting to device 0 and moving
    # later.
    if torch.cuda.is_available():
        device_env = os.environ.get("DEVICE")
        if device_env is not None:
            try:
                device_id = int(device_env)
                print(
                    f"Setting torch CUDA device to {device_id} based on DEVICE={device_env}"
                )
                torch.cuda.set_device(device_id)
            except Exception as e:
                print(
                    f"Warning: Failed to set CUDA device from DEVICE={device_env}: {e}"
                )

    # Parse default args first
    args = parse_args(extra_args_provider=None, ignore_unknown_args=True)

    # Create config dict with values we want to override
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", hf_config.num_attention_heads
    )
    rope_theta = int(getattr(hf_config, "rope_theta", 10000))

    config_dict = {
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "seq_length": 4096,
        "max_position_embeddings": getattr(hf_config, "max_position_embeddings", 4096),
        "micro_batch_size": 1,
        "global_batch_size": 1,
        "tensor_model_parallel_size": 1,
        "encoder_tensor_model_parallel_size": 1,  # Must match tensor_model_parallel_size
        "pipeline_model_parallel_size": 1,
        "masked_softmax_fusion": False,
        "bias_gelu_fusion": False,
        "bias_dropout_fusion": False,
        "gradient_accumulation_fusion": False,
        "async_tensor_model_parallel_allreduce": False,  # Disable to avoid CUDA_DEVICE_MAX_CONNECTIONS requirement
        "bf16": True,
        "normalization": "RMSNorm",
        "position_embedding_type": "rope",
        "swiglu": True,
        "untie_embeddings_and_output_weights": True,
        "disable_bias_linear": True,
        "position_embedding": False,
        "use_rotary_position_embeddings": True,
        "rotary_percent": 1.0,
        "rotary_base": rope_theta,
        "num_query_groups": num_kv_heads,
        "load": dcp_dir,
        "no_load_optim": True,
        "no_load_rng": True,
        "transformer_impl": "transformer_engine",  # Use TE which supports RMSNorm
        "num_experts": 0,
        "rotary_seq_len_interpolation_factor": 1.0,
        "padded_vocab_size": hf_config.vocab_size,
        "tokenizer_type": "HuggingFaceTokenizer",
        "tokenizer_model": hf_model_dir,
    }

    # Override default args with our config
    for key, value in config_dict.items():
        setattr(args, key, value)

    # Validate and set global variables
    validate_args(args)
    set_global_variables(args)

    # Re-initialize model parallel state after set_global_variables
    # set_global_variables may reset the parallel state, so we need to reinitialize
    # Use the manual approach from Megatron's checkpoint loader
    mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(
        args.virtual_pipeline_model_parallel_size or 1
    )
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)

    # Initialize CUDA RNG tracker for model parallel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(args.seed)

    # Also load fused kernels if needed
    try:
        from megatron.legacy import fused_kernels

        fused_kernels.load(args)
    except Exception as e:
        print(f"Warning: Could not load fused kernels: {e}")

    # Build model using custom qwen2_model_provider. By this point the
    # current CUDA device has already been set from DEVICE (if
    # available), so model parameters are created directly on that
    # device.
    print("Building Megatron GPT model...")
    model = qwen2_model_provider(pre_process=True, post_process=True)

    # Load checkpoint
    print(f"Loading checkpoint from {dcp_dir}...")
    # Disable weights_only mode for checkpoint loading since we trust our own converted checkpoint
    # PyTorch 2.6 changed the default to weights_only=True which requires allowlisting all custom types
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        iteration = load_checkpoint([model], None, None)
        print(f"Loaded checkpoint at iteration {iteration}")
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {str(e)[:100]}")
        print(
            "Using randomly initialized model instead (sufficient for testing weights writer)"
        )

    return model


def build_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = 0
    extra_vocab_size = getattr(args, "extra_vocab_size", 0)
    args.padded_vocab_size = tokenizer.vocab_size + extra_vocab_size


def qwen2_model_provider(pre_process=True, post_process=True):
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer import TransformerConfig
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    @dataclass
    class Qwen2TransformerConfig(TransformerConfig):
        transformer_impl: str = "transformer_engine"
        moe_ffn_hidden_size: int = None
        shared_moe_ffn_hidden_size: int = None
        enable_shared_expert: bool = False
        num_shared_experts: int = None
        moe_layer_freq: int = None
        rotary_base: int = None
        rotary_scaling_factor: int = None
        max_position_embeddings: int = None
        moe_aux_loss_coeff: float = 0.0

    args = get_args()
    build_tokenizer(args)
    print("building qwen2 model ...")
    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"
    if use_te:
        print("building qwen2 model in TE...")
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
    else:
        print("building qwen2 model in Mcore...")
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )
    return model
