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

from typing import Tuple
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig


def is_huggingface_available() -> bool:
    """
    Check if HuggingFace is accessible.

    Returns:
        True if HuggingFace is accessible, False otherwise
    """
    try:
        import urllib.request
        import socket

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
        from modelscope.hub.snapshot_download import snapshot_download
        return True
    except ImportError:
        print("Warning: modelscope is not installed. Install it with: pip install modelscope")
        return False


def megatron_model_from_hf(
    model_path: str = "Qwen/Qwen2-1.5B",
) -> Tuple[list, PretrainedConfig]:
    """
    Convert HuggingFace model to DCP format and load into Megatron.

    This function:
    1. Downloads HuggingFace model if needed
    2. Converts HF weights to Megatron DCP format using convert.py
    3. Initializes Megatron model with TP=PP=DP=EP=CP=1
    4. Loads the DCP checkpoint into Megatron model
    5. Returns Megatron model list and HF config

    Args:
        model_path: HuggingFace model path (default: Qwen/Qwen2-1.5B)

    Returns:
        Tuple of ([megatron_model], hf_config)
        The model is a real Megatron GPT model wrapped in a list for VPP support

    Note:
        This creates a temporary DCP checkpoint in /tmp/megatron_dcp_<model_name>
    """
    import sys
    import tempfile
    import subprocess

    # Detect network and use appropriate source
    use_modelscope = False
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
            model_path_for_download = modelscope_map.get(model_path, model_path.replace("Qwen/", "qwen/"))
        else:
            print("Warning: Neither HuggingFace nor ModelScope is available. Attempting to load from local cache...")

    print(f"Loading model from {'ModelScope' if use_modelscope else 'HuggingFace'}: {model_path}")

    # Download model from ModelScope if needed
    if use_modelscope:
        try:
            from modelscope import snapshot_download
            local_model_path = snapshot_download(model_path_for_download, cache_dir=os.path.expanduser("~/.cache/modelscope"))
            print(f"Model downloaded to: {local_model_path}")
            hf_model_dir = local_model_path
        except Exception as e:
            print(f"Failed to download from ModelScope: {e}")
            print("Falling back to HuggingFace (may fail if not accessible)...")
            hf_model_dir = model_path
    else:
        # Download from HuggingFace
        from transformers import AutoModelForCausalLM
        print(f"Downloading {model_path} from HuggingFace...")
        AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        hf_model_dir = model_path

    # Load config
    hf_config = AutoConfig.from_pretrained(
        hf_model_dir,
        trust_remote_code=True,
    )

    print("HF Config loaded:")
    print(f"  Model type: {hf_config.model_type}")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num attention heads: {hf_config.num_attention_heads}")
    print(f"  Num KV heads: {getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)}")
    print(f"  Vocab size: {hf_config.vocab_size}")

    # Create temporary directory for DCP checkpoint
    model_name = model_path.split("/")[-1]
    dcp_dir = f"/tmp/megatron_dcp_{model_name}"
    os.makedirs(dcp_dir, exist_ok=True)

    print(f"\nConverting HF weights to Megatron DCP format...")
    print(f"  Source: {hf_model_dir}")
    print(f"  Target: {dcp_dir}")

    # Check if checkpoint already exists to skip conversion
    if os.path.exists(f"{dcp_dir}/iter_0000001") or os.path.exists(f"{dcp_dir}/latest_checkpointed_iteration.txt"):
        print(f"DCP checkpoint already exists at {dcp_dir}, skipping conversion")
    else:
        # Find convert.py in Megatron-LM (assume it's on Python path)
        try:
            import megatron.training
            # Try to get the path from a submodule that has __file__
            megatron_module_path = megatron.training.__file__
            if megatron_module_path:
                # Go up from megatron/training/__init__.py to Megatron-LM root
                megatron_root = os.path.dirname(os.path.dirname(os.path.dirname(megatron_module_path)))
            else:
                raise RuntimeError("Cannot determine Megatron-LM path from module")
        except Exception as e:
            raise RuntimeError(
                f"Cannot find Megatron-LM installation: {e}. "
                "Please ensure Megatron-LM is properly installed and on PYTHONPATH."
            )

        convert_script = f"{megatron_root}/tools/checkpoint/convert.py"

        if not os.path.exists(convert_script):
            raise RuntimeError(
                f"Cannot find Megatron conversion script at {convert_script}. "
                "Please ensure Megatron-LM is properly installed."
            )

        print(f"Using Megatron-LM from: {megatron_root}")

        # Determine tokenizer model path
        tokenizer_model = f"{hf_model_dir}/tokenizer.model" if os.path.exists(f"{hf_model_dir}/tokenizer.model") else hf_model_dir

        convert_cmd = [
            sys.executable, convert_script,
            "--model-type", "GPT",
            "--loader", "llama_mistral",
            "--saver", "core",
            "--model-size", "qwen2.5",
            "--checkpoint-type", "hf",
            "--load-dir", hf_model_dir,
            "--save-dir", dcp_dir,
            "--tokenizer-model", tokenizer_model,
            "--target-tensor-parallel-size", "1",
            "--target-pipeline-parallel-size", "1",
            "--bf16",
        ]

        print(f"Running conversion command: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Conversion failed with return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Failed to convert HF model to DCP format")

        print(f"Conversion stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Conversion stderr:\n{result.stderr}")

        print("Conversion completed successfully!")

    # Now initialize Megatron and load the checkpoint
    print("\nInitializing Megatron model...")
    model = initialize_megatron_and_load_checkpoint(dcp_dir, hf_config)

    # Return as list (Megatron expects a list for virtual pipeline parallelism support)
    return [model], hf_config


def initialize_megatron_and_load_checkpoint(dcp_dir, hf_config):
    """
    Initialize Megatron with all parallel sizes = 1 and load DCP checkpoint.

    Args:
        dcp_dir: Directory containing the DCP checkpoint
        hf_config: HuggingFace config

    Returns:
        Megatron GPTModel instance
    """
    import sys

    # Add Megatron-LM root to path for model_provider and gpt_builders imports
    import megatron.training
    megatron_module_path = megatron.training.__file__
    megatron_root = os.path.dirname(os.path.dirname(os.path.dirname(megatron_module_path)))
    if megatron_root not in sys.path:
        sys.path.insert(0, megatron_root)

    from megatron.training import get_args
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_args, set_global_variables
    from megatron.core import mpu
    from megatron.training.checkpointing import load_checkpoint
    from megatron.core.models.gpt import GPTModel
    from model_provider import model_provider
    from gpt_builders import gpt_builder

    # Create Megatron args
    num_kv_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)

    megatron_args = [
        "--num-layers", str(hf_config.num_hidden_layers),
        "--hidden-size", str(hf_config.hidden_size),
        "--num-attention-heads", str(hf_config.num_attention_heads),
        "--seq-length", "4096",
        "--max-position-embeddings", str(getattr(hf_config, 'max_position_embeddings', 4096)),
        "--micro-batch-size", "1",
        "--global-batch-size", "1",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--no-position-embedding",
        "--use-rotary-position-embeddings",
        "--rotary-percent", "1.0",
        "--rotary-base", str(getattr(hf_config, 'rope_theta', 10000)),
        "--num-query-groups", str(num_kv_heads),
        "--load", dcp_dir,
        "--no-load-optim",
        "--no-load-rng",
    ]

    args = parse_args(megatron_args)
    args.padded_vocab_size = hf_config.vocab_size

    # Set global variables
    set_global_variables(args)

    # Build model
    print("Building Megatron GPT model...")
    model = model_provider(gpt_builder, pre_process=True, post_process=True)

    # Load checkpoint
    print(f"Loading checkpoint from {dcp_dir}...")
    iteration = load_checkpoint([model], None, None)
    print(f"Loaded checkpoint at iteration {iteration}")

    return model



def convert_hf_to_megatron_state_dict(
    hf_model: torch.nn.Module,
    hf_config: PretrainedConfig,
):
    """
    Convert HuggingFace model state_dict to Megatron format.

    This function transforms parameter names and shapes from HuggingFace format
    to Megatron format, making it compatible with awex/converter/mcore_converter.py

    HuggingFace -> Megatron naming conversions:
    - model.embed_tokens.weight -> embedding.word_embeddings.weight
    - model.layers.X.self_attn.q_proj -> decoder.layers.X.self_attention.query_key_value (fused QKV)
    - model.layers.X.self_attn.o_proj -> decoder.layers.X.self_attention.dense
    - model.layers.X.mlp.gate_proj -> decoder.layers.X.mlp.dense_h_to_4h (gate+up fused)
    - model.layers.X.mlp.up_proj -> (fused with gate_proj)
    - model.layers.X.mlp.down_proj -> decoder.layers.X.mlp.dense_4h_to_h
    - model.norm.weight -> decoder.final_layernorm.weight
    - lm_head.weight -> output_layer.weight

    Args:
        hf_model: HuggingFace model instance
        hf_config: HuggingFace config

    Returns:
        Dict[str, torch.Tensor]: State dict in Megatron format
    """
    print("\nConverting HuggingFace state_dict to Megatron format...")

    hf_state_dict = hf_model.state_dict()
    megatron_state_dict = {}

    num_layers = hf_config.num_hidden_layers
    hidden_size = hf_config.hidden_size
    num_attention_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_attention_heads)
    head_dim = hidden_size // num_attention_heads

    print("Model architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")

    for name, param in hf_state_dict.items():
        new_name = None
        new_param = param

        # Embedding layer
        if name == "model.embed_tokens.weight":
            new_name = "embedding.word_embeddings.weight"

        # Layer-specific conversions
        elif "model.layers." in name:
            # Extract layer number
            parts = name.split(".")
            layer_idx = int(parts[2])

            # Attention QKV - need to fuse q_proj, k_proj, v_proj
            if "self_attn.q_proj" in name:
                # Collect Q, K, V weights
                q_weight = hf_state_dict[
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                ]
                k_weight = hf_state_dict[
                    f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                ]
                v_weight = hf_state_dict[
                    f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                ]

                # For GQA (Grouped Query Attention), K and V may have fewer heads
                # Megatron format: [num_heads * head_dim + 2 * num_kv_heads * head_dim, hidden_size]
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                new_name = (
                    f"decoder.layers.{layer_idx}.self_attention.query_key_value.weight"
                )
                new_param = qkv_weight

            # Skip k_proj and v_proj as they're fused with q_proj
            elif "self_attn.k_proj" in name or "self_attn.v_proj" in name:
                continue

            # Attention output projection
            elif "self_attn.o_proj" in name:
                new_name = f"decoder.layers.{layer_idx}.self_attention.dense.weight"

            # MLP gate and up projections - need to fuse
            elif "mlp.gate_proj" in name:
                gate_weight = hf_state_dict[
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                ]
                up_weight = hf_state_dict[
                    f"model.layers.{layer_idx}.mlp.up_proj.weight"
                ]
                # Megatron fuses gate and up: [2 * intermediate_size, hidden_size]
                gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
                new_name = f"decoder.layers.{layer_idx}.mlp.dense_h_to_4h.weight"
                new_param = gate_up_weight

            # Skip up_proj as it's fused with gate_proj
            elif "mlp.up_proj" in name:
                continue

            # MLP down projection
            elif "mlp.down_proj" in name:
                new_name = f"decoder.layers.{layer_idx}.mlp.dense_4h_to_h.weight"

            # Input LayerNorm
            elif "input_layernorm" in name:
                new_name = f"decoder.layers.{layer_idx}.input_layernorm.weight"

            # Post-attention LayerNorm
            elif "post_attention_layernorm" in name:
                new_name = f"decoder.layers.{layer_idx}.post_attention_layernorm.weight"

        # Final LayerNorm
        elif name == "model.norm.weight":
            new_name = "decoder.final_layernorm.weight"

        # Output layer (LM head)
        elif name == "lm_head.weight":
            new_name = "output_layer.weight"

        # Add converted parameter
        if new_name:
            megatron_state_dict[new_name] = new_param
            print(f"  {name} -> {new_name} | shape: {new_param.shape}")
        elif name not in [
            "model.layers",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.up_proj",
        ]:
            # Warn about unconverted parameters (except ones we intentionally skip)
            print(f"  WARNING: Skipped unconverted parameter: {name}")

    print("\nConversion complete:")
    print(f"  HuggingFace parameters: {len(hf_state_dict)}")
    print(f"  Megatron parameters: {len(megatron_state_dict)}")

    return megatron_state_dict
