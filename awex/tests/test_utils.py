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

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig


def megatron_model_from_hf(
    model_path: str = "Qwen/Qwen2-1.5B",
) -> Tuple[torch.nn.Module, PretrainedConfig]:
    """
    Load model from HuggingFace and prepare it for Megatron-style weight conversion.

    This is a simple approach that:
    1. Loads model using HuggingFace transformers (no Megatron initialization)
    2. Returns model with state_dict that can be converted to Megatron format
    3. Attaches converter function for use with awex/converter/mcore_converter.py

    Args:
        model_path: HuggingFace model path (default: Qwen/Qwen2-1.5B)

    Returns:
        Tuple of (hf_model, hf_config)
        The model has attached converter: model.convert_to_megatron_state_dict()

    Note:
        This function does NOT initialize Megatron. It only loads HF model.
        For testing weights exchange between Megatron and SGLang.
    """
    print(f"Loading model from HuggingFace: {model_path}")

    # Load config
    hf_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print("Config loaded:")
    print(f"  Model type: {hf_config.model_type}")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num attention heads: {hf_config.num_attention_heads}")
    print(
        f"  Num KV heads: {getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)}"
    )
    print(f"  Vocab size: {hf_config.vocab_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Load to CPU (no GPU required)
    )

    model = model.cpu()

    print("Model loaded successfully:")
    print(f"  Model class: {type(model).__name__}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Attach converter function to model
    model = convert_hf_to_megatron_state_dict(model, hf_config)
    return model, hf_config


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
