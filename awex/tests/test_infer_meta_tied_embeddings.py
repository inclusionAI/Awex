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

"""Tied-embedding lm_head alias in the inference parameter metadata."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from awex.meta.infer_meta_resolver import InferParamMetaResolver


class Qwen3ForCausalLM:
    def __init__(self, config, params):
        self.config = config
        self._params = params

    def named_parameters(self):
        return iter(self._params)


def _model(tie_word_embeddings=True):
    config = SimpleNamespace(
        tie_word_embeddings=tie_word_embeddings,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        hidden_size=16,
        architectures=["Qwen3ForCausalLM"],
    )
    params = [("model.embed_tokens.weight", torch.randn(32, 16))]
    return Qwen3ForCausalLM(config, params)


def _resolve(engine_name, model):
    identity = SimpleNamespace(convert_param=lambda name, param: [(name, param)])
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "awex.meta.infer_meta_resolver.get_rank_info_extractor",
                return_value=lambda ctx, rank: SimpleNamespace(
                    tp_rank=0, ep_rank=0, global_rank=0
                ),
            )
        )
        stack.enter_context(
            patch(
                "awex.meta.infer_meta_resolver.get_infer_weights_converter",
                return_value=identity,
            )
        )
        meta = InferParamMetaResolver._get_model_param_info(
            engine_name,
            SimpleNamespace(tp_size=1, ep_size=1, device_backend="cuda"),
            convert_params=True,
            engine_rank=0,
            model=model,
            model_context={"pp_rank": 0, "pp_size": 1},
        )
    return {entry["name"] for entry in meta["params_meta"]}


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
def test_tied_embeddings_get_an_lm_head_alias_on_every_engine(engine_name):
    """The reader adds the same alias, so the metadata has to match it.

    Without the alias the transfer plan reports lm_head.weight as missing,
    because the training side publishes it while the inference side ties the
    output weights to the embedding.
    """
    assert "lm_head.weight" in _resolve(engine_name, _model())


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
def test_untied_embeddings_get_no_alias(engine_name):
    names = _resolve(engine_name, _model(tie_word_embeddings=False))
    assert "lm_head.weight" not in names
