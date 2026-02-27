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

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

from transformers import PretrainedConfig

from awex import logging
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.vllm_converter import VLLMToHFWeightConverter
from awex.sharding.param_sharding import ShardingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    sharding_strategy: Callable[..., ShardingStrategy]
    mcore_converter: Optional[Callable[..., Any]]
    sglang_converter: Optional[Callable[..., Any]]


class _ModelRegistry:
    def __init__(self, models: Dict[str, ModelConfig]):
        self.models = models

    def get_registered_models(self):
        return dict(self.models)

    def get_model_config(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        else:
            logger.info(f"Model {model_name} not found, using default strategy.")
            return ModelConfig(
                sharding_strategy=ShardingStrategy,
                mcore_converter=None,
                sglang_converter=None,
            )


def _get_config_value(config, name: str, default=None):
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


@lru_cache
def import_model_configs():
    model_arch_name_to_config = {}
    package_name = "awex.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. {e}")
                continue
            if hasattr(module, "CONFIG"):
                entry = module.CONFIG
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        model_name = tmp["model_name"]
                        assert model_name not in model_arch_name_to_config, (
                            f"Duplicated model config for {model_name}"
                        )
                        model_arch_name_to_config[model_name] = tmp
                else:
                    model_name = entry["model_name"]
                    assert model_name not in model_arch_name_to_config, (
                        f"Duplicated model config for {model_name}"
                    )
                    model_arch_name_to_config[model_name] = entry

    return model_arch_name_to_config


ModelRegistry = _ModelRegistry(import_model_configs())


def get_sharding_strategy(model_name: str):
    config = ModelRegistry.get_model_config(model_name)
    return config.sharding_strategy


def _resolve_converter(
    config_value: Optional[Callable[..., Any]],
    default: Callable[..., Any],
):
    if config_value is None:
        return default
    if inspect.isclass(config_value):
        return config_value
    resolved = config_value()
    if not inspect.isclass(resolved):
        raise TypeError(f"Expected converter class from factory, got {type(resolved)}")
    return resolved


def get_train_weights_converter(
    engine_name: str,
    model_name: str,
    hf_config: PretrainedConfig,
    rank_info,
    infer_conf: Dict,
    tf_config=None,
):
    config = ModelRegistry.get_model_config(model_name)
    if engine_name == "mcore":
        from awex.converter.mcore_converter import McoreToHFWeightConverter

        converter = _resolve_converter(
            _get_config_value(config, "mcore_converter"),
            McoreToHFWeightConverter,
        )
        if tf_config is None:
            tf_config = (
                infer_conf.get("tf_config") if isinstance(infer_conf, dict) else None
            )
        if tf_config is None:
            raise ValueError("tf_config is required for McoreToHFWeightConverter")
        return converter(hf_config, rank_info, infer_conf, tf_config=tf_config)
    else:
        raise NotImplementedError(f"Engine {engine_name} not implemented.")


def get_infer_weights_converter(
    engine_name: str,
    model_name: str,
    hf_config: PretrainedConfig,
    rank_info,
    infer_engine_config: Dict,
):
    config = ModelRegistry.get_model_config(model_name)
    if engine_name == "sglang":
        converter = _resolve_converter(
            _get_config_value(config, "sglang_converter"),
            SGlangToHFWeightConverter,
        )
        return converter(hf_config, infer_engine_config, rank_info)
    if engine_name == "vllm":
        converter = _resolve_converter(
            _get_config_value(config, "vllm_converter"),
            VLLMToHFWeightConverter,
        )
        return converter(hf_config, infer_engine_config, rank_info)
    else:
        raise NotImplementedError(f"Engine {engine_name} not implemented.")
