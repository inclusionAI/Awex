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
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict

from transformers import PretrainedConfig

from awex import logging
from awex.converter.mcore_converter import McoreToHFWeightConverter
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.sharding.param_sharding import ShardingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    sharding_strategy: Callable[..., ShardingStrategy]
    mcore_converter: Callable[..., McoreToHFWeightConverter]
    sglang_converter: Callable[..., SGlangToHFWeightConverter]


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
                mcore_converter=McoreToHFWeightConverter,
                sglang_converter=SGlangToHFWeightConverter,
            )


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


def get_train_weights_converter(
    engine_name: str,
    model_name: str,
    hf_config: PretrainedConfig,
    rank_info,
    infer_conf: Dict,
):
    config = ModelRegistry.get_model_config(model_name)
    if engine_name == "mcore":
        converter = config.mcore_converter or McoreToHFWeightConverter
        return converter(hf_config, rank_info, infer_conf)
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
        converter = config.sglang_converter or SGlangToHFWeightConverter
        return converter(hf_config, infer_engine_config, rank_info)
    else:
        raise NotImplementedError(f"Engine {engine_name} not implemented.")
