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

import time
from typing import Any, Dict, List, Optional, Union

from awex import logging
from awex.config import InferenceConfig
from awex.engine.core import InferenceEngine
from awex.reader.weights_reader import get_weights_exchange_reader

logger = logging.getLogger(__name__)


_VLLM_TASK_SIGNATURES = {
    # Example/template: fill in supported task names and the allowed kwargs.
    # "awex_execute": {
    #     "required": ["task_module", "task_qualname", "task_kwargs"],
    #     "optional": [],
    # },
}


class VLLMEngine(InferenceEngine):
    def __init__(
        self,
        config: Union[Dict[str, Any], InferenceConfig],
        vllm_engine,
        hf_config=None,
    ):
        if isinstance(config, dict):
            config = InferenceConfig.from_dict(config)
        if hf_config is None:
            if config.model_path is None:
                raise ValueError("hf_config or config.model_path must be provided.")
            hf_config = _load_hf_config(config.model_path)
        super().__init__(hf_config)
        self._config = config
        self._vllm_engine = vllm_engine
        self.node_rank = config.node_rank or 0
        self.released_tags = set()
        self.weights_exchange_reader = None
        self.rank_coordinate = f"{config.engine_rank}-{self.node_rank}"
        self._initialized = False

    @property
    def engine_name(self):
        return "vllm"

    @property
    def config(self):
        return self._config

    def initialize(self) -> None:
        if not getattr(self._vllm_engine, "initialized", False):
            self._vllm_engine.initialize()
        if self.config.node_rank == 0:
            logger.info(
                f"Start to initialize weights exchange reader for {self.rank_coordinate}"
            )
            self._initialized = True
            self.weights_exchange_reader = get_weights_exchange_reader(self)
            self.weights_exchange_reader.initialize()
            logger.info(
                f"Finished initializing weights exchange reader for {self.rank_coordinate}"
            )
        else:
            logger.info(
                f"Skip initializing weights exchange reader for {self.rank_coordinate}"
            )

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str] = None
    ):
        if load_format is not None:
            logger.warning(
                "vLLM remote update does not support load_format; ignoring %s",
                load_format,
            )
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        logger.info(
            f"Start to update weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}"
        )
        self._vllm_engine.call_utility(
            "awex_update_weights_from_disk",
            args=[model_path, load_format],
            kwargs={},
            run_on_all=True,
        )
        logger.info(
            f"Finished updating weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}"
        )

    def update_weights(self, **kwargs):
        logger.info(
            f"Start to update weights for step {self.global_step} for {self.rank_coordinate}"
        )
        start_time = time.time()
        self.weights_exchange_reader.update_weights(step_id=self.global_step, **kwargs)
        duration = time.time() - start_time
        logger.info(
            f"Finished updating weights for step {self.global_step} for {self.rank_coordinate}, "
            f"took {duration:.3f} seconds"
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        if self._initialized and self.node_rank == 0:
            if set(tags) - self.released_tags != set(tags):
                tags = list(set(tags) - self.released_tags)
            self.released_tags.update(tags)
            if not tags:
                logger.info("No memory occupation to release")
                return
            logger.info(f"Start to release memory occupation {tags}")
            self._vllm_engine._engine.offload()
            logger.info("Finished releasing memory occupation")

    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        if self._initialized and self.node_rank == 0:
            tags = list(self.released_tags & set(tags))
            self.released_tags.difference_update(tags)
            if not tags:
                logger.info("No memory occupation to resume")
                return
            logger.info(f"Start to resume memory occupation {tags}")
            self._vllm_engine._engine.onload(tags=tags)
            logger.info("Finished resuming memory occupation")

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        if self.node_rank != 0:
            raise RuntimeError(
                f"Non-zero rank node {self.rank_coordinate} is not allowed to "
                f"execute task in model workers"
            )
        run_on_all = kwargs.pop("run_on_all", True)
        if isinstance(fn, str):
            method = fn
            payload = kwargs
        else:
            method = "awex_execute"
            payload = {
                "task_module": fn.__module__,
                "task_qualname": fn.__qualname__,
                "task_kwargs": kwargs,
            }
        payload = _adapt_task_kwargs(method, payload)
        responses = self._vllm_engine.call_utility(
            method, args=[], kwargs=payload, run_on_all=run_on_all
        )
        if run_on_all:
            results = []
            for response in responses:
                results.extend(response.get("results", []))
            return results
        return responses.get("results", [])

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank


def _load_hf_config(model_path: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def _adapt_task_kwargs(method: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt Awex task kwargs to the vLLM utility surface."""
    signature = _VLLM_TASK_SIGNATURES.get(method)
    if signature is None:
        return kwargs
    required = signature.get("required", [])
    optional = signature.get("optional", [])
    allowed = set(required) | set(optional)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    missing = [k for k in required if k not in filtered]
    if missing:
        raise ValueError(f"Missing required args for {method}: {missing}")
    return filtered
