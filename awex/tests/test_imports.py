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

"""Public inference imports must not depend on installed training frameworks."""

import subprocess
import sys


def test_public_imports_without_optional_backends():
    code = """
import importlib.abc
import sys

class NoBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'megatron', 'sglang', 'vllm', 'mooncake'}:
            raise ModuleNotFoundError(f'Backend is deliberately unavailable: {fullname}')

sys.meta_path.insert(0, NoBackends())
from awex import InferenceConfig
assert 'torch' not in sys.modules
from awex import WeightsReader, NCCLWorkerWeightsReader, NCCLWeightsWriter
from awex.engine.sglang import SGLangEngine
from awex.engine.vllm import VLLMEngine
from awex.models.registry import ModelRegistry
from awex.models.ling_linear import CONFIG
for entry in CONFIG if isinstance(CONFIG, list) else [CONFIG]:
    assert entry['model_name'] in ModelRegistry.get_registered_models()
assert not any(name == 'megatron' or name.startswith('megatron.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
