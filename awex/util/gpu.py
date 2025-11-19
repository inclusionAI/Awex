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

import subprocess

import torch

from awex import logging
from awex.util.common import pretty_bytes

logger = logging.getLogger(__name__)


def get_gpu_status() -> str:
    """Get GPU status information in CSV format.

    Returns:
        str: GPU status information including name, utilization, and memory usage
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv",
            ],
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Failed to get GPU status: {e}")
        raise e


def print_gpu_status(stage):
    logger.info(f"GPU status for {stage}:\n{get_gpu_status()}")


def print_current_gpu_status(stage):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    mem_free, mem_total = torch.cuda.mem_get_info()
    occupy = mem_total - mem_free
    logger.info(
        f"Device gpu memory status for [{stage}]: torch allocated {pretty_bytes(allocated)}, "
        f"torch reserved {pretty_bytes(reserved)} "
        f"device mem_free {pretty_bytes(mem_free)}, device occupy {pretty_bytes(occupy)}"
    )
