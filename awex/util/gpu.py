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
from awex.util import device as device_util
from awex.util.common import pretty_bytes

logger = logging.getLogger(__name__)


def get_gpu_status() -> str:
    """Get accelerator status information in CSV format."""
    if device_util.get_device_type() == "npu":
        try:
            return subprocess.check_output(["npu-smi", "info"], text=True)
        except subprocess.CalledProcessError as e:
            return f"Failed to get NPU status via npu-smi: {e}"
        except FileNotFoundError:
            return "npu-smi not found; NPU status unavailable."
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        return f"Failed to get GPU status via nvidia-smi: {e}"
    except FileNotFoundError:
        return "nvidia-smi not found; GPU status unavailable."


def print_gpu_status(stage):
    logger.info(f"GPU status for {stage}:\n{get_gpu_status()}")


def print_current_gpu_status(stage):
    device_type = device_util.get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            logger.info(
                f"Device npu memory status for [{stage}]: torch.npu unavailable"
            )
            return
        allocated = npu_mod.memory_allocated()
        reserved = npu_mod.memory_reserved()
        mem_free, mem_total = npu_mod.mem_get_info()
    else:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        mem_free, mem_total = torch.cuda.mem_get_info()
    occupy = mem_total - mem_free
    logger.info(
        f"Device {device_type} memory status for [{stage}]: torch allocated {pretty_bytes(allocated)}, "
        f"torch reserved {pretty_bytes(reserved)} "
        f"device mem_free {pretty_bytes(mem_free)}, device occupy {pretty_bytes(occupy)}"
    )
