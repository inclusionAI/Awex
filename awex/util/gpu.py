import subprocess

from awex import logging

import torch

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
    logger.info(f"Device gpu memory status for [{stage}]: torch allocated {pretty_bytes(allocated)}, "
                f"torch reserved {pretty_bytes(reserved)} "
                f"device mem_free {pretty_bytes(mem_free)}, device occupy {pretty_bytes(occupy)}")
