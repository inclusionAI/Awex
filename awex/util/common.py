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

import json
import math
import os
import pickle
import socket
import struct
from enum import Enum
from typing import List

import torch

from awex import logging

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, force=True):
    logging.basicConfig(
        level=level,
        format="%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(process)d -- %(message)s",
        force=force,
    )


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def to_binary(data):
    # Serialize messages using pickle
    pickled_data = pickle.dumps(data)
    # Get the length of the pickled data
    data_len = len(pickled_data)
    # Create the binary response: length of data (4 bytes) + pickled data
    return struct.pack("!I", data_len) + pickled_data


def from_binary(binary):
    # Extract the length of the pickled data
    data_len = struct.unpack("!I", binary[:4])[0]
    # Extract and unpickle the data
    pickled_data = binary[4 : 4 + data_len]
    data = pickle.loads(pickled_data)
    return data


def to_dict(param_meta, ignore_keys=None) -> dict:
    """Convert the parameter meta to a dict."""
    ignore_keys = ignore_keys or set()

    def convert_value(v):
        if isinstance(v, Enum):
            return v.value  # Handle enums
        if isinstance(v, (tuple, list)):
            return [convert_value(x) for x in v]
        if isinstance(v, torch.dtype):
            return str(v)  # Handle torch.dtype
        if isinstance(v, slice):
            return str(v)  # Handle slice
        if isinstance(v, dict):
            return {k: convert_value(v) for k, v in v.items() if k not in ignore_keys}
        if hasattr(v, "__dict__"):
            return {
                k: convert_value(v)
                for k, v in v.__dict__.items()
                if not k.startswith("_") and k not in ignore_keys
            }
        if hasattr(v, "__slots__"):
            return {
                k: convert_value(getattr(v, k))
                for k in v.__slots__
                if not k.startswith("_") and k not in ignore_keys
            }
        return v

    param_dict = convert_value(param_meta)
    return param_dict


def to_json(param_meta, ignore_keys=None) -> str:
    """Convert the parameter meta to a json string."""
    return json.dumps(to_dict(param_meta, ignore_keys), indent=2)


def compute_statistics(stage_history: dict, step_id: int, duration: float, stage: str):
    if stage not in stage_history:
        stage_history[stage] = []
    history = stage_history[stage]
    history.append(duration)
    if len(history) > 10000:
        history.pop(0)
    if step_id == 2:
        # first step contains init time
        history.pop(history.index(max(history)))
    num_updates = len(history)
    stage_history[stage] = history = sorted(history)
    avg_time = sum(history) / num_updates
    median_time = history[num_updates // 2]
    max_time = history[-1]
    min_time = history[0]
    logger.info(
        f"{stage} time statistics for step {step_id}: average time: {avg_time:.4f} seconds, median time: {median_time:.4f} seconds, "
        f"min time: {min_time:.4f} seconds,  max time: {max_time:.4f} seconds"
    )


def check_train_infer_params_meta(
    training_params_meta: List,
    infer_parameters_meta: List,
    raise_exception: bool = False,
):
    infer_meta = {param_meta.name: param_meta for param_meta in infer_parameters_meta}
    train_meta = {param_meta.name: param_meta for param_meta in training_params_meta}
    common_params = set(infer_meta.keys()) & set(train_meta.keys())
    if len(common_params) != len(infer_meta) or len(common_params) != len(train_meta):
        if len(train_meta) > len(infer_meta):
            diff = set(train_meta.keys()) - common_params
        else:
            diff = set(infer_meta.keys()) - common_params
        logger.error(
            f"Inconsistent parameters meta: "
            f"train {len(train_meta)} infer {len(infer_meta)} diff keys {diff}"
        )
        if raise_exception:
            raise ValueError(
                f"Inconsistent parameters meta for inference and training: "
                f"{len(common_params)} {len(infer_meta)} {len(train_meta)}, diff keys {diff}"
            )
    for param_name in common_params:
        infer_param_meta = infer_meta[param_name]
        train_param_meta = train_meta[param_name]
        if infer_param_meta.global_numel != train_param_meta.global_numel:
            error_msg = (
                f"Inconsistent number of elements for parameter {param_name}: "
                f"{infer_param_meta.global_numel} != {train_param_meta.global_numel}"
            )
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
        if infer_param_meta.global_shape != train_param_meta.global_shape:
            error_msg = f"Inconsistent shape for parameter {param_name}: {infer_param_meta.global_shape} != {train_param_meta.global_shape}"
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
        if infer_param_meta.dtype != train_param_meta.dtype:
            error_msg = f"Inconsistent dtype for parameter {param_name}: {infer_param_meta.dtype} != {train_param_meta.dtype}"
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
        infer_tp_size = len(infer_param_meta.replicas[0].shards)
        train_tp_size = len(train_param_meta.replicas[0].shards)
        if infer_tp_size < train_tp_size or infer_tp_size % train_tp_size != 0:
            error_msg = (
                f"Inference for parameter {param_name} has wrong tp_size: "
                f"infer {infer_tp_size} train {train_tp_size}"
            )
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)


def pretty_bytes(size_bytes):
    if size_bytes == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, index)
    converted_size = round(size_bytes / power, 2)
    return f"{converted_size} {units[index]}"


def stripped_env_vars():
    vars = {}
    for k, v in os.environ.items():
        if "secret" not in k.lower():
            vars[k] = v
    vars.pop("LS_COLORS", None)
    return vars


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure that attributes refer to dictionary keys
        self.__dict__ = self


def simple_hf_config(hg_config):
    config = hg_config.to_dict()
    final_config = {}
    for k, v in config.items():
        if "sglang" in str(v):
            logger.warning(f"Skipping sglang config {k}: {v}")
            continue
        final_config[k] = v
    return AttrDict(**config)
