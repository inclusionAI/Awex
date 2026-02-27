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

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def _get_npu_module():
    try:
        import torch_npu  # type: ignore

        return torch_npu
    except Exception:
        return None


def is_npu_available() -> bool:
    torch_npu = _get_npu_module()
    if torch_npu is None:
        return False
    npu_mod = getattr(torch, "npu", None) or getattr(torch_npu, "npu", None)
    if npu_mod is None:
        return False
    try:
        return npu_mod.is_available()
    except Exception:
        return False


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device_type() -> str:
    override = os.environ.get("AWEX_DEVICE_TYPE", "").strip().lower()
    if override:
        return override
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def device_count() -> int:
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        return 0 if npu_mod is None else npu_mod.device_count()
    if device_type == "cuda":
        return torch.cuda.device_count()
    return 0


def current_device() -> int:
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        return 0 if npu_mod is None else npu_mod.current_device()
    if device_type == "cuda":
        return torch.cuda.current_device()
    return 0


def set_device(device_id: int) -> None:
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            raise RuntimeError("torch.npu is not available; cannot set NPU device.")
        npu_mod.set_device(device_id)
        return
    if device_type == "cuda":
        torch.cuda.set_device(device_id)


def synchronize(device_id: int | None = None) -> None:
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            return
        if device_id is None:
            npu_mod.synchronize()
        else:
            npu_mod.synchronize(device=device_id)
        return
    if device_type == "cuda":
        torch.cuda.synchronize(device=device_id)


def get_device_name(device_id: int | None = None) -> str:
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            return "npu"
        idx = current_device() if device_id is None else device_id
        return npu_mod.get_device_name(idx)
    if device_type == "cuda":
        idx = current_device() if device_id is None else device_id
        return torch.cuda.get_device_name(idx)
    return "cpu"


def get_torch_device(device_id: int | None = None) -> torch.device:
    device_type = get_device_type()
    if device_type in {"cuda", "npu"}:
        idx = current_device() if device_id is None else device_id
        return torch.device(f"{device_type}:{idx}")
    return torch.device("cpu")


def get_device_properties(device_id: int | None = None):
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            torch_npu = _get_npu_module()
            if torch_npu is None or getattr(torch_npu, "npu", None) is None:
                raise RuntimeError(
                    "torch.npu is not available; cannot get NPU properties."
                )
            npu_mod = torch_npu.npu
        idx = current_device() if device_id is None else device_id
        if hasattr(npu_mod, "get_device_properties"):
            return npu_mod.get_device_properties(idx)
        raise RuntimeError("torch.npu.get_device_properties is not available.")
    if device_type == "cuda":
        idx = current_device() if device_id is None else device_id
        return torch.cuda.get_device_properties(idx)
    raise RuntimeError("Device properties only available for CUDA/NPU.")


def visible_devices_env_names() -> list[str]:
    device_type = get_device_type()
    if device_type == "npu":
        return ["ASCEND_RT_VISIBLE_DEVICES"]
    return ["CUDA_VISIBLE_DEVICES"]


def visible_devices_env_value() -> str:
    for name in visible_devices_env_names():
        value = os.environ.get(name)
        if value:
            return value
    return ""


def get_stream_class() -> type | None:
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        return None if npu_mod is None else npu_mod.Stream
    if device_type == "cuda":
        return torch.cuda.Stream
    return None


def create_stream(device_id: int | None = None):
    stream_cls = get_stream_class()
    if stream_cls is None:
        return None
    if device_id is None:
        return stream_cls()
    return stream_cls(device=device_id)


@contextmanager
def stream(stream_obj) -> Iterator[None]:
    if stream_obj is None:
        yield
        return
    device_type = get_device_type()
    if device_type == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            yield
        else:
            with npu_mod.stream(stream_obj):
                yield
        return
    if device_type == "cuda":
        with torch.cuda.stream(stream_obj):
            yield
        return
    yield
