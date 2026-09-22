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

"""Allocate portable CUDA IPC buffers without changing training allocations."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def cuda_ipc_allocation() -> Iterator[None]:
    """Use ordinary CUDA storage for new IPC buffers, then restore the allocator.

    Expandable IPC handles are not portable across all PyTorch versions. Scope
    this context to staging-buffer creation during serialized weight publishing,
    while training is paused: the allocator switch is process-wide. Existing
    training storage is untouched, and restoration precedes IPC serialization.
    """
    settings = torch.cuda.memory._snapshot()["allocator_settings"]
    if not settings["expandable_segments"]:
        yield
        return

    # Preserve runtime settings, not environment defaults. The setter resets
    # other options (including split size and GC threshold) on every call.
    original = settings["PYTORCH_CUDA_ALLOC_CONF"]
    # A preceding setter can omit this option while retaining its True value.
    # Replaying that string alone would leave our temporary False in effect.
    restore = original
    if not re.search(r"expandable_segments\s*:", original):
        restore = (
            f"{original},expandable_segments:True"
            if original
            else "expandable_segments:True"
        )
    staging = re.sub(r"expandable_segments\s*:\s*(True|False)", "", original)
    staging = ",".join(part for part in staging.split(",") if part.strip())
    staging = (
        f"{staging},expandable_segments:False"
        if staging
        else "expandable_segments:False"
    )
    try:
        torch.cuda.memory._set_allocator_settings(staging)
        yield
    finally:
        torch.cuda.memory._set_allocator_settings(restore)
