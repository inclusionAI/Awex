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

"""Mooncake data transfers, with AWEX metadata used only for rendezvous."""

import os
from typing import Callable, Dict, Optional

import torch

from awex.meta.meta_server import MetaServerClient
from awex.transfer.transfer_plan import TransferPlan, slice_tensor
from awex.util import device as device_util
from awex.util.common import get_ip_address


def _op_signature(op):
    return (
        op.send_shard_meta.name,
        op.recv_shard_meta.name,
        tuple(op.send_offset),
        tuple(op.recv_offset),
        tuple(op.overlap_shape),
        tuple((s.start, s.stop, s.step) for s in op.train_slices),
        tuple((s.start, s.stop, s.step) for s in op.inf_slices),
    )


class MooncakeTransport:
    """Keep exported buffers alive until every reader acknowledges completion.

    A failed exchange must abort the job. In particular, a metadata timeout does
    not prove that remote DMA has stopped: retain registered buffers on failure
    and reject subsequent exchanges instead of allowing their storage to be reused.
    """

    def __init__(self, meta_client: MetaServerClient, rank: int, timeout: float):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise ImportError(
                "Mooncake requires the optional dependency: pip install 'awex[mooncake]'"
            ) from exc
        self.protocol = os.environ.get("AWEX_MOONCAKE_PROTOCOL", "rdma")
        if self.protocol not in ("rdma", "tcp"):
            raise ValueError("AWEX_MOONCAKE_PROTOCOL must be rdma or tcp")
        if device_util.get_device_type() not in ("cpu", "cuda"):
            raise ValueError("Mooncake weight exchange supports CPU and CUDA devices")
        self.engine = TransferEngine()
        host = os.environ.get("AWEX_MOONCAKE_HOST") or get_ip_address()
        self._check(
            self.engine.initialize(
                host,
                "P2PHANDSHAKE",
                self.protocol,
                os.environ.get("AWEX_MOONCAKE_DEVICE", ""),
            ),
            "initialize",
        )
        self.endpoint = f"{host}:{self.engine.get_rpc_port()}"
        self.meta_client = meta_client
        self.rank = rank
        self.timeout = timeout
        self.sequence = 0
        self.failed = False
        self.buffers = {}

    @staticmethod
    def _check(result, operation):
        if result != 0:
            raise RuntimeError(f"Mooncake {operation} failed with code {result}")

    def _begin(self, step_id):
        if self.failed or self.buffers:
            raise RuntimeError("Previous Mooncake exchange failed; restart the job")
        # Include an invocation counter so repeated updates at the same training
        # step cannot consume a previous exchange's descriptors or acknowledgments.
        key = f"mooncake_{step_id}_{self.sequence}"
        self.sequence += 1
        return key

    def _register(self, tensor):
        address = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        if size:
            self._check(self.engine.register_memory(address, size), "register_memory")
            self.buffers[address] = tensor
        return address, size

    def _release(self):
        for address in list(self.buffers):
            self._check(self.engine.unregister_memory(address), "unregister_memory")
            del self.buffers[address]

    @torch.no_grad()
    def send(
        self,
        parameters: Dict[str, torch.Tensor],
        plan: TransferPlan,
        step_id: int,
        *,
        offload: bool = False,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        """Snapshot and consume converted parameters, then serve all peer reads."""
        prefix = self._begin(step_id)
        published = False
        try:
            descriptors = {}
            slices = {}
            for peer, operations in sorted(plan.operations.items()):
                entries = []
                for op in operations:
                    cache_key = (
                        op.send_shard_meta.name,
                        tuple((s.start, s.stop, s.step) for s in op.train_slices),
                    )
                    if cache_key not in slices:
                        source = slice_tensor(
                            parameters[op.send_shard_meta.name], op, True
                        )
                        target_device = (
                            "cpu"
                            if offload or self.protocol == "tcp"
                            else source.device
                        )
                        # Own the exact allocation registered with Mooncake. Views
                        # can overlap, and colocated model weights are offloaded
                        # before inference resumes its allocations.
                        tensor = source.to(device=target_device, copy=True).contiguous()
                        del source
                        address, size = self._register(tensor)
                        slices[cache_key] = (
                            address,
                            size,
                            tuple(tensor.shape),
                            str(tensor.dtype),
                        )
                    entries.append((_op_signature(op), *slices[cache_key]))
                descriptors[peer] = entries
            device_util.synchronize()
            # Drop converted tensors before the colocate callback offloads model
            # weights. Registered snapshots remain owned by this transport.
            parameters.clear()
            if on_ready is not None:
                on_ready()
            for peer, entries in descriptors.items():
                # A failed HTTP response may still have published the addresses.
                published = True
                self.meta_client.put_object(
                    f"{prefix}_{self.rank}_{peer}_buffers", (self.endpoint, entries)
                )
            for peer in descriptors:
                key = f"{prefix}_{self.rank}_{peer}"
                error = self.meta_client.get_object(f"{key}_done", timeout=self.timeout)
                if error:
                    raise RuntimeError(f"Mooncake reader {peer} failed: {error}")
            self._release()
            for peer in descriptors:
                key = f"{prefix}_{self.rank}_{peer}"
                self.meta_client.delete_if_exists(f"{key}_buffers")
                self.meta_client.delete_if_exists(f"{key}_done")
        except Exception:
            self.failed = True
            if not published:
                self._release()
            raise

    @torch.no_grad()
    def receive(
        self, parameters: Dict[str, torch.Tensor], plan: TransferPlan, step_id: int
    ) -> None:
        prefix = self._begin(step_id)
        try:
            for peer, operations in sorted(plan.operations.items()):
                key = f"{prefix}_{peer}_{self.rank}"
                endpoint, entries = self.meta_client.get_object(
                    f"{key}_buffers", timeout=self.timeout
                )
                try:
                    if len(entries) != len(operations):
                        raise ValueError(
                            "Mooncake transfer plan operation count mismatch"
                        )
                    for op, entry in zip(operations, entries):
                        signature, address, size, shape, dtype = entry
                        target = slice_tensor(
                            parameters[op.recv_shard_meta.name], op, False
                        )
                        if (
                            signature != _op_signature(op)
                            or tuple(target.shape) != shape
                            or str(target.dtype) != dtype
                            or target.numel() * target.element_size() != size
                        ):
                            raise ValueError("Mooncake transfer plan tensor mismatch")
                        if not size:
                            continue
                        # Staging supports strided slices and transposed converter
                        # views; TCP uses host buffers on both sides.
                        buffer = torch.empty(
                            shape,
                            dtype=target.dtype,
                            device="cpu" if self.protocol == "tcp" else target.device,
                        )
                        local_address, _ = self._register(buffer)
                        device_util.synchronize()
                        self._check(
                            self.engine.transfer_sync_read(
                                endpoint, local_address, address, size
                            ),
                            "transfer_sync_read",
                        )
                        target.copy_(buffer)
                        # The copy must finish before unregistering or reusing the
                        # staging allocation, including CUDA's asynchronous copies.
                        device_util.synchronize()
                        self._release()
                    self.meta_client.put_object(f"{key}_done", "")
                except Exception as exc:
                    self.meta_client.put_object(f"{key}_done", str(exc))
                    raise
        except Exception:
            self.failed = True
            raise
