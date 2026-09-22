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

"""Exact metadata boundaries for Qwen4Exp immutable inference state.

This declaration does not prove tensor values or lifecycle preservation. Runtime
integration must additionally bind the checkpoint evidence and visual backup,
validate original parameters before each exchange, and use the same declaration
for metadata and payload converters. No automatic registration or exclusion is
performed here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn


def mcore_visual_parameter_name(name: str, contract: Qwen4ExpFrozenContract) -> str:
    """Map the replicated ModelScope HF vision tower to receiver identities."""
    while name.startswith("module."):
        name = name[len("module.") :]
    if not name.startswith("visual.visual."):
        raise ValueError(f"Unsupported Qwen4Exp actor visual parameter: {name}")
    canonical = "model.visual." + name[len("visual.visual.") :]
    if canonical not in contract.visual_parameter_names:
        canonical = canonical.replace(".attn.qkv.", ".attn.qkv_proj.")
    if canonical not in contract.visual_parameter_names:
        raise ValueError(f"Actor visual parameter is outside frozen contract: {name}")
    return canonical


@dataclass(frozen=True)
class Qwen4ExpFrozenContract:
    checkpoint_manifest_sha256: str
    ple_table_names: frozenset[str]
    visual_parameter_names: frozenset[str]
    language_model_only: bool
    freeze_ple_table: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("Unsupported Qwen4Exp frozen contract schema")
        if (
            type(self.language_model_only) is not bool
            or self.freeze_ple_table is not True
        ):
            raise ValueError(
                "Frozen exclusions require an explicit model mode and frozen PLE"
            )
        if self.schema_version == 1 and not self.language_model_only:
            raise ValueError("Vision actors require frozen contract schema 2")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_manifest_sha256):
            raise ValueError("Expected a SHA256 checkpoint manifest identity")
        for names in (self.ple_table_names, self.visual_parameter_names):
            if not isinstance(names, frozenset) or not names:
                raise ValueError("Frozen parameter names must be nonempty frozen sets")
        for name in self.ple_table_names:
            if not re.fullmatch(
                r"model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.weight",
                name,
            ):
                raise ValueError(f"Invalid frozen PLE table name: {name}")
        for name in self.visual_parameter_names:
            if not re.fullmatch(
                r"model\.visual\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", name
            ):
                raise ValueError(f"Invalid frozen visual parameter name: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "language_model_only": self.language_model_only,
            "freeze_ple_table": self.freeze_ple_table,
            "ple_table_names": sorted(self.ple_table_names),
            "visual_parameter_names": sorted(self.visual_parameter_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Qwen4ExpFrozenContract:
        data = dict(payload)
        expected = {
            "schema_version",
            "checkpoint_manifest_sha256",
            "language_model_only",
            "freeze_ple_table",
            "ple_table_names",
            "visual_parameter_names",
        }
        if data.keys() != expected:
            raise ValueError("Missing or unexpected Qwen4Exp frozen contract fields")
        for key in ("ple_table_names", "visual_parameter_names"):
            names = data[key]
            if not isinstance(names, list) or not all(
                isinstance(n, str) for n in names
            ):
                raise ValueError(f"Expected a string list for {key}")
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate names in {key}")
            data[key] = frozenset(names)
        return cls(**data)

    def excludes(self, name: str, side: Literal["actor", "inference"]) -> bool:
        if side not in ("actor", "inference"):
            raise ValueError(f"Unknown contract side: {side}")
        return name in self.ple_table_names or (
            (side == "inference" or not self.language_model_only)
            and name in self.visual_parameter_names
        )

    @staticmethod
    def _validate_table(name: str, parameter: nn.Parameter) -> None:
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Validate original Parameter objects, not detached tensors: {name}"
            )
        if parameter.requires_grad:
            raise ValueError(f"PLE table is trainable: {name}")
        if parameter.dtype != torch.bfloat16 or parameter.ndim != 2:
            raise ValueError(f"Expected the validated BF16 PLE table layout: {name}")

    def validate_actor_parameters(
        self,
        parameters: Mapping[str, nn.Parameter],
        local_table_names: frozenset[str],
        local_visual_names: frozenset[str] | None = None,
    ) -> None:
        """Validate canonical original parameters on this PP stage, before detach.

        The caller must verify global PP ownership coverage separately; a PP stage
        without PLE legitimately has an empty local table set.
        """
        if not local_table_names <= self.ple_table_names:
            raise ValueError("Local PLE ownership is outside the frozen contract")
        observed_visual = frozenset(
            n for n in parameters if n.startswith("model.visual.")
        )
        if self.language_model_only and observed_visual:
            raise ValueError(
                "Language-only actor unexpectedly contains visual parameters"
            )
        if not self.language_model_only:
            if local_visual_names not in (frozenset(), self.visual_parameter_names):
                raise ValueError("Explicit complete local visual ownership is required")
            if observed_visual != local_visual_names:
                raise ValueError("Actor visual parameters do not match local ownership")
            for name in observed_visual:
                parameter = parameters[name]
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        f"Validate original visual Parameter, not detached tensor: {name}"
                    )
                if parameter.requires_grad:
                    raise ValueError(f"Frozen visual parameter is trainable: {name}")
                if parameter.dtype != torch.bfloat16:
                    raise ValueError(f"Expected BF16 frozen visual parameter: {name}")
        observed = {name for name in parameters if ".ple_embedding." in name}
        if observed != local_table_names:
            raise ValueError(
                "Actor PLE parameters do not match declared local ownership"
            )
        for name in local_table_names:
            self._validate_table(name, parameters[name])

    def validate_inference_parameters(
        self,
        parameters: Mapping[str, nn.Parameter],
        preserved_visual_names: frozenset[str],
    ) -> None:
        """Require exact exclusions and the same keys in the visual backup path."""
        observed_visual = {n for n in parameters if n.startswith("model.visual.")}
        if observed_visual != self.visual_parameter_names:
            raise ValueError(
                "Inference visual parameters differ from the frozen contract"
            )
        if preserved_visual_names != self.visual_parameter_names:
            raise ValueError("Visual preservation keys differ from transfer exclusions")
        observed_tables = {n for n in parameters if ".ple_embedding." in n}
        if observed_tables != self.ple_table_names:
            raise ValueError("Inference PLE parameters differ from the frozen contract")
        for name in self.ple_table_names:
            self._validate_table(name, parameters[name])
            if parameters[name].device.type != "cpu":
                raise ValueError(f"Expected the validated CPU PLE residency: {name}")
