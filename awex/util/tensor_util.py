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

import io
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Dict, Iterable, List, Tuple, Union

import torch
from packaging import version
from torch.multiprocessing import reductions

from awex import logging

mp.current_process().authkey = b"ipc_serialize_cpu_tensor_authkey"

logger = logging.getLogger(__name__)


def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""

    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified

    reductions.init_reductions()


# The signature has not been changed for years, and we will not need this when the next version is released,
# so it looks safe to use a constant.
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(
        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
    )
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    raw_args = args
    try:
        args = _modify_tuple(
            args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid
        )
        return reductions._rebuild_cuda_tensor_original(*args)
    except Exception as e:
        msg = f"Error rebuilding cuda tensor: {e}, raw_args: {raw_args}, args: {args}"
        logger.exception(msg)
        raise RuntimeError(msg) from e


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):
    return *t[:index], modifier(t[index]), *t[index + 1 :]


def monkey_patch_torch_compile():
    if version.parse(torch.__version__) < version.parse("2.8.0"):
        # These things are cacheable by torch.compile. torch.compile just doesn't know it.
        # This was fixed in PyTorch 2.8, but until then, we monkey patch.
        import torch._higher_order_ops.auto_functionalize as af

        af.auto_functionalized_v2._cacheable = True
        af.auto_functionalized._cacheable = True


def validate_tensor_for_ipc(tensor: torch.Tensor) -> bool:
    """
    Validate if a tensor is suitable for IPC serialization.

    Args:
        tensor: The tensor to validate

    Returns:
        bool: True if tensor is valid for IPC serialization
    """
    try:
        # Check if tensor is valid
        if tensor is None:
            return False

        # Check if tensor has elements
        if tensor.numel() == 0:
            return False

        # Check if tensor is on CUDA
        if not tensor.is_cuda:
            return False

        # Check if tensor is contiguous
        if not tensor.is_contiguous():
            return False

        # Check if tensor has valid storage
        if tensor.storage().size() == 0:
            return False

        # Check if tensor has valid data pointer
        if tensor.data_ptr() == 0:
            return False

        return True
    except Exception:
        return False


def ipc_serialize(value) -> bytes:
    buf = io.BytesIO()
    ForkingPickler(buf).dump(value)
    buf.seek(0)
    output = buf.read()
    return output


def ipc_deserialize(data: bytes):
    return ForkingPickler.loads(data)


def cuda_ipc_serialize(value) -> bytes:
    monkey_patch_torch_reductions()
    buf = io.BytesIO()
    ForkingPickler(buf).dump(value)
    buf.seek(0)
    output = buf.read()
    return output


def cuda_ipc_deserialize(data: bytes):
    monkey_patch_torch_reductions()
    return ForkingPickler.loads(data)


@torch.no_grad()
def group_tensors_by_shape_and_dtype(
    tensors: List[torch.Tensor], max_tensor_size: int = 5 * 1024 * 1024 * 1024, **kwargs
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Group tensors by shape and dtype, ensuring each group's total size is less than max_tensor_size.
    If one tensor is too large, it will be put into a separate group.

    Args:
        tensors: List of torch.Tensor to group
        max_tensor_size: Maximum size in bytes for each tensor group (default: 5GB)

    Returns:
        Tuple containing:
        - List of tensor groups, where each group is a list of tensors with same shape and dtype
        - List of metadata dictionaries for reconstructing original tensors

    Raises:
        ValueError: If any tensor is not suitable for IPC serialization
    """

    if not tensors:
        return [], []

    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    logger.info(
        f"Start to group tensors, total size: {total_size}, num tensors: {len(tensors)}"
    )
    # 1. Group tensors by shape and dtype
    tensor_groups: Dict[
        Tuple[torch.Size, torch.dtype], List[Tuple[int, torch.Tensor]]
    ] = {}
    for i, tensor in enumerate(tensors):
        key = (tensor.shape, tensor.dtype)
        if key not in tensor_groups:
            tensor_groups[key] = []
        tensor_groups[key].append((i, tensor))

    # 2. Create groups and metadata
    final_tensor_groups = []
    metadata = []

    for _, group_tensors in tensor_groups.items():
        # Sort by original index to maintain order
        group_tensors.sort(key=lambda x: x[0])
        # Split into multiple groups - try to group tensors efficiently
        current_group = []
        current_group_indices = []
        current_group_size = 0

        for orig_idx, tensor in group_tensors:
            tensor_size = tensor.element_size() * tensor.numel()
            current_group.append(tensor)
            current_group_indices.append(orig_idx)
            current_group_size += tensor_size
            # Check if this tensor can fit in current group
            if current_group_size > max_tensor_size:
                # Finalize current group and start new one
                # Use clone() to ensure a copy so caller can safely release original tensors
                concatenated = torch.cat(current_group, dim=0).clone()
                final_tensor_groups.append(concatenated)
                # Record metadata for tensors in this group
                offset_elements = 0
                for i, group_tensor in enumerate(current_group):
                    group_tensor_elements = group_tensor.numel()
                    metadata.append(
                        {
                            "original_index": current_group_indices[i],
                            "shape": group_tensor.shape,
                            "dtype": group_tensor.dtype,
                            "group_index": len(final_tensor_groups) - 1,
                            "offset": offset_elements,
                            "size": group_tensor_elements,
                        }
                    )
                    offset_elements += group_tensor_elements
                # Start new group
                current_group = []
                current_group_indices = []
                current_group_size = 0

        # Finalize any remaining group
        if current_group:
            # Use clone() to ensure a copy so caller can safely release original tensors
            concatenated = torch.cat(current_group, dim=0).clone()
            final_tensor_groups.append(concatenated)
            # Record metadata for tensors in this group
            offset_elements = 0
            for i, group_tensor in enumerate(current_group):
                group_tensor_elements = group_tensor.numel()
                metadata.append(
                    {
                        "original_index": current_group_indices[i],
                        "shape": group_tensor.shape,
                        "dtype": group_tensor.dtype,
                        "group_index": len(final_tensor_groups) - 1,
                        "offset": offset_elements,
                        "size": group_tensor_elements,
                    }
                )
                offset_elements += group_tensor_elements
    logger.info(
        f"Grouped tensors, num groups: {len(final_tensor_groups)}, num tensors: {len(metadata)}"
    )
    return final_tensor_groups, metadata


@torch.no_grad()
def reconstruct_tensors_from_groups(
    tensor_groups: List[torch.Tensor], metadata: List[Dict]
) -> List[torch.Tensor]:
    """
    Reconstruct original tensors from grouped tensors and metadata.

    Args:
        tensor_groups: List of tensor groups, where each group is a list of tensors
        metadata: List of metadata dictionaries for reconstructing original tensors

    Returns:
        List[torch.Tensor]: List of reconstructed tensors in their original order
    """

    if not metadata:
        return []

    # Find the maximum original index to determine result size
    max_index = max(item["original_index"] for item in metadata)
    result_tensors = [None] * (max_index + 1)

    # Group metadata by group_index for efficient processing
    group_metadata = {}
    for item in metadata:
        group_idx = item["group_index"]
        if group_idx not in group_metadata:
            group_metadata[group_idx] = []
        group_metadata[group_idx].append(item)

    # Process each tensor group
    for group_idx, group_tensor in enumerate(tensor_groups):
        if (
            group_idx not in group_metadata
            or group_tensor is None
            or group_tensor.numel() == 0
        ):
            raise ValueError(f"Group {group_idx} not found in metadata")
        # Process each tensor in this group
        for item in group_metadata[group_idx]:
            orig_idx = item["original_index"]
            shape = item["shape"]
            offset = item["offset"]
            size = item["size"]
            # For complete tensors, extract based on offset and size
            start_element = offset
            end_element = start_element + size
            tensor_data = group_tensor.view(-1)[start_element:end_element].view(shape)
            # Complete tensor
            result_tensors[orig_idx] = tensor_data
    return result_tensors


def release_tensors(tensors: Union[Iterable[torch.Tensor], torch.Tensor]):
    if not isinstance(tensors, Iterable):
        tensors = [tensors]
    for tensor in tensors:
        tensor.untyped_storage().resize_(0)


def check_and_log_nan_values(tensor, tensor_name, stage_info="", max_indices=20):
    """
    Check for NaN values in a tensor and log detailed information including indices.

    Args:
        tensor: The tensor to check for NaN values
        tensor_name: Name of the tensor for logging
        stage_info: Additional stage information for logging context
        max_indices: Maximum number of NaN indices to log (default: 20)

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(
            f"Parameter {tensor_name} contains NaN values{stage_info}! Shape: {tensor.shape}"
        )
        logger.warning(f"NaN count in {tensor_name}: {nan_count}")

        # Find and print indices of NaN values
        nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=True)
        logger.warning(f"NaN indices in {tensor_name}{stage_info}: {nan_indices}")

        # Print first max_indices NaN indices
        if nan_count <= max_indices:
            logger.warning(
                f"All NaN values in {tensor_name}{stage_info}: {nan_indices}"
            )
        else:
            logger.warning(
                f"First {max_indices} NaN indices in {tensor_name}{stage_info}: {tuple(idx[:max_indices] for idx in nan_indices)}"
            )

        return True
    return False


def compare_and_log_tensor_differences(
    tensor1,
    tensor2,
    tensor_name,
    atol=1e-08,
    rtol=1e-05,
    max_differences=20,
    exact_match=False,
):
    """
    Compare two tensors and log detailed information about inconsistent elements.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        tensor_name: Name of the tensor for logging
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        max_differences: Maximum number of differences to log (default: 20)

    Returns:
        bool: True if tensors are consistent, False otherwise
    """
    # Check if shapes match
    if tensor1.shape != tensor2.shape:
        logger.error(
            f"Shape mismatch for {tensor_name}: {tensor1.shape} vs {tensor2.shape}"
        )
        return False

    if tensor1.dtype != tensor2.dtype:
        logger.error(
            f"Tensor {tensor_name} has different dtypes: {tensor1.dtype} vs {tensor2.dtype}"
        )
        return False

    # Check if tensors are close using torch.allclose
    if exact_match:
        if torch.equal(tensor1, tensor2):
            return True
    else:
        if torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            return True
    logger.error(
        f"Tensors are not close for {tensor_name}, get {tensor1.shape} \n{tensor1} expect {tensor2.shape} \n{tensor2}"
    )

    # Find elements that are not close
    close_mask = torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    inconsistent_mask = ~close_mask

    if inconsistent_mask.any():
        inconsistent_count = inconsistent_mask.sum().item()
        logger.error(
            f"Parameter {tensor_name} has {inconsistent_count} inconsistent elements"
        )

        # Find indices of inconsistent elements
        inconsistent_indices = torch.nonzero(inconsistent_mask, as_tuple=True)

        # Calculate absolute and relative differences
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (
            torch.abs(tensor2) + 1e-8
        )  # Add small epsilon to avoid division by zero

        # Get the actual values at inconsistent positions
        tensor1_values = tensor1[inconsistent_indices]
        tensor2_values = tensor2[inconsistent_indices]
        abs_diff_values = abs_diff[inconsistent_indices]
        rel_diff_values = rel_diff[inconsistent_indices]

        # Log summary statistics
        max_abs_diff = abs_diff_values.max().item()
        max_rel_diff = rel_diff_values.max().item()
        mean_abs_diff = abs_diff_values.mean().item()
        mean_rel_diff = rel_diff_values.mean().item()

        logger.error(
            f"Max absolute difference: {max_abs_diff:.6f}, Max relative difference: {max_rel_diff:.6f}"
        )
        logger.error(
            f"Mean absolute difference: {mean_abs_diff:.6f}, Mean relative difference: {mean_rel_diff:.6f}"
        )

        # Log detailed information for first max_differences elements
        num_to_log = min(int(inconsistent_count), max_differences)

        for i in range(num_to_log):
            idx = tuple(idx[i] for idx in inconsistent_indices)
            val1 = tensor1_values[i].item()
            val2 = tensor2_values[i].item()
            abs_diff_val = abs_diff_values[i].item()
            rel_diff_val = rel_diff_values[i].item()

            logger.error(
                f"  Index {idx}: {val1:.6f} vs {val2:.6f} (abs_diff: {abs_diff_val:.6f}, rel_diff: {rel_diff_val:.6f})"
            )

        if inconsistent_count > max_differences:
            logger.error(
                f"  ... and {inconsistent_count - max_differences} more differences"
            )

        return False

    return True
