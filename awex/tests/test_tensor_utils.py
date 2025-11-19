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

import multiprocessing as mp

import pytest
import torch

from awex.util.tensor_util import (
    group_tensors_by_shape_and_dtype,
    reconstruct_tensors_from_groups,
)

mp.set_start_method("spawn", force=True)


class TestGroupTensorsByShapeAndDtype:
    """Test cases for group_tensors_by_shape_and_dtype function."""

    def test_empty_tensors_list(self):
        """Test function with empty tensors list."""
        tensors = []
        result_groups, result_metadata = group_tensors_by_shape_and_dtype(tensors)

        assert result_groups == []
        assert result_metadata == []

    def test_single_tensor(self):
        """Test function with a single tensor."""
        tensor = torch.randn(3, 4, dtype=torch.float32)
        tensors = [tensor]

        result_groups, result_metadata = group_tensors_by_shape_and_dtype(tensors)

        assert len(result_groups) == 1
        assert len(result_metadata) == 1

        # Check that the group contains the original tensor
        assert torch.equal(result_groups[0], tensor)

        # Check metadata
        metadata = result_metadata[0]
        assert metadata["original_index"] == 0
        assert metadata["shape"] == tensor.shape
        assert metadata["dtype"] == tensor.dtype
        assert metadata["group_index"] == 0
        assert metadata["offset"] == 0
        assert metadata["size"] == tensor.numel()

    def test_tensors_same_shape_and_dtype(self):
        """Test function with multiple tensors of same shape and dtype."""
        tensor1 = torch.randn(2, 3, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, dtype=torch.float32)
        tensor3 = torch.randn(2, 3, dtype=torch.float32)
        tensors = [tensor1, tensor2, tensor3]

        result_groups, result_metadata = group_tensors_by_shape_and_dtype(tensors)

        assert len(result_groups) == 1
        assert len(result_metadata) == 3

        # Check that all tensors are concatenated in one group
        expected_concatenated = torch.cat([tensor1, tensor2, tensor3], dim=0)
        assert torch.equal(result_groups[0], expected_concatenated)

        # Check metadata for each tensor
        for i, metadata in enumerate(result_metadata):
            assert metadata["original_index"] == i
            assert metadata["shape"] == tensor1.shape
            assert metadata["dtype"] == tensor1.dtype
            assert metadata["group_index"] == 0
            assert metadata["size"] == tensor1.numel()

    def test_tensors_different_shapes(self):
        """Test function with tensors of different shapes."""
        tensor1 = torch.randn(2, 3, dtype=torch.float32)
        tensor2 = torch.randn(4, 5, dtype=torch.float32)
        tensor3 = torch.randn(2, 3, dtype=torch.float32)
        tensors = [tensor1, tensor2, tensor3]

        result_groups, result_metadata = group_tensors_by_shape_and_dtype(tensors)

        assert len(result_groups) == 2  # Two different shapes
        assert len(result_metadata) == 3

        # Check that tensors with same shape are grouped together
        # tensor1 and tensor3 should be in one group
        # tensor2 should be in another group
        shape_groups = {}
        for metadata in result_metadata:
            shape = metadata["shape"]
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append(metadata["original_index"])

        assert set(shape_groups[torch.Size([2, 3])]) == {0, 2}
        assert set(shape_groups[torch.Size([4, 5])]) == {1}

    def test_tensors_different_dtypes(self):
        """Test function with tensors of different dtypes."""
        tensor1 = torch.randn(2, 3, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, dtype=torch.float64)
        tensor3 = torch.randn(2, 3, dtype=torch.float32)
        tensors = [tensor1, tensor2, tensor3]

        result_groups, result_metadata = group_tensors_by_shape_and_dtype(tensors)

        assert len(result_groups) == 2  # Two different dtypes
        assert len(result_metadata) == 3

        # Check that tensors with same dtype are grouped together
        dtype_groups = {}
        for metadata in result_metadata:
            dtype = metadata["dtype"]
            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append(metadata["original_index"])

        assert set(dtype_groups[torch.float32]) == {0, 2}
        assert set(dtype_groups[torch.float64]) == {1}

    def test_max_tensor_size_limit(self):
        """Test function with max_tensor_size limit."""
        # Create tensors that would exceed the default 5GB limit when grouped
        large_tensor = torch.randn(1000, 1000, dtype=torch.float32)  # ~4MB each
        tensors = [large_tensor] * 6  # ~24MB total

        # Set a smaller max_tensor_size to force splitting
        max_size = 10 * 1024 * 1024  # 10MB
        result_groups, result_metadata = group_tensors_by_shape_and_dtype(
            tensors, max_size
        )

        # Should create multiple groups
        assert len(result_groups) > 1
        assert len(result_metadata) == len(tensors)

        # Check that each group size is within limit
        for group in result_groups:
            group_size = group.element_size() * group.numel()
            assert group_size <= max_size * 2

    def test_complex_tensor_shapes(self):
        """Test function with complex tensor shapes."""
        tensor1 = torch.randn(2, 3, 4, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, 4, dtype=torch.float32)
        tensor3 = torch.randn(1, 5, dtype=torch.float32)
        tensors = [tensor1, tensor2, tensor3]

        result_groups, result_metadata = group_tensors_by_shape_and_dtype(tensors)

        assert len(result_groups) == 2  # Two different shapes
        assert len(result_metadata) == 3

        # Verify reconstruction works correctly
        reconstructed = reconstruct_tensors_from_groups(result_groups, result_metadata)
        assert len(reconstructed) == 3
        assert torch.equal(reconstructed[0], tensor1)
        assert torch.equal(reconstructed[1], tensor2)
        assert torch.equal(reconstructed[2], tensor3)


class TestReconstructTensorsFromGroups:
    """Test cases for reconstruct_tensors_from_groups function."""

    def test_empty_metadata(self):
        """Test function with empty metadata."""
        result = reconstruct_tensors_from_groups([], [])
        assert result == []

    def test_single_tensor_reconstruction(self):
        """Test reconstruction of a single tensor."""
        original_tensor = torch.randn(3, 4, dtype=torch.float32)
        tensor_groups = [original_tensor]
        metadata = [
            {
                "original_index": 0,
                "shape": original_tensor.shape,
                "dtype": original_tensor.dtype,
                "group_index": 0,
                "offset": 0,
                "size": original_tensor.numel(),
            }
        ]

        result = reconstruct_tensors_from_groups(tensor_groups, metadata)

        assert len(result) == 1
        assert torch.equal(result[0], original_tensor)

    def test_multiple_tensors_same_group(self):
        """Test reconstruction of multiple tensors from same group."""
        tensor1 = torch.randn(2, 3, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, dtype=torch.float32)
        concatenated = torch.cat([tensor1, tensor2], dim=0)

        tensor_groups = [concatenated]
        metadata = [
            {
                "original_index": 0,
                "shape": tensor1.shape,
                "dtype": tensor1.dtype,
                "group_index": 0,
                "offset": 0,
                "size": tensor1.numel(),
            },
            {
                "original_index": 1,
                "shape": tensor2.shape,
                "dtype": tensor2.dtype,
                "group_index": 0,
                "offset": tensor1.numel(),
                "size": tensor2.numel(),
            },
        ]

        result = reconstruct_tensors_from_groups(tensor_groups, metadata)

        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)

    def test_multiple_groups(self):
        """Test reconstruction from multiple groups."""
        # Group 1: tensors with shape (2, 3)
        tensor1 = torch.randn(2, 3, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, dtype=torch.float32)
        group1 = torch.cat([tensor1, tensor2], dim=0)

        # Group 2: tensor with shape (4, 5)
        tensor3 = torch.randn(4, 5, dtype=torch.float32)
        group2 = tensor3

        tensor_groups = [group1, group2]
        metadata = [
            {
                "original_index": 0,
                "shape": tensor1.shape,
                "dtype": tensor1.dtype,
                "group_index": 0,
                "offset": 0,
                "size": tensor1.numel(),
            },
            {
                "original_index": 1,
                "shape": tensor2.shape,
                "dtype": tensor2.dtype,
                "group_index": 0,
                "offset": tensor1.numel(),
                "size": tensor2.numel(),
            },
            {
                "original_index": 2,
                "shape": tensor3.shape,
                "dtype": tensor3.dtype,
                "group_index": 1,
                "offset": 0,
                "size": tensor3.numel(),
            },
        ]

        result = reconstruct_tensors_from_groups(tensor_groups, metadata)

        assert len(result) == 3
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)
        assert torch.equal(result[2], tensor3)

    def test_non_sequential_indices(self):
        """Test reconstruction with non-sequential original indices."""
        tensor1 = torch.randn(2, 3, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, dtype=torch.float32)
        concatenated = torch.cat([tensor1, tensor2], dim=0)

        tensor_groups = [concatenated]
        metadata = [
            {
                "original_index": 5,  # Non-sequential index
                "shape": tensor1.shape,
                "dtype": tensor1.dtype,
                "group_index": 0,
                "offset": 0,
                "size": tensor1.numel(),
            },
            {
                "original_index": 2,  # Non-sequential index
                "shape": tensor2.shape,
                "dtype": tensor2.dtype,
                "group_index": 0,
                "offset": tensor1.numel(),
                "size": tensor2.numel(),
            },
        ]

        result = reconstruct_tensors_from_groups(tensor_groups, metadata)

        assert len(result) == 6  # Max index + 1
        assert torch.equal(result[5], tensor1)
        assert torch.equal(result[2], tensor2)
        # Other positions should be None
        assert result[0] is None
        assert result[1] is None
        assert result[3] is None
        assert result[4] is None

    def test_missing_group_raises_error(self):
        """Test that missing group raises ValueError."""
        tensor_groups = [torch.randn(2, 3)]
        metadata = [
            {
                "original_index": 0,
                "shape": torch.Size([2, 3]),
                "dtype": torch.float32,
                "group_index": 1,  # Group index that doesn't exist
                "offset": 0,
                "size": 6,
            }
        ]

        with pytest.raises(ValueError, match="Group 0 not found in metadata"):
            reconstruct_tensors_from_groups(tensor_groups, metadata)

    def test_empty_group_raises_error(self):
        """Test that empty group raises ValueError."""
        tensor_groups = [torch.randn(2, 3)]
        metadata = [
            {
                "original_index": 0,
                "shape": torch.Size([2, 3]),
                "dtype": torch.float32,
                "group_index": 0,
                "offset": 0,
                "size": 6,
            }
        ]

        # Create a situation where the group has zero elements
        empty_tensor = torch.tensor([])
        tensor_groups[0] = empty_tensor
        with pytest.raises(ValueError, match="Group 0 not found in metadata"):
            reconstruct_tensors_from_groups(tensor_groups, metadata)

    def test_none_group_raises_error(self):
        """Test that None group raises ValueError."""
        tensor_groups = [None]
        metadata = [
            {
                "original_index": 0,
                "shape": torch.Size([2, 3]),
                "dtype": torch.float32,
                "group_index": 0,
                "offset": 0,
                "size": 6,
            }
        ]

        with pytest.raises(ValueError, match="Group 0 not found in metadata"):
            reconstruct_tensors_from_groups(tensor_groups, metadata)

    def test_complex_tensor_shapes(self):
        """Test reconstruction with complex tensor shapes."""
        tensor1 = torch.randn(2, 3, 4, dtype=torch.float32)
        tensor2 = torch.randn(2, 3, 4, dtype=torch.float32)
        concatenated = torch.cat([tensor1, tensor2], dim=0)

        tensor_groups = [concatenated]
        metadata = [
            {
                "original_index": 0,
                "shape": tensor1.shape,
                "dtype": tensor1.dtype,
                "group_index": 0,
                "offset": 0,
                "size": tensor1.numel(),
            },
            {
                "original_index": 1,
                "shape": tensor2.shape,
                "dtype": tensor2.dtype,
                "group_index": 0,
                "offset": tensor1.numel(),
                "size": tensor2.numel(),
            },
        ]

        result = reconstruct_tensors_from_groups(tensor_groups, metadata)

        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)
        assert result[0].shape == torch.Size([2, 3, 4])
        assert result[1].shape == torch.Size([2, 3, 4])


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_group_and_reconstruct_workflow(self):
        """Test the complete workflow of grouping and reconstructing tensors."""
        # Create test tensors with different shapes and dtypes
        tensors = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, 5, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float64),
            torch.randn(1, 1, dtype=torch.float32),
        ]

        # Group the tensors
        tensor_groups, metadata = group_tensors_by_shape_and_dtype(tensors)

        # Reconstruct the tensors
        reconstructed = reconstruct_tensors_from_groups(tensor_groups, metadata)

        # Verify reconstruction
        assert len(reconstructed) == len(tensors)
        for i, (original, reconstructed_tensor) in enumerate(
            zip(tensors, reconstructed)
        ):
            assert torch.equal(original, reconstructed_tensor), f"Tensor {i} mismatch"

    def test_large_tensor_workflow(self):
        """Test workflow with tensors that exceed size limits."""
        # Create tensors that would exceed default size limit when grouped
        tensors = [
            torch.randn(100, 100, dtype=torch.float32) for _ in range(100)
        ]  # ~4MB total
        # Set a small max_tensor_size to force splitting
        max_size = 100 * 1024  # 100KB
        tensor_groups, metadata = group_tensors_by_shape_and_dtype(tensors, max_size)

        # Should create multiple groups
        assert len(tensor_groups) > 1

        # Reconstruct and verify
        reconstructed = reconstruct_tensors_from_groups(tensor_groups, metadata)
        assert len(reconstructed) == len(tensors)
        for i, (original, reconstructed_tensor) in enumerate(
            zip(tensors, reconstructed)
        ):
            assert torch.equal(original, reconstructed_tensor), f"Tensor {i} mismatch"

    def test_different_dtypes_workflow(self):
        """Test workflow with tensors of different dtypes."""
        tensors = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float16),
        ]

        tensor_groups, metadata = group_tensors_by_shape_and_dtype(tensors)
        reconstructed = reconstruct_tensors_from_groups(tensor_groups, metadata)

        assert len(reconstructed) == len(tensors)
        for i, (original, reconstructed_tensor) in enumerate(
            zip(tensors, reconstructed)
        ):
            assert torch.equal(original, reconstructed_tensor), f"Tensor {i} mismatch"
            assert original.dtype == reconstructed_tensor.dtype, (
                f"Tensor {i} dtype mismatch"
            )
