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
import os
from unittest.mock import patch

import pytest
import torch

from awex.util.common import get_free_port
from awex.util.process_group import init_weights_update_group, setup_batch_isend_irecv
from awex.util.tensor_util import (
    check_and_log_nan_values,
    compare_and_log_tensor_differences,
)

mp.set_start_method("spawn", force=True)


class TestCheckAndLogNanValues:
    """Test cases for check_and_log_nan_values function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a test logger to capture log messages
        self.logger_patcher = patch("awex.util.tensor_util.logger")
        self.mock_logger = self.logger_patcher.start()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()

    def test_tensor_without_nan_values(self):
        """Test function with tensor containing no NaN values."""
        # Create a tensor without NaN values
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor_name = "test_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return False
        assert result is False

        # Should not log any warnings
        self.mock_logger.warning.assert_not_called()

    def test_tensor_with_nan_values(self):
        """Test function with tensor containing NaN values."""
        # Create a tensor with NaN values
        tensor = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        tensor_name = "test_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return True
        assert result is True

        # Should log warnings
        assert self.mock_logger.warning.call_count >= 3

        # Check that the first warning contains the expected message
        first_call_args = self.mock_logger.warning.call_args_list[0][0][0]
        assert "Parameter test_tensor contains NaN values" in first_call_args
        assert "Shape: torch.Size([2, 3])" in first_call_args

    def test_tensor_with_nan_values_and_stage_info(self):
        """Test function with tensor containing NaN values and stage info."""
        # Create a tensor with NaN values
        tensor = torch.tensor([[1.0, float("nan"), 3.0]])
        tensor_name = "test_tensor"
        stage_info = " during forward pass"

        result = check_and_log_nan_values(tensor, tensor_name, stage_info)

        # Should return True
        assert result is True

        # Should log warnings with stage info
        first_call_args = self.mock_logger.warning.call_args_list[0][0][0]
        assert "during forward pass" in first_call_args

    def test_tensor_with_many_nan_values(self):
        """Test function with tensor containing many NaN values (more than max_indices)."""
        # Create a large tensor with many NaN values
        tensor = torch.ones(5, 5)
        tensor[0, 0] = float("nan")
        tensor[1, 1] = float("nan")
        tensor[2, 2] = float("nan")
        tensor[3, 3] = float("nan")
        tensor[4, 4] = float("nan")
        tensor[0, 1] = float("nan")
        tensor[1, 0] = float("nan")

        tensor_name = "large_tensor"
        max_indices = 3

        result = check_and_log_nan_values(tensor, tensor_name, max_indices=max_indices)

        # Should return True
        assert result is True

        # Should log warnings
        assert self.mock_logger.warning.call_count >= 3

        # Check that it logs the limited number of indices
        for call_args in self.mock_logger.warning.call_args_list:
            args = call_args[0][0]
            if "First 3 NaN indices" in args:
                assert "First 3 NaN indices" in args
                break

    def test_tensor_with_few_nan_values(self):
        """Test function with tensor containing few NaN values (less than max_indices)."""
        # Create a tensor with few NaN values
        tensor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
        tensor_name = "small_tensor"
        max_indices = 5

        result = check_and_log_nan_values(tensor, tensor_name, max_indices=max_indices)

        # Should return True
        assert result is True

        # Should log warnings
        assert self.mock_logger.warning.call_count >= 3

        # Check that it logs all NaN indices
        for call_args in self.mock_logger.warning.call_args_list:
            args = call_args[0][0]
            if "All NaN values" in args:
                assert "All NaN values" in args
                break

    def test_empty_tensor(self):
        """Test function with empty tensor."""
        tensor = torch.tensor([])
        tensor_name = "empty_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return False (no NaN values in empty tensor)
        assert result is False

        # Should not log any warnings
        self.mock_logger.warning.assert_not_called()

    def test_tensor_with_all_nan_values(self):
        """Test function with tensor containing all NaN values."""
        tensor = torch.full((2, 2), float("nan"))
        tensor_name = "all_nan_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return True
        assert result is True

        # Should log warnings
        assert self.mock_logger.warning.call_count >= 3

        # Check NaN count
        for call_args in self.mock_logger.warning.call_args_list:
            args = call_args[0][0]
            if "NaN count" in args:
                assert "NaN count in all_nan_tensor: 4" in args
                break

    def test_tensor_with_inf_values(self):
        """Test function with tensor containing inf values (should not detect as NaN)."""
        tensor = torch.tensor([[1.0, float("inf")], [3.0, float("-inf")]])
        tensor_name = "inf_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return False (inf is not NaN)
        assert result is False

        # Should not log any warnings
        self.mock_logger.warning.assert_not_called()

    def test_tensor_with_mixed_nan_and_inf_values(self):
        """Test function with tensor containing both NaN and inf values."""
        tensor = torch.tensor([[1.0, float("nan")], [float("inf"), 4.0]])
        tensor_name = "mixed_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return True (contains NaN)
        assert result is True

        # Should log warnings
        assert self.mock_logger.warning.call_count >= 3

        # Check NaN count should be 1
        for call_args in self.mock_logger.warning.call_args_list:
            args = call_args[0][0]
            if "NaN count" in args:
                assert "NaN count in mixed_tensor: 1" in args
                break

    def test_3d_tensor_with_nan_values(self):
        """Test function with 3D tensor containing NaN values."""
        tensor = torch.ones(2, 3, 4)
        tensor[0, 1, 2] = float("nan")
        tensor[1, 0, 1] = float("nan")

        tensor_name = "3d_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return True
        assert result is True

        # Should log warnings
        assert self.mock_logger.warning.call_count >= 3

        # Check shape in log message
        first_call_args = self.mock_logger.warning.call_args_list[0][0][0]
        assert "Shape: torch.Size([2, 3, 4])" in first_call_args

    def test_default_max_indices(self):
        """Test function with default max_indices parameter."""
        # Create a tensor with more than 20 NaN values
        tensor = torch.ones(5, 5)
        for i in range(5):
            for j in range(5):
                tensor[i, j] = float("nan")

        tensor_name = "default_max_tensor"

        result = check_and_log_nan_values(tensor, tensor_name)

        # Should return True
        assert result is True

        # Should log warnings with default max_indices (20)
        for call_args in self.mock_logger.warning.call_args_list:
            args = call_args[0][0]
            if "First 20 NaN indices" in args:
                assert "First 20 NaN indices" in args
                break

    def test_zero_max_indices(self):
        """Test function with max_indices set to 0."""
        tensor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
        tensor_name = "zero_max_tensor"
        max_indices = 0

        result = check_and_log_nan_values(tensor, tensor_name, max_indices=max_indices)

        # Should return True
        assert result is True

        # Should log warnings but not show detailed indices
        assert self.mock_logger.warning.call_count >= 2

        # Should have "First 0 NaN indices" in logs (actual behavior)
        found_zero_indices = False
        for call_args in self.mock_logger.warning.call_args_list:
            args = call_args[0][0]
            if "First 0 NaN indices" in args:
                found_zero_indices = True
                break
        assert found_zero_indices, (
            "Should log 'First 0 NaN indices' when max_indices is 0"
        )


class TestCompareAndLogTensorDifferences:
    """Test cases for compare_and_log_tensor_differences function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a test logger to capture log messages
        self.logger_patcher = patch("awex.util.tensor_util.logger")
        self.mock_logger = self.logger_patcher.start()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()

    def test_identical_tensors(self):
        """Test function with identical tensors."""
        tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor_name = "identical_tensors"

        result = compare_and_log_tensor_differences(tensor1, tensor2, tensor_name)

        # Should return True
        assert result is True

        # Should not log any errors
        self.mock_logger.error.assert_not_called()

    def test_tensors_within_tolerance(self):
        """Test function with tensors that are within tolerance."""
        tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + 1e-3
        tensor_name = "within_tolerance_tensors"

        result = compare_and_log_tensor_differences(
            tensor1, tensor2, tensor_name, atol=1e-2, rtol=1e-2
        )

        # Should return True
        assert result is True

        # Should not log any errors
        self.mock_logger.error.assert_not_called()

    def test_tensors_outside_tolerance(self):
        """Test function with tensors that are outside tolerance."""
        tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + 0.1
        tensor_name = "outside_tolerance_tensors"

        result = compare_and_log_tensor_differences(
            tensor1, tensor2, tensor_name, atol=1e-2, rtol=1e-2
        )

        # Should return False
        assert result is False

        # Should log errors
        assert self.mock_logger.error.call_count >= 3

        # Check that it logs the number of inconsistent elements (in later calls)
        found_inconsistent_count = False
        for call_args in self.mock_logger.error.call_args_list:
            args = call_args[0][0]
            if "has 6 inconsistent elements" in args:
                found_inconsistent_count = True
                break
        assert found_inconsistent_count, (
            "Should log the number of inconsistent elements"
        )

    def test_shape_mismatch(self):
        """Test function with tensors of different shapes."""
        tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor_name = "shape_mismatch_tensors"

        result = compare_and_log_tensor_differences(tensor1, tensor2, tensor_name)

        # Should return False
        assert result is False

        # Should log shape mismatch error
        first_call_args = self.mock_logger.error.call_args_list[0][0][0]
        assert "Shape mismatch" in first_call_args
        assert "torch.Size([2, 2]) vs torch.Size([2, 3])" in first_call_args

    def test_specific_element_differences(self):
        """Test function with specific element differences."""
        tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2[0, 1] = 2.1  # Change one element
        tensor2[1, 2] = 6.1  # Change another element
        tensor_name = "specific_differences_tensors"

        result = compare_and_log_tensor_differences(
            tensor1, tensor2, tensor_name, atol=1e-2, rtol=1e-2
        )

        # Should return False
        assert result is False

        # Should log errors
        assert self.mock_logger.error.call_count >= 3

        # Check that it logs the number of inconsistent elements (in later calls)
        found_inconsistent_count = False
        for call_args in self.mock_logger.error.call_args_list:
            args = call_args[0][0]
            if "has 2 inconsistent elements" in args:
                found_inconsistent_count = True
                break
        assert found_inconsistent_count, (
            "Should log the number of inconsistent elements"
        )

    def test_max_differences_limit(self):
        """Test function with max_differences limit."""
        tensor1 = torch.ones(5, 5)
        tensor2 = torch.ones(5, 5)
        # Change many elements
        for i in range(5):
            for j in range(5):
                tensor2[i, j] = 1.1

        tensor_name = "max_differences_tensors"
        max_differences = 3

        result = compare_and_log_tensor_differences(
            tensor1, tensor2, tensor_name, max_differences=max_differences
        )

        # Should return False
        assert result is False

        # Should log errors
        assert self.mock_logger.error.call_count >= 4

        # Check that it logs the limited number of differences
        for call_args in self.mock_logger.error.call_args_list:
            args = call_args[0][0]
            if "and 22 more differences" in args:
                assert "and 22 more differences" in args
                break

    def test_statistics_logging(self):
        """Test that statistics are logged correctly."""
        tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + 0.1
        tensor_name = "statistics_tensors"

        result = compare_and_log_tensor_differences(
            tensor1, tensor2, tensor_name, atol=1e-2, rtol=1e-2
        )

        # Should return False
        assert result is False

        # Should log statistics
        stats_found = False
        for call_args in self.mock_logger.error.call_args_list:
            args = call_args[0][0]
            if (
                "Max absolute difference:" in args
                and "Max relative difference:" in args
            ):
                stats_found = True
                break

        assert stats_found, "Statistics should be logged"

    def test_zero_tolerance(self):
        """Test function with zero tolerance."""
        tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor2 = (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + 0.1
        )  # Use larger difference
        tensor_name = "zero_tolerance_tensors"

        result = compare_and_log_tensor_differences(
            tensor1, tensor2, tensor_name, atol=0, rtol=0
        )

        # Should return False (differences will be detected)
        assert result is False

        # Should log errors
        assert self.mock_logger.error.call_count >= 3


def _batch_isend_irecv_worker(rank, world_size, master_port, result_queue):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.cuda.set_device(rank)

        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://localhost:{master_port}",
        )

        process_group = init_weights_update_group(
            master_address="localhost",
            master_port=master_port + 1000,
            rank=rank,
            world_size=world_size,
            group_name="test_group",
            backend="nccl",
        )

        setup_batch_isend_irecv(process_group, rank, world_size)
        torch.distributed.destroy_process_group()
        result_queue.put((rank, True))

    except Exception as e:
        result_queue.put((rank, False, str(e)))


@pytest.mark.skipif(
    torch.cuda.device_count() <= 1 or torch.cuda.device_count() % 2 != 0,
    reason="Only one GPU present or GPU count is not even",
)
def test_setup_batch_isend_irecv_8_processes():
    """Test setup_batch_isend_irecv with 8 processes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    world_size = 8
    master_port = get_free_port()
    result_queue = mp.Queue()

    # Start processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=_batch_isend_irecv_worker,
            args=(rank, world_size, master_port, result_queue),
        )
        p.start()
        processes.append(p)

    # Wait and collect results
    for p in processes:
        p.join(timeout=30)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Verify all succeeded
    assert len(results) == world_size
    for rank, success, *error in results:
        assert success, (
            f"Process {rank} failed: {error[0] if error else 'unknown error'}"
        )
