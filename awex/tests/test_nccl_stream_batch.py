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

import torch

from awex.transfer.nccl_stream_batch import (
    _clone_p2p_send_tensor,
    _prepare_p2p_recv_tensor,
    _sync_p2p_recv_tensor_pairs,
)


def test_prepare_p2p_recv_tensor_uses_dense_buffer_for_noncontiguous_view():
    base = torch.zeros(4, 4)
    view = base[:, 1]
    assert not view.is_contiguous()

    recv_tensor, copyback_pair = _prepare_p2p_recv_tensor(view)

    assert recv_tensor.is_contiguous()
    assert copyback_pair is not None
    recv_tensor.copy_(torch.arange(4, dtype=base.dtype))

    _sync_p2p_recv_tensor_pairs([copyback_pair])

    torch.testing.assert_close(base[:, 1], torch.arange(4, dtype=base.dtype))
    torch.testing.assert_close(base[:, 0], torch.zeros(4))


def test_prepare_p2p_recv_tensor_reuses_contiguous_tensor():
    tensor = torch.zeros(4)

    recv_tensor, copyback_pair = _prepare_p2p_recv_tensor(tensor)

    assert recv_tensor is tensor
    assert copyback_pair is None


def test_clone_p2p_send_tensor_returns_contiguous_clone():
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    view = base[:, 1]
    assert not view.is_contiguous()

    cloned = _clone_p2p_send_tensor(view)

    assert cloned.is_contiguous()
    assert cloned.data_ptr() != view.data_ptr()
    torch.testing.assert_close(cloned, view)
