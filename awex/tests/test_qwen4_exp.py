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

import pytest
import torch

from awex.models.qwen4_exp_layout import Qwen4ExpGDNLayout


def _labels(heads, widths, tail):
    # Encode each semantic coordinate independently of the implementation's
    # reshape/split operations. MCore concatenates whole head groups.
    rows = []
    lookup = {}
    for head in range(heads):
        for category, width in enumerate(widths):
            for channel in range(width):
                value = category * 100000 + head * 1000 + channel
                rows.append(value)
                lookup[category, head, channel] = value
    tensor = torch.tensor(rows, dtype=torch.int64)
    return tensor.reshape(-1, *([1] * len(tail))).expand(-1, *tail).clone(), lookup


def _expected(lookup, heads, widths, categories, infer_tp, tail):
    rows = []
    for rank in range(infer_tp):
        for category in categories:
            for head in range(rank * heads // infer_tp, (rank + 1) * heads // infer_tp):
                for channel in range(widths[category]):
                    rows.append(lookup[category, head, channel])
    return (
        torch.tensor(rows, dtype=torch.int64)
        .reshape(-1, *([1] * len(tail)))
        .expand(-1, *tail)
    )


@pytest.mark.parametrize("train_tp", [1, 2, 4, 8])
@pytest.mark.parametrize("infer_tp", [1, 2, 4, 8])
def test_gdn_packing_multiple_heads_preserves_semantic_coordinates(train_tp, infer_tp):
    """Actual model head geometry; narrow hidden width keeps this CPU test small."""
    layout = Qwen4ExpGDNLayout(16, 48, 128, 128)
    widths = (128, 128, 384, 384, 3, 3)
    source, lookup = _labels(16, widths, (3,))
    original = source.clone()
    qkvz, ba = layout.pack_input(source, train_tp, infer_tp)
    for actual, categories in ((qkvz, range(4)), (ba, range(4, 6))):
        expected = _expected(lookup, 16, widths, categories, infer_tp, (3,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)

    conv, conv_lookup = _labels(16, widths[:3], (1, 4))
    actual_conv = layout.pack_conv(conv, train_tp, infer_tp)
    expected_conv = _expected(conv_lookup, 16, widths[:3], range(3), infer_tp, (1, 4))
    torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
    for component, sizes in (("qkvz", widths[:4]), ("ba", widths[4:])):
        decoupled, labels = _labels(16, sizes, (3,))
        actual = layout.pack_decoupled(decoupled, train_tp, infer_tp, component)
        expected = _expected(labels, 16, sizes, range(len(sizes)), infer_tp, (3,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("infer_tp", [1, 2, 4, 8])
@pytest.mark.parametrize("tail", [(), (3,)])
def test_gated_qkv_preserves_head_gate_pairs_and_replicates_kv(infer_tp, tail):
    from awex.models.qwen4_exp_layout import pack_qwen4_exp_gated_qkv

    heads, kv_heads, dim = 24, 2, 4
    queries = [
        [10000 + h * 100 + g * 10 + c for g in range(2) for c in range(dim)]
        for h in range(heads)
    ]
    keys = [[20000 + h * 100 + c for c in range(dim)] for h in range(kv_heads)]
    values = [[30000 + h * 100 + c for c in range(dim)] for h in range(kv_heads)]
    source = []
    for kv in range(kv_heads):
        for head in range(kv * 12, (kv + 1) * 12):
            source.extend(queries[head])
        source.extend(keys[kv])
        source.extend(values[kv])
    expected = []
    for rank in range(infer_tp):
        for head in range(rank * heads // infer_tp, (rank + 1) * heads // infer_tp):
            expected.extend(queries[head])
        owners = range(kv_heads) if infer_tp == 1 else [rank // (infer_tp // kv_heads)]
        for category in (keys, values):
            for owner in owners:
                expected.extend(category[owner])

    def tensor(rows):
        return torch.tensor(rows).reshape(-1, *([1] * len(tail))).expand(-1, *tail)

    actual = pack_qwen4_exp_gated_qkv(tensor(source), heads, kv_heads, dim, infer_tp)
    torch.testing.assert_close(actual, tensor(expected), rtol=0, atol=0)
