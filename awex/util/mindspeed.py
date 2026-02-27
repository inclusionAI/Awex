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

from awex import logging
from awex.util import device as device_util

logger = logging.getLogger(__name__)
_PATCHED = False


def ensure_mindspeed_patched(reason: str | None = None) -> bool:
    """Ensure MindSpeed patches are applied before importing Megatron.

    Returns True if MindSpeed is available and patching is applied, False otherwise.
    """
    global _PATCHED
    if _PATCHED:
        return True

    use_mindspeed = os.environ.get("AWEX_USE_MINDSPEED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_mindspeed and device_util.get_device_type() != "npu":
        return False

    try:
        import mindspeed.megatron_adaptor  # noqa: F401

        _PATCHED = True
        logger.info(
            "MindSpeed patches applied%s.",
            f" ({reason})" if reason else "",
        )
        return True
    except Exception as exc:
        logger.warning(
            "MindSpeed not available or failed to patch%s: %s",
            f" ({reason})" if reason else "",
            exc,
        )
        return False
