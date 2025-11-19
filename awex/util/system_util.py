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

import glob
import os
import subprocess
from typing import Dict

import psutil


def get_rlimit_nofile() -> tuple[int, int]:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft, hard


def count_open_fds() -> int:
    return len(glob.glob(f"/proc/{os.getpid()}/fd/*"))


def count_sysv_ipc() -> Dict[str, int]:
    pid = str(os.getpid())
    counts = {"msg": 0, "sem": 0, "shm": 0}

    for ipc_type in counts.keys():
        try:
            out = subprocess.check_output(
                ["ipcs", "-p"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                cols = line.split()
                if len(cols) >= 6 and pid in cols[-3:]:
                    counts[ipc_type] += 1
        except Exception:
            counts[ipc_type] = -1
    return counts


def count_posix_ipc() -> Dict[str, int]:
    pid = os.getpid()
    proc = psutil.Process(pid)
    counts = {"shm": 0, "sem": 0, "mq": 0}

    for fd in proc.open_files() + proc.memory_maps():
        path = fd.path if hasattr(fd, "path") else fd
        if "/dev/shm/" in path:
            counts["shm"] += 1
        elif path.startswith("/dev/mqueue/"):
            counts["mq"] += 1
        elif "anon_inode:[eventfd]" in str(fd):
            counts["sem"] += 1  # 近似

    return counts


def get_handle_counts():
    soft, hard = get_rlimit_nofile()
    open_fds = count_open_fds()
    sysv = count_sysv_ipc()
    posix = count_posix_ipc()

    return {
        "soft_limit": soft,
        "hard_limit": hard,
        "open_fds": open_fds,
        "sysv_ipc": sum(sysv.values()),
        "posix_ipc": sum(posix.values()),
        "total_ipc": sum(sysv.values()) + sum(posix.values()),
    }
