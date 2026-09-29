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

"""Exercise real client polling/retries with bounded, synthetic network delays."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from awex.meta import meta_server
from awex.util.common import to_binary


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def advance(seconds):
        clock.now += seconds

    clock.advance = advance
    monkeypatch.setattr(meta_server.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(meta_server.time, "sleep", advance)
    monkeypatch.setattr(meta_server.random, "random", lambda: 0)
    return clock


@pytest.fixture
def client(monkeypatch):
    session = Mock()
    monkeypatch.setattr(meta_server.requests, "Session", lambda: session)
    return meta_server.MetaServerClient("127.0.0.1", 12345)


def response(*, exists=True, value=None):
    return SimpleNamespace(
        status_code=200,
        content=to_binary(value),
        json=lambda: {"has_key": exists},
        raise_for_status=lambda: None,
        close=Mock(),
    )


def test_slow_key_queries_consume_deadline(client, clock):
    budgets = []

    def get(url, timeout):
        budgets.append(timeout.total)
        clock.advance(min(0.4, timeout.total))
        return response(exists=False)

    client._session.get.side_effect = get
    with pytest.raises(TimeoutError, match="Timeout waiting for key"):
        client.wait_key("missing", timeout=1)
    assert clock.now == pytest.approx(1)
    assert budgets == pytest.approx([1, 0.4])


@pytest.mark.parametrize("method", ["get_binary", "get_object"])
def test_wait_and_fetch_share_deadline(client, clock, method):
    budgets = []

    def get(url, timeout):
        budgets.append(timeout.total)
        clock.advance(0.6 if "has_key" in url else 0.3)
        return response(value={"weight": 42})

    client._session.get.side_effect = get
    result = getattr(client, method)("ready", timeout=1)
    assert result == (
        to_binary({"weight": 42}) if method == "get_binary" else {"weight": 42}
    )
    assert budgets == pytest.approx([1, 0.4])


def test_connection_failure_retry_does_not_restart_deadline(client, clock):
    def get(url, timeout):
        clock.advance(0.3)
        raise requests.ConnectionError("synthetic disconnect")

    client._session.get.side_effect = get
    with pytest.raises(TimeoutError):
        client.wait_key("missing", timeout=1)
    assert clock.now == pytest.approx(1)
    assert client._session.get.call_count == 1


def test_key_can_appear_before_deadline(client, clock):
    checks = []

    def get(url, timeout):
        clock.advance(0.1)
        if "has_key" in url:
            checks.append(url)
            return response(exists=len(checks) > 1)
        return response(value="ready")

    client._session.get.side_effect = get
    assert client.get_object("later", timeout=1) == "ready"
    assert clock.now < 1


def test_late_fetch_returns_default_and_closes_response(client, clock):
    late = response(value="too late")

    def get(url, timeout):
        if "has_key" in url:
            clock.advance(0.6)
            return response()
        clock.advance(timeout.total)
        return late

    client._session.get.side_effect = get
    assert client.get_object("late", timeout=1, default_value="fallback") == "fallback"
    assert clock.now == pytest.approx(1)
    late.close.assert_called_once_with()


def test_set_polling_cannot_outlive_deadline(client, clock):
    def get(url, timeout):
        clock.advance(min(0.3, timeout.total))
        return response(value={0})

    client._session.get.side_effect = get
    with pytest.raises(TimeoutError):
        client.wait_set_until_size("members", size=2, timeout=1)
    assert clock.now == pytest.approx(1)
