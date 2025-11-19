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
import requests

from awex.meta.meta_server import (
    MetaServer,
    MetaServerClient,
    start_meta_server,
)
from awex.util.common import from_binary, to_binary


# Define test class at module level so it can be pickled
# Note: This is not a pytest test class, just a helper class for testing
class StructClass:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))


class TestMetaServer:
    """Test MetaServer class directly"""

    def test_meta_server_initialization(self):
        server = MetaServer("127.0.0.1", 8080)
        assert server.host == "127.0.0.1"
        assert server.port == 8080
        assert len(server.storage) == 0

    def test_binary_operations(self):
        server = MetaServer()
        test_data = b"test binary data"

        server.put_binary("test_key", test_data)
        assert server.get_binary("test_key") == test_data

    def test_object_operations(self):
        server = MetaServer()
        test_obj = {"name": "test", "value": 123, "nested": {"data": [1, 2, 3]}}

        server.put_object("test_key", test_obj)
        assert server.get_object("test_key") == test_obj

        # Test with a class instance (now defined at module level)
        complex_obj = StructClass("test_class", 456)
        server.put_object("complex_key", complex_obj)
        assert server.get_object("complex_key") == complex_obj

    def test_json_operations(self):
        server = MetaServer()
        test_json = {"name": "test", "value": 123}

        server.put_json("test_key", test_json)
        assert server.get_json("test_key") == test_json

    def test_key_error_on_missing_binary(self):
        server = MetaServer()
        with pytest.raises(KeyError):
            server.get_binary("missing_key")

    def test_key_error_on_missing_json(self):
        server = MetaServer()
        with pytest.raises(KeyError):
            server.get_json("missing_key")

    def test_delete_operations(self):
        server = MetaServer()

        # Test deleting existing key
        server.put_binary("test_key", b"test data")
        assert "test_key" in server.storage

        server.delete("test_key")
        assert "test_key" not in server.storage

        # Test deleting non-existent key
        with pytest.raises(ValueError, match="Key 'missing_key' not found in storage"):
            server.delete("missing_key")


class TestMetaServerHTTP:
    """Test MetaServer via HTTP endpoints"""

    @pytest.fixture
    def server(self):
        server = MetaServer("127.0.0.1", 0)
        server.start()
        yield server
        server.stop()

    @pytest.fixture
    def base_url(self, server):
        return f"http://{server.host}:{server.port}"

    def test_health_check(self, base_url):
        response = requests.get(f"{base_url}/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"

    def test_binary_http_operations(self, base_url):
        test_data = b"test binary data"

        # PUT binary
        response = requests.put(f"{base_url}/v1/put_binary/test_key", data=test_data)
        assert response.status_code == 200

        # GET binary
        response = requests.get(f"{base_url}/v1/get_binary/test_key")
        assert response.status_code == 200
        assert response.content == test_data

        # GET non-existent binary
        response = requests.get(f"{base_url}/v1/get_binary/missing_key")
        assert response.status_code == 404

    def test_json_http_operations(self, base_url):
        test_json = {"name": "test", "value": 123}

        # PUT JSON
        response = requests.put(f"{base_url}/v1/put_json/test_key", json=test_json)
        assert response.status_code == 200

        # GET JSON
        response = requests.get(f"{base_url}/v1/get_json/test_key")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == test_json

        # GET non-existent JSON
        response = requests.get(f"{base_url}/v1/get_json/missing_key")
        assert response.status_code == 404

    def test_object_http_operations(self, base_url):
        # Test with a complex object
        test_obj = {"name": "test", "value": 123, "nested": {"data": [1, 2, 3]}}

        # PUT object using binary format
        import pickle
        import struct

        pickled_data = pickle.dumps(test_obj)
        data_len = len(pickled_data)
        binary_data = struct.pack("!I", data_len) + pickled_data

        response = requests.put(f"{base_url}/v1/put_binary/test_key", data=binary_data)
        assert response.status_code == 200

        # GET object (returns binary data)
        response = requests.get(f"{base_url}/v1/get_binary/test_key")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

        # Deserialize the binary response
        binary_response = response.content
        data_len = struct.unpack("!I", binary_response[:4])[0]
        pickled_response = binary_response[4 : 4 + data_len]
        retrieved_obj = pickle.loads(pickled_response)
        assert retrieved_obj == test_obj

        # Test with a class instance (now defined at module level)
        complex_obj = StructClass("test_class", 456)

        # PUT complex object
        pickled_data = pickle.dumps(complex_obj)
        data_len = len(pickled_data)
        binary_data = struct.pack("!I", data_len) + pickled_data

        response = requests.put(
            f"{base_url}/v1/put_binary/complex_key", data=binary_data
        )
        assert response.status_code == 200

        # GET complex object
        response = requests.get(f"{base_url}/v1/get_binary/complex_key")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

        # Deserialize the binary response
        binary_response = response.content
        data_len = struct.unpack("!I", binary_response[:4])[0]
        pickled_response = binary_response[4 : 4 + data_len]
        retrieved_complex = pickle.loads(pickled_response)
        assert retrieved_complex == complex_obj

        # GET non-existent object
        response = requests.get(f"{base_url}/v1/get_binary/missing_key")
        assert response.status_code == 404

    def test_list_keys(self, base_url):
        # PUT some data
        requests.put(f"{base_url}/v1/put_json/key1", json={"data": "value1"})
        requests.put(f"{base_url}/v1/put_json/key2", json={"data": "value2"})

        # List keys
        response = requests.get(f"{base_url}/v1/keys")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "key1" in data["keys"]
        assert "key2" in data["keys"]

    def test_delete_http_operations(self, base_url):
        # PUT some data first
        test_data = b"test binary data"
        response = requests.put(f"{base_url}/v1/put_binary/test_key", data=test_data)
        assert response.status_code == 200

        # Verify data exists
        response = requests.get(f"{base_url}/v1/get_binary/test_key")
        assert response.status_code == 200
        assert response.content == test_data

        # DELETE the data
        response = requests.delete(f"{base_url}/v1/delete/test_key")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Key 'test_key' deleted"

        # Verify data is deleted
        response = requests.get(f"{base_url}/v1/get_binary/test_key")
        assert response.status_code == 404

        # Test deleting non-existent key
        response = requests.delete(f"{base_url}/v1/delete/missing_key")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"]

    def test_delete_with_different_data_types(self, base_url):
        # Test deleting JSON data
        test_json = {"name": "test", "value": 123}
        response = requests.put(f"{base_url}/v1/put_json/json_key", json=test_json)
        assert response.status_code == 200

        response = requests.delete(f"{base_url}/v1/delete/json_key")
        assert response.status_code == 200

        # Verify JSON data is deleted
        response = requests.get(f"{base_url}/v1/get_json/json_key")
        assert response.status_code == 404

        # Test deleting object data
        import pickle
        import struct

        test_obj = {"name": "test", "value": 123, "nested": {"data": [1, 2, 3]}}
        pickled_data = pickle.dumps(test_obj)
        data_len = len(pickled_data)
        binary_data = struct.pack("!I", data_len) + pickled_data

        response = requests.put(f"{base_url}/v1/put_binary/obj_key", data=binary_data)
        assert response.status_code == 200

        response = requests.delete(f"{base_url}/v1/delete/obj_key")
        assert response.status_code == 200

        # Verify object data is deleted
        response = requests.get(f"{base_url}/v1/get_binary/obj_key")
        assert response.status_code == 404

    def test_set_http_operations(self, base_url):
        """Test set operations via HTTP endpoints"""
        # Test adding objects to a set
        test_obj1 = ("test1", 123)  # Use tuple instead of dict
        test_obj2 = ("test2", 456)  # Use tuple instead of dict
        test_obj3 = StructClass("test_class", 789)

        # Add objects to set using to_binary
        response = requests.put(
            f"{base_url}/v1/add_object_to_set/test_set", data=to_binary(test_obj1)
        )
        assert response.status_code == 200

        response = requests.put(
            f"{base_url}/v1/add_object_to_set/test_set", data=to_binary(test_obj2)
        )
        assert response.status_code == 200

        response = requests.put(
            f"{base_url}/v1/add_object_to_set/test_set", data=to_binary(test_obj3)
        )
        assert response.status_code == 200

        # Get the set using get_binary
        response = requests.get(f"{base_url}/v1/get_binary/test_set")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

        # Deserialize using from_binary
        retrieved_set = from_binary(response.content)

        # Verify the set contains all three objects
        assert isinstance(retrieved_set, set)
        assert len(retrieved_set) == 3
        assert test_obj1 in retrieved_set
        assert test_obj2 in retrieved_set
        assert test_obj3 in retrieved_set

        # Test adding duplicate object (should not affect set size)
        response = requests.put(
            f"{base_url}/v1/add_object_to_set/test_set", data=to_binary(test_obj1)
        )
        assert response.status_code == 200

        retrieved_set = from_binary(
            requests.get(f"{base_url}/v1/get_binary/test_set").content
        )
        assert len(retrieved_set) == 3

        # Test getting non-existent set
        response = requests.get(f"{base_url}/v1/get_binary/missing_set")
        assert response.status_code == 404


class TestMetaServerClient:
    """Test MetaServerClient class"""

    @pytest.fixture
    def server_and_client(self):
        address, port = start_meta_server("127.0.0.1", 0)
        client = MetaServerClient(address, port)
        yield client
        client.close()

    def test_client_initialization(self, server_and_client):
        client = server_and_client
        assert client._address == "127.0.0.1"
        assert client._port > 0
        assert client._base_url.startswith("http://127.0.0.1:")

    def test_binary_operations(self, server_and_client):
        client = server_and_client
        test_data = b"test binary data"

        # PUT binary
        result = client.put_binary("test_key", test_data)
        assert result["success"] is True

        # GET binary
        retrieved_data = client.get_binary("test_key")
        assert retrieved_data == test_data

        # GET non-existent binary
        with pytest.raises(ValueError):
            client.get_binary("missing_key")

    def test_json_operations(self, server_and_client):
        client = server_and_client
        test_json = {"name": "test", "value": 123}

        # PUT JSON
        result = client.put_json("test_key", test_json)
        assert result["success"] is True

        # GET JSON
        retrieved_json = client.get_json("test_key")
        assert retrieved_json == test_json

        # GET non-existent JSON
        with pytest.raises(ValueError):
            client.get_json("missing_key")

    def test_object_operations(self, server_and_client):
        client = server_and_client
        # Test with a complex object that can't be JSON serialized
        test_obj = {"name": "test", "value": 123, "nested": {"data": [1, 2, 3]}}

        # PUT object
        result = client.put_object("test_key", test_obj)
        assert result["success"] is True

        # GET object
        retrieved_obj = client.get_object("test_key")
        assert retrieved_obj == test_obj

        # Test with a more complex object (class instance)
        complex_obj = StructClass("test_class", 456)

        # PUT complex object
        result = client.put_object("complex_key", complex_obj)
        assert result["success"] is True

        # GET complex object
        retrieved_complex = client.get_object("complex_key")
        assert retrieved_complex == complex_obj

        # Test with None (should return None)
        result = client.put_object("none_key", None)
        assert result["success"] is True

        retrieved_none = client.get_object("none_key")
        assert retrieved_none is None

        # Test with a list containing the class instance
        list_with_obj = [1, 2, complex_obj, "test"]
        result = client.put_object("list_key", list_with_obj)
        assert result["success"] is True

        retrieved_list = client.get_object("list_key")
        assert retrieved_list == list_with_obj

    def test_health_check(self, server_and_client):
        client = server_and_client
        health = client.health_check()
        assert health["success"] is True
        assert health["status"] == "healthy"
        assert "storage_size" in health
        assert "address" in health
        assert "port" in health

    def test_list_keys(self, server_and_client):
        client = server_and_client

        # PUT some data
        client.put_json("key1", {"data": "value1"})
        client.put_json("key2", {"data": "value2"})

        # List keys
        keys = client.list_keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_delete_operations(self, server_and_client):
        client = server_and_client

        # PUT some data first
        test_data = b"test binary data"
        client.put_binary("test_key", test_data)

        # Verify data exists
        retrieved_data = client.get_binary("test_key")
        assert retrieved_data == test_data

        # DELETE the data
        result = client.delete("test_key")
        assert result["success"] is True
        assert result["message"] == "Key 'test_key' deleted"

        # Verify data is deleted
        with pytest.raises(ValueError, match="Key 'test_key' not found"):
            client.get_binary("test_key")

        # Test deleting non-existent key
        with pytest.raises(requests.exceptions.HTTPError):
            client.delete("missing_key")

    def test_delete_with_different_data_types(self, server_and_client):
        client = server_and_client

        # Test deleting JSON data
        test_json = {"name": "test", "value": 123}
        client.put_json("json_key", test_json)

        result = client.delete("json_key")
        assert result["success"] is True

        # Verify JSON data is deleted
        with pytest.raises(ValueError, match="Key 'json_key' not found"):
            client.get_json("json_key")

        # Test deleting object data
        test_obj = {"name": "test", "value": 123, "nested": {"data": [1, 2, 3]}}
        client.put_object("obj_key", test_obj)

        result = client.delete("obj_key")
        assert result["success"] is True

        # Verify object data is deleted
        retrieved_obj = client.get_object("obj_key")
        assert retrieved_obj is None  # get_object returns None for missing keys

    def test_set_operations(self, server_and_client):
        """Test set operations using client methods"""
        client = server_and_client

        # Test adding objects to a set
        test_obj1 = ("test1", 123)  # Use tuple instead of dict
        test_obj2 = ("test2", 456)  # Use tuple instead of dict
        test_obj3 = StructClass("test_class", 789)

        # Add first object to set
        result = client.add_object_to_set("test_set", test_obj1)
        assert result["success"] is True

        # Add second object to set
        result = client.add_object_to_set("test_set", test_obj2)
        assert result["success"] is True

        # Add third object (class instance) to set
        result = client.add_object_to_set("test_set", test_obj3)
        assert result["success"] is True

        # Get the set using get_object
        retrieved_set = client.get_object("test_set")

        # Verify the set contains all three objects
        assert isinstance(retrieved_set, set)
        assert len(retrieved_set) == 3
        assert test_obj1 in retrieved_set
        assert test_obj2 in retrieved_set
        assert test_obj3 in retrieved_set

        # Test adding duplicate object (should not affect set size)
        result = client.add_object_to_set("test_set", test_obj1)
        assert result["success"] is True

        retrieved_set = client.get_object("test_set")

        # Set size should still be 3 (duplicate not added)
        assert len(retrieved_set) == 3

        # Test getting non-existent set
        retrieved_set = client.get_object("missing_set")
        assert retrieved_set is None

        # Test with timeout - should raise TimeoutError
        with pytest.raises(TimeoutError, match="Timeout waiting for key 'missing_set'"):
            client.get_object("missing_set", timeout=0.1)

    def test_set_operations_with_timeout(self, server_and_client):
        """Test set operations with timeout functionality"""
        client = server_and_client

        # Test timeout when waiting for non-existent set
        with pytest.raises(TimeoutError, match="Timeout waiting for key 'missing_set'"):
            client.get_object("missing_set", timeout=0.1)

        # Test with default value
        retrieved_set = client.get_object(
            "missing_set", timeout=0.1, default_value=set()
        )
        assert retrieved_set == set()

        # Test successful retrieval after adding
        test_obj = ("test", 123)  # Use tuple instead of dict
        client.add_object_to_set("timeout_test_set", test_obj)

        # Should succeed immediately
        retrieved_set = client.get_object("timeout_test_set", timeout=1.0)
        assert isinstance(retrieved_set, set)
        assert test_obj in retrieved_set


class TestStartMetaServer:
    """Test start_meta_server function"""

    def test_start_meta_server(self):
        address, port = start_meta_server("127.0.0.1", 0)
        assert address == "127.0.0.1"
        assert port > 0

        # Test that server is actually running
        client = MetaServerClient(address, port)
        health = client.health_check()
        assert health["success"] is True
        client.close()

    def test_start_meta_server_empty_host(self):
        address, port = start_meta_server("", 0)
        assert address != ""
        assert port > 0

        # Test that server is actually running
        client = MetaServerClient(address, port)
        health = client.health_check()
        assert health["success"] is True
        client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
