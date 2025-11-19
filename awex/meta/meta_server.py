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

import asyncio
import json
import multiprocessing
import os
import pickle
import random
import signal
import socket
import threading
import time
import traceback
from typing import Any, Dict, Tuple, Union

import requests
from aiohttp import web

from awex import logging
from awex.util.common import (
    from_binary,
    get_free_port,
    get_ip_address,
    to_binary,
)

logger = logging.getLogger(__name__)


class MetaServer:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.storage: Dict[str, Any] = {}
        self.app = web.Application(
            middlewares=[self.error_handler],
            client_max_size=2 * 1024**3,  # 2GB max size
        )
        self.id_counter = 0
        self.runner = None
        self.site = None
        self._server_thread = None
        self._server_error = None

        # Define routes
        self.app.router.add_get("/v1/get_binary/{key}", self.get_binary_handler)
        self.app.router.add_put("/v1/put_binary/{key}", self.put_binary_handler)
        self.app.router.add_put(
            "/v1/add_object_to_set/{key}", self.add_object_to_set_handler
        )
        self.app.router.add_get("/v1/get_json/{key}", self.get_json_handler)
        self.app.router.add_put("/v1/put_json/{key}", self.put_json_handler)
        self.app.router.add_delete("/v1/delete/{key}", self.delete_handler)
        self.app.router.add_get("/v1/health", self.health_check)
        self.app.router.add_get("/v1/keys", self.list_keys)
        self.app.router.add_get("/v1/has_key/{key}", self.has_key)
        self.app.router.add_post(
            "/v1/allocate_auto_grow_id", self.allocate_auto_grow_id
        )
        logger.info(f"[{os.getpid()}] Created meta server with {self.host}:{self.port}")

    def get_binary(self, key: str) -> bytes:
        """Get binary data from storage"""
        return self.storage[key]

    def put_binary(self, key: str, binary: bytes):
        """Store binary data"""
        self.storage[key] = binary

    def get_object(self, key: str) -> Any:
        """Get pickled object from storage"""
        return pickle.loads(self.storage[key])

    def put_object(self, key: str, obj: Any):
        """Store object as pickled data"""
        self.storage[key] = pickle.dumps(obj)

    def get_json(self, key: str) -> dict:
        """Get JSON data from storage"""
        return json.loads(self.storage[key].decode("utf-8"))

    def put_json(self, key: str, json_data: dict):
        """Store JSON data"""
        self.storage[key] = json.dumps(json_data).encode("utf-8")

    def delete(self, key: str):
        """Delete data from storage"""
        if key in self.storage:
            del self.storage[key]
        else:
            logger.warning(f"Key '{key}' not found in storage")
            raise ValueError(f"Key '{key}' not found in storage")

    def get_address(self) -> str:
        """Get server address"""
        return self.host

    def get_port(self) -> int:
        """Get server port"""
        return self.port

    def get_address_and_port(self) -> Tuple[str, int]:
        """Get server address and port"""
        return self.host, self.port

    def _run_server(self):
        """Run the server in a separate thread"""
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
        logger.info(f"[{os.getpid()}] Start server with {self.host}, {self.port}")
        try:
            # If port is 0, let the OS assign a port
            if self.port == 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    actual_port = s.getsockname()[1]
                    s.close()
                self.port = actual_port

            web.run_app(self.app, host=self.host, port=self.port, handle_signals=False)
        except BaseException as e:
            logger.exception(
                f"[{os.getpid()}] Start server with {self.host}, {self.port} failed"
            )
            self._server_error = e
            raise e

    def start(self):
        """Start the server in a daemon thread"""
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True, name="AsystemMetaServer"
        )
        self._server_thread.start()

        # Poll health check until server is ready
        max_attempts = 50  # 5 seconds with 0.1s intervals
        for _attempt in range(max_attempts):
            # Check if server thread failed
            if not self._server_thread.is_alive():
                if self._server_error:
                    raise RuntimeError(
                        f"Server failed to start on {self.host}:{self.port}: {self._server_error}"
                    ) from self._server_error
                else:
                    raise RuntimeError(
                        f"Server thread died unexpectedly on {self.host}:{self.port}"
                    )

            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/v1/health", timeout=0.5
                )
                if response.status_code == 200:
                    logger.info(f"Meta server is ready on {self.host}:{self.port}")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)

        # Timeout reached
        raise RuntimeError(
            f"Server failed to respond within timeout on {self.host}:{self.port}"
        )

    def stop(self):
        """Stop the server"""
        if self.runner:
            asyncio.run(self.runner.cleanup())

    @web.middleware
    async def error_handler(self, request, handler):
        """A middleware error handler."""
        try:
            response = await handler(request)
            if isinstance(response, web.Response):
                return response
            else:
                return web.json_response({"success": True, "data": response})
        except web.HTTPException as e:
            return web.json_response(
                {"success": False, "error": str(e)}, status=e.status_code
            )
        except Exception as e:
            logger.error("An error occurred: %s", str(e), exc_info=True)
            tb_str = "".join(
                traceback.format_exception(type(e), value=e, tb=e.__traceback__)
            )
            return web.json_response(
                {"success": False, "error": f"An error occurred: {tb_str}."}, status=500
            )

    async def get_binary_handler(self, request):
        """Handle GET binary request"""
        key = request.match_info["key"]
        if key not in self.storage:
            return web.json_response(
                {"success": False, "error": f"Key '{key}' not found"}, status=404
            )
        data = self.storage[key]

        # If data is already bytes, return it directly
        if isinstance(data, bytes):
            return web.Response(body=data, content_type="application/octet-stream")

        # Otherwise, serialize the object to binary format
        try:
            binary_data = to_binary(data)
            return web.Response(
                body=binary_data, content_type="application/octet-stream"
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to serialize object: {str(e)}"},
                status=500,
            )

    async def put_binary_handler(self, request):
        """Handle PUT binary request"""
        key = request.match_info["key"]
        data = await request.read()
        self.storage[key] = data
        return web.json_response(
            {"success": True, "message": f"Binary data stored for key '{key}'"}
        )

    async def add_object_to_set_handler(self, request):
        """Handle PUT object to set request"""
        key = request.match_info["key"]
        try:
            # Read binary data and use from_binary to deserialize
            binary_data = await request.read()
            obj = from_binary(binary_data)
            # Store the pickled data
            if key not in self.storage:
                self.storage[key] = set()
            self.storage[key].add(obj)
            return web.json_response(
                {"success": True, "message": f"Object stored for key '{key}'"}
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to deserialize object: {str(e)}"},
                status=500,
            )

    async def get_json_handler(self, request):
        """Handle GET JSON request"""
        key = request.match_info["key"]
        if key not in self.storage:
            return web.json_response(
                {"success": False, "error": f"Key '{key}' not found"}, status=404
            )
        try:
            json_data = json.loads(self.storage[key].decode("utf-8"))
            return web.json_response({"success": True, "data": json_data})
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to parse JSON: {str(e)}"},
                status=500,
            )

    async def put_json_handler(self, request):
        """Handle PUT JSON request"""
        key = request.match_info["key"]
        try:
            data = await request.json()
            self.storage[key] = json.dumps(data).encode("utf-8")
            return web.json_response(
                {"success": True, "message": f"JSON data stored for key '{key}'"}
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to store JSON: {str(e)}"},
                status=500,
            )

    async def delete_handler(self, request):
        """Handle DELETE request"""
        key = request.match_info["key"]
        try:
            self.delete(key)
            return web.json_response(
                {"success": True, "message": f"Key '{key}' deleted"}
            )
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=404)
        except Exception as e:
            logger.error("An error occurred during deletion: %s", str(e), exc_info=True)
            tb_str = "".join(
                traceback.format_exception(type(e), value=e, tb=e.__traceback__)
            )
            return web.json_response(
                {
                    "success": False,
                    "error": f"An error occurred during deletion: {tb_str}.",
                },
                status=500,
            )

    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response(
            {
                "success": True,
                "status": "healthy",
                "storage_size": len(self.storage),
                "address": self.host,
                "port": self.port,
            }
        )

    async def has_key(self, request):
        """Check if a key exists on server"""
        key = request.match_info["key"]
        return web.json_response({"success": True, "has_key": key in self.storage})

    async def list_keys(self, request):
        """List all stored keys"""
        return web.json_response({"success": True, "keys": list(self.storage.keys())})

    async def allocate_auto_grow_id(self, request):
        """Allocate an auto grow id"""
        new_id = self.id_counter
        self.id_counter += 1
        return web.json_response({"success": True, "id": new_id})


def run_meta_server_process(host: str, port: int, result_queue):
    """
    Run MetaServer in a subprocess and communicate the address/port back to parent
    This function is designed to be called by multiprocessing.Process
    """
    try:
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Set uvloop event loop policy")
        except Exception as e:
            logger.warning(f"Failed to set uvloop event loop policy: {e}")

        # Create and start the server
        server = MetaServer(host, port)
        server.start()

        # Get the actual address and port (in case port was 0)
        address, actual_port = server.get_address_and_port()

        # Send the result back to parent process
        result_queue.put((True, address, actual_port))

        # Keep the server running
        try:
            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, shutting down meta server...")
                server.stop()
                os._exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Keep the process alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down meta server...")
            server.stop()
            os._exit(0)

    except Exception as e:
        logger.exception(f"Failed to start meta server in subprocess: {e}")
        result_queue.put((False, str(e), None))
        os._exit(1)


def start_meta_server(host: str = "", port: int = 0) -> Tuple[str, int]:
    """
    Start a meta server in a subprocess using multiprocessing spawn, and return address and port
    """
    if host == "":
        host = get_ip_address()
    if port == 0:
        # Let the OS assign a random port
        port = get_free_port()

    # Get spawn context for better isolation
    ctx = multiprocessing.get_context("spawn")

    # Create a queue for communication between processes
    result_queue = ctx.Queue()

    # Create and start the subprocess
    process = ctx.Process(
        target=run_meta_server_process,
        args=(host, port, result_queue),
        name="MetaServerProcess",
    )
    # Make it a daemon process so it automatically terminates when parent exits
    process.daemon = True
    process.start()

    try:
        # Wait for the subprocess to start and get the result
        success, address, actual_port = result_queue.get(timeout=120)

        if not success:
            raise RuntimeError(f"Meta server startup failed in subprocess: {address}")

        # Store the process instance globally so it doesn't get garbage collected
        start_meta_server._server_process = process

        logger.info(
            f"[{os.getpid()}] Meta server started in subprocess (PID: {process.pid}) on {address}:{actual_port}"
        )
        return address, actual_port

    except multiprocessing.TimeoutError:
        # Clean up the process if it timed out
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        raise RuntimeError("Meta server startup timed out after 10 seconds") from None
    except Exception as e:
        # Clean up the process if it failed
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        raise RuntimeError(f"Meta server startup failed: {e}") from e


def stop_meta_server():
    """
    Stop the meta server subprocess if it's running
    """
    if hasattr(start_meta_server, "_server_process"):
        process = start_meta_server._server_process
        if process.is_alive():
            logger.info(f"Terminating meta server process (PID: {process.pid})...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                logger.info("Force killing meta server process...")
                process.kill()
                process.join()
            logger.info("Meta server process terminated")
            return True
    return False


def retry(
    fn,
    client,
    max_retries: int = 10,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            logger.error(
                f"Retry {attempt + 1}/{max_retries} failed for {fn.__name__}: {e}"
            )
            if attempt == max_retries - 1:
                raise e
            try:
                client._session.close()
            except Exception:
                logger.exception("Failed to close session")
            client._session = requests.Session()
            delay = (
                (max_delay - initial_delay) / (max_retries - 1) * attempt
                + initial_delay
                + random.random()
            )
            time.sleep(delay)


class MetaServerClient:
    def __init__(self, address: str, port: Union[int, str]):
        self._address = address
        self._port = int(port)
        self._base_url = f"http://{address}:{port}"
        self._session = requests.Session()
        self.timeout = 120

    def _retry(
        self,
        fn,
        max_retries: int = 10,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
    ):
        return retry(fn, self, max_retries, initial_delay, max_delay)

    def _get(self, url: str, timeout: float = 120) -> requests.Response:
        return self._retry(lambda: self._session.get(url, timeout=timeout))

    def _put(self, url: str, data: bytes, timeout: float = 120) -> requests.Response:
        return self._retry(lambda: self._session.put(url, data=data, timeout=timeout))

    def _delete(self, url: str, timeout: float = 120) -> requests.Response:
        return self._retry(lambda: self._session.delete(url, timeout=timeout))

    def get_binary(self, key: str, timeout: float = 0) -> bytes:
        """Get binary data from server"""
        if timeout > 0:
            self.wait_key(key, timeout)
        response = self._get(f"{self._base_url}/v1/get_binary/{key}")
        if response.status_code == 404:
            raise ValueError(f"Key '{key}' not found")
        response.raise_for_status()
        return response.content

    def put_binary(self, key: str, binary: bytes):
        """Store binary data on server"""
        response = self._put(f"{self._base_url}/v1/put_binary/{key}", data=binary)
        response.raise_for_status()
        return response.json()

    def get_object_then_delete(self, key):
        self.get_object(key, 1024**3)
        self.delete(key)

    def _has_key(self, key: str) -> bool:
        try:
            return self.has_key(key)
        except requests.exceptions.ConnectTimeout:
            return False
        except requests.exceptions.ReadTimeout:
            return False

    def wait_key(self, key: str, timeout: float):
        # Backoff strategy: start with 0.5s, max 3s, exponential backoff
        backoff_interval = 0.2
        max_backoff = 1
        backoff_multiplier = 1.5
        start_time = time.time()
        last_print_time = 0
        last_check_time = time.time()
        while not self._has_key(key):
            if time.time() - last_check_time > 3:
                time.sleep(3)
            if time.time() - start_time > 60:
                backoff_interval = 3
            time.sleep(backoff_interval)
            last_check_time = time.time()
            elapsed = time.time() - start_time
            timeout -= backoff_interval

            if timeout <= 0:
                raise TimeoutError(f"Timeout waiting for key '{key}'")

            # Increase backoff interval exponentially, capped at max_backoff
            backoff_interval = min(backoff_interval * backoff_multiplier, max_backoff)

            # Log every 3 seconds using last_print_time
            if elapsed - last_print_time >= 3:
                logger.info(
                    f"Waiting for key '{key}' from meta server, waited {elapsed:.1f}s, "
                    f"remaining wait time: {timeout:.1f}s, backoff: {backoff_interval:.1f}s"
                )
                last_print_time = elapsed

    def get_object(
        self, key: str, timeout: float = 0, default_value: Any = None
    ) -> Any:
        """Get object from server"""
        if timeout > 0:
            try:
                self.wait_key(key, timeout)
            except TimeoutError:
                if default_value is not None:
                    return default_value
                raise
        response = self._get(f"{self._base_url}/v1/get_binary/{key}")
        if response.status_code == 404:
            return default_value
        response.raise_for_status()

        # Use from_binary to deserialize the object
        binary_data = response.content
        return from_binary(binary_data)

    def put_object(self, key: str, obj: Any):
        """Store object on server"""
        # Use to_binary to serialize the object
        binary_data = to_binary(obj)
        response = self._put(f"{self._base_url}/v1/put_binary/{key}", data=binary_data)
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"Failed to put object {key}: {result.get('error')}")
        return result

    def add_object_to_set(self, key: str, obj: Any):
        """Store object to set on server"""
        # Use to_binary to serialize the object
        binary_data = to_binary(obj)
        response = self._put(
            f"{self._base_url}/v1/add_object_to_set/{key}", data=binary_data
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(
                f"Failed to add object to set {key}: {result.get('error')}"
            )
        return result

    def get_set(self, key: str) -> set:
        """Get set from server"""
        return self.get_object(key)

    def wait_set_until_size(
        self, key: str, size: int, timeout: float, delete_after_wait: bool = False
    ):
        """Wait for set to reach a certain size"""
        start_time = time.time()
        last_print_time = 0
        # Backoff strategy: start with 0.2s, max 3s, exponential backoff
        backoff_interval = 0.2
        max_backoff = 1
        backoff_multiplier = 1.5

        while True:
            current_set = self.get_object(
                key, timeout=backoff_interval, default_value=set()
            )
            if len(current_set) >= size:
                break
            if time.time() - start_time > 60:
                backoff_interval = 3
                max_backoff = 3
            time.sleep(backoff_interval)
            elapsed = time.time() - start_time
            remaining_timeout = max(0, timeout - elapsed)
            if remaining_timeout <= 0:
                raise TimeoutError(
                    f"Timeout waiting for set {key} to reach size {size}"
                )

            # Increase backoff interval exponentially, capped at max_backoff
            backoff_interval = min(backoff_interval * backoff_multiplier, max_backoff)

            # Log every 3 seconds using last_print_time
            if elapsed - last_print_time >= 3:
                logger.info(
                    f"Waiting for set {key} to reach size {size}, waited {elapsed:.1f}s, remaining wait time: {remaining_timeout:.1f}s, backoff: {backoff_interval:.1f}s"
                )
                last_print_time = elapsed
        if delete_after_wait:
            self.delete_if_exists(key)

    def barrier(self, key: str, rank: int, world_size: int, timeout: float):
        """Barrier for a set"""
        self.add_object_to_set(key, (rank))
        self.wait_set_until_size(key, world_size, timeout)
        if rank == 0:
            self.delete_if_exists(key)

    def get_json(self, key: str) -> dict:
        """Get JSON data from server"""
        response = self._get(f"{self._base_url}/v1/get_json/{key}")
        if response.status_code == 404:
            raise ValueError(f"Key '{key}' not found")
        response.raise_for_status()
        result = response.json()
        return result.get("data")

    def put_json(self, key: str, json_data: dict):
        """Store JSON data on server"""
        response = self._retry(
            lambda: self._session.put(
                f"{self._base_url}/v1/put_json/{key}",
                json=json_data,
                timeout=self.timeout,
            )
        )
        response.raise_for_status()
        return response.json()

    def delete(self, key: str):
        """Delete data from server"""
        response = self._delete(f"{self._base_url}/v1/delete/{key}")
        response.raise_for_status()
        return response.json()

    def delete_if_exists(self, key: str):
        """Delete data from server if it exists"""
        try:
            self.delete(key)
        except Exception:
            pass

    def health_check(self) -> dict:
        """Check server health"""
        response = self._get(f"{self._base_url}/v1/health")
        response.raise_for_status()
        return response.json()

    def has_key(self, key: str) -> bool:
        """Check if a key exists on server"""
        response = self._get(f"{self._base_url}/v1/has_key/{key}")
        response.raise_for_status()
        return response.json().get("has_key")

    def list_keys(self) -> list:
        """List all keys on server"""
        response = self._get(f"{self._base_url}/v1/keys")
        response.raise_for_status()
        result = response.json()
        return result.get("keys", [])

    def allocate_auto_grow_id(self) -> int:
        """Allocate an auto grow id"""
        response = self._retry(
            lambda: self._session.post(
                f"{self._base_url}/v1/allocate_auto_grow_id", timeout=self.timeout
            )
        )
        response.raise_for_status()
        result = response.json()
        return int(result.get("id"))

    def close(self):
        """Close the client session"""
        self._session.close()


if __name__ == "__main__":
    # Start the meta server
    address, port = start_meta_server("", 0)

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down meta server...")
        stop_meta_server()
