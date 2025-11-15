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
import torch.distributed as dist
import os
from concurrent.futures import ThreadPoolExecutor
import time

from awex.config import InferenceConfig
from awex.meta.meta_server import start_meta_server
from awex.meta.meta_server import MetaServerClient
from awex.tests.test_utils import simple_torch_model
from awex.transfer.transfer_plan import TransferPlanBuilder, slice_tensor
from awex.util.process_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from awex.util.common import get_free_port, simple_hf_config
from awex import logging

logger = logging.getLogger(__name__)
_env_backup = dict(os.environ)


def create_mocked_mcore_engine():
    os.environ["NCCL_DEBUG"] = "WARNING"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_MAX_NCHANNELS"] = "8"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ALLOC"
    os.environ["GLOO_USE_LIBUV"] = "0"
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["LOCAL_RANK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["DEVICE"] = "1"
    ip, port = start_meta_server()
    config = {
        "meta_server_addr": f"{ip}:{port}",
        "comm_backend": "nccl",
        "enable_debug_mode": False,
    }
    from awex.engine.mcore import MegatronEngine

    model, hf_config = simple_torch_model()
    return MegatronEngine(config, hf_config, model)


@pytest.mark.skipif(
    torch.cuda.device_count() <= 1,
    reason="Only one GPU present",
)
def test_weights_writer():
    mcore_engine = create_mocked_mcore_engine()
    weights_writer = mcore_engine.weights_exchange_writer
    print(f"backend.meta_server_addr: {mcore_engine.meta_server_addr}")
    meta_server_client = MetaServerClient(*mcore_engine.meta_server_addr.split(":"))
    meta_server_client.put_object("num_infer_engines", 1)
    infer_engine_config = InferenceConfig(model_path="mock")
    hf_config = mcore_engine.hf_config
    meta_server_client.put_object(
        "infer_conf",
        {
            "infer_atten_tp_size": 1,
            "infer_world_size": 1,
            "infer_engine_config": infer_engine_config,
            "hf_config": simple_hf_config(hf_config),
        },
    )

    # Start the reader process first to put master_info
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=weights_reader, args=(mcore_engine.meta_server_addr,))
    p.start()

    # Wait for the reader to put master_info
    max_wait_time = 30
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            master_info = meta_server_client.get_object("master_info", timeout=1)
            logger.info(f"Found master_info: {master_info}")
            break
        except Exception:
            logger.exception("Failed to get master info")
            time.sleep(0.5)
    else:
        raise TimeoutError("Reader did not put master_info within 30 seconds")

    def put_infer_params_meta():
        while not hasattr(weights_writer, "parameters_meta"):
            time.sleep(1)
        meta_server_client.put_object(
            "infer_params_meta", weights_writer.parameters_meta
        )

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(put_infer_params_meta)
    weights_writer.write_weights(step_id=0)
    weights_writer.finish_step(step_id=0)

    # Wait for the reader process to finish
    p.kill()
    os.environ.clear()
    os.environ.update(_env_backup)


def init_process_group(rank, world_size, port):
    """Initialize the default torch process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)


def weights_reader(meta_server_addr):
    os.environ["LOCAL_RANK"] = "0"
    os.environ["DEVICE"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.cuda.set_device(0)
    init_process_group(0, 1, get_free_port())
    os.environ.pop("MASTER_ADDR")
    os.environ.pop("MASTER_PORT")
    os.environ.pop("RANK")
    os.environ.pop("WORLD_SIZE")

    meta_server_client = MetaServerClient(*meta_server_addr.split(":"))
    # Put master_info to meta server (like the real NCCLWorkerWeightsReader does)
    master_address = "localhost"
    master_port = get_free_port()
    master_info = (master_address, master_port)
    meta_server_client.put_object("master_info", master_info)
    logger.info(f"Put master info: {master_info}")

    timeout = 180
    builder = TransferPlanBuilder(
        1,
        1,
        1,
        True,
    )
    import copy

    parameters_meta = meta_server_client.get_object(
        "training_params_meta", timeout=timeout
    )
    logger.info("Get training params meta from meta server")
    # Create different metadata for inference and training to avoid duplicates
    # In the real system, inference and training have different shard configurations
    inference_meta = copy.deepcopy(parameters_meta)
    training_meta = copy.deepcopy(parameters_meta)

    # Modify inference metadata to have different shard configurations
    # This simulates the real scenario where inference and training have different sharding
    for param_meta in inference_meta:
        for shard in param_meta.shards:
            # Change the rank to simulate different inference ranks
            # shard.global_rank += 1  # Inference ranks start after training ranks
            shard.engine_rank = 0  # Set engine rank for inference

    transfer_plan = builder.build_local_transfer_plan(inference_meta, training_meta, 0)
    num_operations = sum(
        len(operations) for operations in transfer_plan.operations.values()
    )
    logger.info(f"Number of operations of reads: {num_operations}")
    weights_update_group = init_weights_update_group(
        master_address=master_address,
        master_port=master_port,
        rank=0,
        world_size=2,
        group_name="weights_exchange",
        role="inference",
    )
    dist.barrier(group=weights_update_group, device_ids=[torch.cuda.current_device()])
    logger.info("Start to test NCCL ready for rank")
    dist.recv(torch.tensor(1).cuda(), src=1, group=weights_update_group)
    logger.info("NCCL ready: recv tensor from rank 1")
    logger.info("Start to receive weights")
    logger.info(f"Recv ranks: {transfer_plan.operations.keys()}")
    setup_batch_isend_irecv(weights_update_group, 0, 2)
    # Debug: Check the first few operations to understand tensor shapes
    logger.info("Debug: Checking tensor shapes for first 3 operations")
    for rank, operations in transfer_plan.operations.items():
        for i, operation in enumerate(operations[:3]):
            logger.info(
                f"Operation {i}: name={operation.recv_shard_meta.name}, "
                f"shape={operation.recv_shard_meta.shape}, "
                f"dtype={operation.recv_shard_meta.dtype}"
            )
        break

    tensors_map = {}
    for rank, operations in transfer_plan.operations.items():
        for operation in operations:
            tensor = torch.ones(
                operation.recv_shard_meta.shape,
                device=f"cuda:{torch.cuda.current_device()}",
                dtype=operation.send_shard_meta.dtype,
            )
            tensors_map[id(operation)] = tensor
    torch.cuda.synchronize(device=torch.cuda.current_device())
    p2p_ops = []
    for rank, operations in transfer_plan.operations.items():
        logger.info(f"Start to recv from rank: {rank}, operations {operations[:5]} ")
        for operation in operations:
            tensor = tensors_map[id(operation)]
            original_shape = tensor.shape
            tensor = slice_tensor(tensor, operation, False)
            logger.info(
                f"Tensor {operation.recv_shard_meta.name}: original_shape={original_shape}, "
                f"sliced_shape={tensor.shape}, dtype={tensor.dtype}"
            )
            p2p_op = dist.P2POp(
                dist.irecv,
                tensor,
                rank,
                group=weights_update_group,
            )
            p2p_ops.append(p2p_op)
    logger.info(f"Start to batch recv weights with {len(p2p_ops)} operations")

    # Debug: Check if tensors are properly allocated
    logger.info("Debug: Checking tensor allocation")
    for i, operation in enumerate(list(transfer_plan.operations.values())[0][:3]):
        tensor = tensors_map[id(operation)]
        logger.info(
            f"Tensor {i}: {operation.recv_shard_meta.name}, "
            f"shape={tensor.shape}, dtype={tensor.dtype}, "
            f"device={tensor.device}, is_contiguous={tensor.is_contiguous()}"
        )

    reqs = dist.batch_isend_irecv(p2p_ops)
    logger.info(f"Started batch_isend_irecv with {len(reqs)} requests")

    # Wait for each request with timeout
    for i, req in enumerate(reqs):
        req.wait()

    torch.cuda.synchronize(device=torch.cuda.current_device())
    logger.info("Finished receiving weights")
    dist.barrier(group=weights_update_group, device_ids=[torch.cuda.current_device()])
    logger.info("Start to destroy process group")
    dist.destroy_process_group()
    logger.info("Destroyed process group")
