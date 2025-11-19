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
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.distributed as dist

from awex import logging
from awex.config import InferenceConfig
from awex.meta.meta_server import MetaServerClient, start_meta_server
from awex.tests.test_utils import megatron_model_from_hf
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.util.common import get_free_port, simple_hf_config
from awex.util.process_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)

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
    # Training rank uses GPU 1 (logical device index 1 in this process)
    os.environ["LOCAL_RANK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["DEVICE"] = "1"
    # Ensure current CUDA device matches DEVICE before Megatron init so
    # that Megatron constructs its model directly on GPU 1.
    torch.cuda.set_device(int(os.environ["DEVICE"]))
    ip, port = start_meta_server()
    config = {
        "meta_server_addr": f"{ip}:{port}",
        "comm_backend": "nccl",
        "enable_debug_mode": True,
    }
    from awex.engine.mcore import MegatronEngine

    model, hf_config = megatron_model_from_hf()
    return MegatronEngine(config, hf_config, model)


@pytest.mark.skipif(
    torch.cuda.device_count() <= 1,
    reason="Only one GPU present",
)
def test_weights_writer():
    # Create Megatron engine first so that Megatron can initialize its
    # own parallel state and CUDA context without relying on a
    # pre-existing default torch process group.
    mcore_engine = create_mocked_mcore_engine()
    # Initialize process group for the writer side on GPU 1
    torch.cuda.set_device(1)
    os.environ["RANK"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["DEVICE"] = "1"
    os.environ["LOCAL_RANK"] = "1"
    init_process_group(0, 1, get_free_port())
    # Initialize Megatron parallel state
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    mcore_engine.initialize()
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
    while not p.is_alive():
        time.sleep(0.1)
    logger.info(f"Starting reader process {p.pid}")

    # Wait for the reader to put master_info
    max_wait_time = 30
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            master_info = meta_server_client.get_object("master_info", timeout=1)
            logger.info(f"Found master_info: {master_info}")
            break
        except Exception as e:
            logger.error(f"Failed to get master info: {e}")
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

    # Wait for the reader process to finish with timeout
    p.join(timeout=10)
    if p.is_alive():
        logger.warning("Reader process did not finish, terminating...")
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            logger.error("Reader process did not terminate, killing...")
            p.kill()
            p.join()

    # Clean up process group
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.error(f"Error destroying process group: {e}")

    os.environ.clear()
    os.environ.update(_env_backup)


def init_process_group(rank, world_size, port):
    """Initialize the default torch process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def weights_reader(meta_server_addr):
    # Set up signal handler for graceful shutdown
    def cleanup_and_exit(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            sys.exit(1)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

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
    logger.info(
        f"Transfer plan operations by rank: {[(rank, len(ops)) for rank, ops in transfer_plan.operations.items()]}"
    )
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
    for _rank, operations in transfer_plan.operations.items():
        for i, operation in enumerate(operations[:3]):
            logger.info(
                f"Operation {i}: name={operation.recv_shard_meta.name}, "
                f"shape={operation.recv_shard_meta.shape}, "
                f"dtype={operation.recv_shard_meta.dtype}"
            )
        break

    # Parameters are keyed by their shard name to mirror how the writer passes tensors.
    parameters = {}
    logger.info(f"Create tensors at device cuda:{torch.cuda.current_device()}")
    for _rank, operations in transfer_plan.operations.items():
        for operation in operations:
            param_name = operation.recv_shard_meta.name
            if param_name in parameters:
                continue
            tensor = torch.ones(
                operation.recv_shard_meta.shape,
                device=f"cuda:{torch.cuda.current_device()}",
                dtype=operation.recv_shard_meta.dtype,
            )
            parameters[param_name] = tensor
    torch.cuda.synchronize(device=torch.cuda.current_device())

    # Build recv operations in round-robin order to match the sender's round-robin pattern
    # The sender uses nccl_build_send_ops which interleaves operations across ranks
    all_ranks = list(transfer_plan.operations.keys())  # Preserve plan's order
    p2p_ops = nccl_build_recv_ops(parameters, transfer_plan, weights_update_group)
    logger.info(
        f"Reader (rank 0): Building recv operations from sender ranks: {all_ranks}"
    )

    # Debug: Check if tensors are properly allocated
    logger.info("Debug: Verifying recv operations were created correctly")
    first_rank = list(transfer_plan.operations.keys())[0]
    first_ops = transfer_plan.operations[first_rank][:3]
    for i, operation in enumerate(first_ops):
        tensor = parameters[operation.recv_shard_meta.name]
        logger.info(
            f"Tensor {i}: {operation.recv_shard_meta.name}, "
            f"shape={tensor.shape}, dtype={tensor.dtype}, "
            f"device={tensor.device}, is_contiguous={tensor.is_contiguous()}"
        )

    logger.info(f"Start to receive weights with {len(p2p_ops)} operations")

    # Execute recv operations via batch_send_recv to share the same
    # scheduling and stream assignment logic as the production reader.
    logger.info(f"Test reader: Executing {len(p2p_ops)} recv ops via batch_send_recv")
    batch_send_recv(send_ops=[], recv_ops=p2p_ops, blocking=True, use_group=True)
    logger.info("All recv operations completed, synchronizing CUDA")
    torch.cuda.synchronize(device=torch.cuda.current_device())
    logger.info("Finished receiving weights")

    # Barrier can also hang, so add timeout
    logger.info("Waiting at barrier")
    try:
        dist.barrier(
            group=weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info("Barrier completed")
    except TimeoutError:
        logger.error("Barrier timed out")
        raise

    logger.info("Start to destroy process group")
    dist.destroy_process_group()
    logger.info("Destroyed process group")
