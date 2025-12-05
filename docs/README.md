**Awex** is a high-performance RL training-inference **weight synchronization** framework,
designed to enable **second-level parameter updates** from training to inference in RL workflows.
It minimizes iteration latency, ensuring rollout phases consistently use the latest model.

## Architecture

The Awex weight exchange framework consists primarily of three components:

- **WeightWriter**: Runs within each training process, responsible for metadata collection and reporting of weight shards for the current training process, weight convert, resharding transfer plan construction, weight transmission, and other functions;
- **WeightReader**: Runs on the control process of each inference instance, which starts a WorkerWeightsReader on each GPU managed by the inference instance, corresponding to the WeightWriter of the training process. Responsible for metadata collection and reporting of weight shards for each inference process, weight convert, resharding transfer plan construction, weight reception, and other functions;
- **MetaServer**: Job-level global server for service discovery and weight metadata exchange between training and inference engines, as well as event notification functions in co-located scenarios;

<div align="center">
  <img width="95%" alt="Apache Fory logo" src="images/awex_arch.png"><br>
</div>

The core functional modules of weight exchange consist mainly of 5 parts:

- **Unified training-inference weight convert**: Responsible for converting weights from training and inference engines with **different parallelism strategies and tensor layouts** into a **unified format** for subsequent weight metadata calculation and weight transmission;
- **Global weight metadata calculation and exchange**: After converting training and inference weights into a unified format, collects all weight shard metadata from each worker and reports to Meta Server for subsequent weight transmission plan construction;
- **P2P weight transmission execution plan**: Training and inference engines obtain global weight shard metadata from all workers, then separately construct peer-to-peer deterministic transfer plan for sending and receiving;
- **NCCL weight transmission**: Uses NCCL's send/recv API for peer-to-peer weight transmission based on the constructed transmission plan;
- **RDMA weight transmission**: Uses NUMA affinity and RDMA communication for globally load-balanced transfer plan for weight updates;

### (1) Unified Training-Inference Weight Convert

Due to different computational workloads, training and inference engines generally adopt different parallelism strategies. Megatron training engine uses 5D parallelism strategy, DeepSpeed/FSDP uses Zero + DP data parallelism, while SGLang and VLLM inference engines mostly use DP + TP + EP. Additionally, different engines perform **fusion, transposition, and quantization** optimizations on weights after loading to adapt to high-performance operators.

To eliminate differences between different engines for subsequent weight exchange, Awex constructs a **unified weight convert layer** that performs the following converts:

- **Weight splitting**: Splits merged weights (such as FFN's gate/up) into independent weights, supporting cross-TP Resharding;
- **Weight name unification**: Converts all internal weights from all engines to the same namespace, establishing weight mapping relationships between training and inference engines;
- **Attention weight ReGroup**: On the training engine side, regroups and aligns QKV weights along the inference engine's TP/DPAttention parallelism strategy, avoiding shard explosion from fine-grained splitting;
- **Quantization, precision, and format conversion**: Automatically converts weights on the training side **according to the precision and format of the inference side**, and this low-precision conversion can also reduce the amount of transmitted data.

The entire weight convert adaptation layer is implemented as a **pluggable structure** that can be fully customized at the engine layer, model weight convert, and sharding layer to meet the customization needs of complex scenarios.

### (2) Global Weight Metadata Management

Each training and inference process needs to **be aware of the weight metadata of all training and inference processes globally** for constructing subsequent weight transfer plan. Awex also performs **consistency validation of weight metadata between training and inference** at this step. The main workflow is as follows:

- Each process in the training engine performs weight convert and obtains metadata for the converted shards
- Through all_gather_object, each rank obtains global training shard metadata
- Rank0 on the training side serializes global metadata and reports it to Meta Server
- Inference instance 0 on the inference side performs similar work; other inference instances have identical metadata and don't need additional computation
- All training and inference processes obtain global metadata from MetaServer
- Training and inference engines each perform shard-level metadata consistency and compatibility validation

### (3) P2P Weight Transmission Execution Plan

After obtaining global weight metadata, Awex constructs a **deterministic point-to-point transmission plan** within each training and inference process.

**Core Strategy** (NCCL mode):

- For each replica of the same tensor shard, assign training shards to inference shards through Round Robin to ensure uniform pulling;
- For overlapping shard intervals, if perfectly aligned, directly map; otherwise, use two sends to different shards;
- Pre-filter shards related to the current process to avoid constructing a global plan (shards can reach tens of millions for trillion-parameter models);
- Ensure strict order consistency of NCCL send/recv;

RDMA is more flexible than NCCL and uses a separate transmission plan, which we will expand on in subsequent articles.

### (4) NCCL Weight Transmission

Awex supports two transmission modes: NCCL (NVIDIA Collective Communications Library) and RDMA (Remote Direct Memory Access). NCCL mode is more user-friendly, while RDMA mode is more flexible with higher performance.

NCCL transmission mode primarily uses NCCL's send/recv interface for weight transmission. There are some implementation differences in Awex for separated and co-located modes, which we will detail here.

**NCCL Separated Weight Transmission**

In separated transmission mode, Awex first constructs a joint training-inference NCCL Process Group for subsequent weight transmission from training to inference. Next, based on the NCCL send and transfer plan created during initialization, **P2P peer-to-peer ordered weight sending and receiving** is performed on each process of the training and inference engines. The overall workflow is as follows:

<div align="center">
  <img width="85%" alt="nccl separate" src="images/nccl_separate.png"><br>
</div>

**CUDA IPC Co-located Zero-Copy Weight Mapping**

Since the rank count of a single NCCL communication group can only equal the number of GPU cards, and in the training-inference co-located case, the rank count is twice the number of GPU cards, a joint training-inference communication group cannot be directly established.

In this case, Awex uses **CUDA IPC to zero-copy map the training process's GPU memory to the inference process**, establishes a global communication group for all inference processes, then uses this communication group for NCCL send/recv to complete weight exchange from training to inference engines:

<div align="center">
  <img width="85%" alt="nccl colocate" src="images/nccl_colocate.png"><br>
</div>

In implementation, we have also made some **performance optimizations**:

- **Problem**: Each CUDA IPC Handle's Open/Close has significant overhead; MOE and other models may have thousands to tens of thousands of weight tensors per card requiring IPC serialization;
- **Solution**: Before IPC serialization, merge tensors by shape and dtype, reducing the count to dozens, greatly reducing CUDA IPC overhead;

(**Note**: CUDA IPC does not support CUDA virtual memory. Future plans include allocating additional physical memory space for weight merging and transmission when enabling virtual GPU memory in the training engine)

### (5) RDMA Weight Transmission

Although NCCL transmission mode can already significantly improve weight exchange performance, NCCL mode has two main limitations:

1. **NCCL versions on training and inference sides need to remain compatible**, otherwise NCCL transmission may hang, preventing independent updates and iterations of training and inference engines;
2. **NCCL's static topology is not friendly to communication domain scaling**, as continued RL training causes inference outputs to gradually grow and workload to increase, requiring scaling of inference instances. NCCL needs to destroy the entire communication group and rebuild;

Considering these two reasons, we also developed an RDMA-based transmission implementation, which can be switched with a single configuration parameter.

<div align="center">
  <img width="85%" alt="RDMA transport" src="images/rmda_transport.png"><br>
</div>

**RDMA Mode Advantages**:

- Removes NCCL version binding, supports independent iteration of training and inference engines
- More flexible transmission plan optimization space
- Supports dynamic scaling of inference instances
- Further performance improvement (1T model from 20 seconds to 6 seconds)

RDMA mode implementation will be open-sourced soon. Stay tuned.
