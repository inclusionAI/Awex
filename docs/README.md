
**Awex** is a high-performance RL training-inference **weight synchronization** framework,
designed to enable **second-level parameter updates** from training to inference in RL workflows.
It minimizes iteration latency, ensuring rollout phases consistently use the latest model.


## Architecture

<div align="center">
  <img width="85%" alt="Apache Fory logo" src="images/awex_arch.png"><br>
</div>

**NCCL Separate Mode**:
<div align="center">
  <img width="85%" alt="Apache Fory logo" src="images/nccl_separate.png"><br>
</div>

**NCCL Colocate Mode**:
<div align="center">
  <img width="85%" alt="Apache Fory logo" src="images/nccl_colocate.png"><br>
</div>

**RDMA Mode**

Coming soon