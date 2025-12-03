# Fault tolerant, decentralized, and heterogeneous training

Integrating TorchTitan with TorchFT builds a modular and extensible training pipeline that supports fault-tolerant, decentralized training. TorchFT provides resilience for distributed setups that replicate model weights, such as Distributed Data Parallel (DDP) or Hybrid Sharded Data Parallel (HSDP) configurations. When integrated into TorchTitan, TorchFT allows training jobs to automatically recover and continue even if one or more nodes fail. For detailed implementation and usage instructions, see the [TorchFT](https://github.com/meta-pytorch/torchft) repository.

## How It Works
In decentralized training, multiple replica groups are launched—each representing a separate training instance responsible for maintaining a replica of the model weights. If one replica group fails, others continue training without losing synchronization or weight consistency.

For example, to run HSDP training across two machines with eight GPUs each, where weights are sharded within each node (a 2×8 device mesh), you can set the following configuration:

```bash
--data_parallel_replica_degree=2
--data_parallel_shard_degree=4
```

In a conventional training setup, a node or trainer failure would force a full restart from the latest checkpoint—causing downtime and resource inefficiency.

## TorchFT Advantage
With TorchFT, the system becomes fault-tolerant at the per-step level. You can launch two training instances, each managing 8 GPUs and coordinated via TorchFT. In this setup, if one replica group fails, the other can continue training seamlessly while the failed group recovers in the background.

This design:

- Minimizes recovery time and resource waste
- Prevents full-job restarts after hardware or node failures
- Ensures continuous training progress even under partial system degradation


## Launching an example decentralized training
To launch decentralized training using TorchFT with TorchTitan, you need to launch a lighthouse engine to enable fault-tolerance and heartbeating and execute training replicas on different clusters/islands. Therefore, you need to run the following three commands in different shell sessions:

1. Launch TorchFT lighthouse:
```bash
RUST_BACKTRACE=1 uv run torchft_lighthouse --bind=<ip>:<port> --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

| Parameter         |   Description |
| -------------     |  ------------ |
| --bind            | Lighthouse server bind address and port. e.g. `10.0.0.1:29510`            |
| --min_replicas    | Minimum number of training replicas that can be active at each time.      |
| --quorum_tick_ms  | The interval at which the quorum is checked.             |
| --join_timeout_ms | The timeout for joining the quorum. By slow network connection, you may need to set a higher value.             |

NOTE: You may run lighthouse on one of the worker nodes as well. However, In a real-world scenario, it is best to host the lighthouse engine on a server which you are certain about its stability and reliablity. A cpu-only server suffice.

2. Launch the first TorchTitan instance:

```bash
TORCHFT_LIGHTHOUSE=http://10.0.0.1:29510 NGPU=8 LOCAL_ADDR=10.0.0.2 MASTER_ADDR=10.0.0.2 MASTER_PORT=29500 NNODES=1 ISHOST=yes CONFIG_FILE="./models/llama3/train_configs/llama3_8b.toml" uv run ./run_train.sh --fault_tolerance.enable --fault_tolerance.replica_id=0 --fault_tolerance.group_size=2
```

3. Launch the second TorchTitan instance:

```bash
TORCHFT_LIGHTHOUSE=http://10.0.0.1:29510 NGPU=8 LOCAL_ADDR=10.0.0.3 MASTER_ADDR=10.0.0.3 MASTER_PORT=29500 NNODES=1 ISHOST=yes CONFIG_FILE="./models/llama3/train_configs/llama3_8b.toml" uv run ./run_train.sh --fault_tolerance.enable --fault_tolerance.replica_id=1 --fault_tolerance.group_size=2
```

Note: You can also run all services for debugging purposes on a single node, by just setting the right number of GPUs e.g., `NGPU=2 CUDA_VISIBLE_DEVICES=0,1`. 

### Explanation

#### Environment variables:
* `LOCAL_ADDR` is the ip/hostname of each compute node in an island.
* `MASTER_ADDR` is the ip/hostname of the master node in the island.
* `NNODES` is the number of nodes/island. It is NOT the number of nodes across the whole decentralized training setup.
* `ISHOST` should be set to true on the master node in the island. On remaining nodes in the island should be set to `false`.

#### Arguments:
* `--fault_tolerance.enable` enables TorchFT functionality.
* `--fault_tolerance.group_size=2` tells TorchTitan that there are two replica groups.
* `--fault_tolerance.replica_id=1` tells TorchTitan that the replica ID of this instance is 1.
* Note that the alive replica group with the smallest replica ID will perform checkpointing saving.

Within the training config of each model, such as `models/llama3/train_config/debug_model.toml`, you can see the training configuration for Diloco-based training, that do not require per-step synchronization and the replica groups can synchronize weights every N steps.

```toml
[fault_tolerance]
enable = true
sync_steps = 10
num_fragments = 2
semi_sync_method = "diloco"
process_group = "gloo"
process_group_timeout_ms = 10000
```

By changing `sync_steps` you can define after how many inner steps, an outer optimization will be performed among all replicas in the decentralzied training. Also, if the network is unstable, you may want to increase the `process_group_timeout_ms` to a higher value.

#### Heterogeneous configurations
We support heterogeneous configurations (i.e., different numbers of GPUs per island) by synchronizing via rank 0. Setting the flag `rank0_synchronization_only = true` in the training configuration enables support for heterogeneity. Ensure that the model is not split into fragments (i.e., set `num_fragments = 1`) when using heterogeneous configurations.

```toml
[fault_tolerance]
enable = true
sync_steps = 10
num_fragments = 1
semi_sync_method = "diloco"
process_group = "gloo"
process_group_timeout_ms = 10000
rank0_synchronization_only = true
```

Note: Due to a bug when using process groups with the Gloo backend on AMD GPU tensors, please set `copy_pseudogradients_to_cpu = true`. This workaround copies all pseudogradients to the CPU before the outer step.

```toml
[fault_tolerance]
enable = true
sync_steps = 10
num_fragments = 1
semi_sync_method = "diloco"
process_group = "gloo"
process_group_timeout_ms = 10000
rank0_synchronization_only = false
copy_pseudogradients_to_cpu = true
```
