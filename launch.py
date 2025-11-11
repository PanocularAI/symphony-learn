import os
import subprocess
import sys
from typing import List

C10D_PORT = int(str(os.getenv("C10D_PORT", "29400")))


def main(script: List[str]) -> int:

    num_nodes = ...

    if num_nodes < 2:
        print("Fault-tolerant training only supported on multi-node configurations")
        return 0

    rank = ...
    procs_per_node = ...

    master_addr = ...

    torchrun_cmd = [
        "python", "-m", "torch.distributed.run",
        "--nnodes", str(num_nodes),
        "--nproc_per_node", str(procs_per_node),
        "--node_rank", str(rank),
        "--max_restarts", "0",
        "--master_addr", master_addr,
        "--master_port", str(C10D_PORT),
        "--rdzv_id", "panocular_ft",
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", f"{master_addr}:{C10D_PORT}",
    ]

    cmd_extension = [
        f"--fault_tolerance.replica_id={rank}",
        f"--fault_tolerance.group_size={num_nodes}",
        f"--parallelism.data_parallel_replicate_degree={num_nodes}"
    ]

    torchrun_cmd.extend(script)
    torchrun_cmd.extend(cmd_extension)

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCHFT_LIGHTHOUSE"] = f"http://{master_addr}:29510"

    if rank == 0:
        env["RUST_BACKTRACE"] = "1"
        torchft_cmd = [
            "torchft_lighthouse",
            "--min_replicas", "1",
            "--quorum_tick_ms", "100",
            "--join_timeout_ms", "10000"
        ]
        torchft = subprocess.Popen(torchft_cmd, env=env)

    torchrun = subprocess.Popen(torchrun_cmd, env=env)
    exit_code = torchrun.wait()

    if rank == 0:
        torchft.terminate()

    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
