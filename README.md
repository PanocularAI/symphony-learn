<div align="center">

# SymphonyLearn

#### A Pytorch-native platform for hetetrogenous & decentralized training of large-scale AI models
</div>

## Overview
As AI models, especially Large Language Models (LLMs) and Vision-Language Models (VLMs), continue to grow in scale and complexity, the need for heterogeneous and decentralized training strategies is becoming increasingly critical. Training such massive models demands enormous computational resources, which are often inaccessible to most researchers and organizations.

HPC centers around the world host a wide variety of GPUs, ranging across different vendors, architectures, and hardware configurations. However, these variations introduce compatibility and utilization challenges, often preventing AI researchers from seamlessly leveraging multiple HPC systems at once.

This platform demonstrates a practical approach to overcoming these challenges by connecting heterogeneous HPC resources in a decentralized manner using the Diloco algorithm. It enables collaborative, cross-platform AI model training without requiring homogeneous hardware or centralized orchestration. In particular, this platform enables two levels of heterogeneity:

* **Cross-Hardware heterogeneity:**
Train models across multiple hardware platforms—leveraging both PyTorch and DaCe integration. This provides flexibility to exploit various computation resources, regardless of vendor or GPU generation. Supporting efficient training on exotic backends can be supported via extending DaCe.

* **Non-uniform GPU distribution:**
HPC clusters vary widely in their node configurations (commonly with 4 or 8 GPUs per node). Our platform offers native support for varying GPU counts and node structures, allowing seamless scaling across diverse systems.

## Why Use This Platform?
* Overcome hardware heterogeneity in HPC environments
* Enable decentralized collaboration for large-model training
* Achieve efficient cross-center resource sharing
* Lower the computational and financial barriers to AI research

## Getting Started
### Install dependencies
1. Before setting up the platform, you need to install `uv` and `rust`.
```bash
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a uv virtual environment:
```
uv venv dtrain_venv --python 3.12
source dtrain_venv/bin/activate
```

3. Clone the repository with:
```
git clone --recursive git@github.com:PanocularAI/symphony-learn.git
```
Make sure that you all pull the submodules using `--recursive` flag.

4. Install dependencies:
```bash
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
uv pip install -r torchtitan/requirements.txt
uv pip install ./torchtitan
uv pip install ./torchft
```

### Configure your HPC nodes
(Add configuration details relevant to your implementation)

### Launch the decentralized training
To start training a model in a decentralized way, you need to execute the following three commands in different shell sessions:

1. Start the lighthouse engine.
```
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

2. Run the training on the first island:
```
NGPU=8 CONFIG_FILE="./models/llama3/train_configs/debug_model.toml" ./run_train.sh --fault_tolerance.enable --fault_tolerance.replica_id=0 --fault_tolerance.group_size=2
```

3. Run the training on the second island:
```
NGPU=8 CONFIG_FILE="./models/llama3/train_configs/debug_model.toml" ./run_train.sh --fault_tolerance.enable --fault_tolerance.replica_id=1 --fault_tolerance.group_size=2
```

## Acknowledgement
This work builds upon the following open-source frameworks:

* [TorchTitan](https://github.com/meta-pytorch/torchtitan) — a PyTorch-native platform for large-scale generative AI model training (Liang et al., ICLR 2025).
* [TorchFT](https://github.com/meta-pytorch/torchft) — a library providing fault-tolerance primitives for distributed PyTorch training (HSDP, LocalSGD, DiLoCo, Streaming DiLoCo).

We gratefully acknowledge the PyTorch, TorchTitan, and TorchFT teams for their foundational contributions to distributed and fault-tolerant ML training infrastructures.