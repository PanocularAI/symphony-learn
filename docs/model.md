# How to Add a New Model

This document explains, step by step, how to integrate a **new model** into the framework. We follow the style introduced by [TorchTitan](https://github.com/pytorch/torchtitan) so that it can be trained with the same scaling, logging, and checkpointing infrastructure.

TorchTitan already has a clear [documentation](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/README.md). Please refer to that for more in-depth information.

TorchTitan’s design goals are:

- Minimal, readable code
- Minimal changes to the model when enabling multi‑dimensional parallelism
- Reusable / swappable infrastructure components configured via TOML

The easiest way to add a model is to **mirror the structure** of the existing Llama 3 integration under `torchtitan/models/llama3/`. This guide will reference those files conceptually; adjust details to your actual codebase.

---

## 1. **Get familiar with the existing model definition**

   Read or skim at least these files:

   - `torchtitan/models/llama3/model/model.py` — Llama 3.1 model definition
   - `torchtitan/models/llama3/model/args.py` — Model arguments
   - `torchtitan/models/llama3/model/state_dict_adapter.py` — Helper functions for converting to/from HF model format.
   - `torchtitan/models/llama3/infra/parallelize.py` — how Data/Tensor/Context Parallel, activation checkpointing, and `torch.compile` are applied
   - `torchtitan/models/llama3/train_configs/*.toml` — TOML files to define training hyperparameters and model config for starting a training.

   Your new model should "plug into" the same infrastructure patterns.

---

## 1. Decide Where Your Model Lives

You may add your model under `./models/<model_name>/`.  
For a new model called `mytransformer`, we recommend:

```text
  models/
    mytransformer/
      __init__.py
      model/
        __init__.py
        config.py
        model.py
      infra/
        __init__.py
        parallelize.py
      train_configs/
        mytransformer_1b.toml
        mytransformer_7b.toml
```


## Registering the new model
Once your model definition is done, you need to register the model in `.models/__init__.py` by adding the name of your model.