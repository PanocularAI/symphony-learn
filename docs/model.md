# How to Add a New Model

This document explains, step by step, how to integrate a **new model** into the framework. We follow the style introduced by [TorchTitan](https://github.com/pytorch/torchtitan) so that it can be trained with the same scaling, logging, and checkpointing infrastructure.

TorchTitan already has a clear [documentation](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/README.md). Please refer to that for more in-depth information.

TorchTitan’s design goals are:

- Minimal, readable code
- Minimal changes to the model when enabling multi‑dimensional parallelism
- Reusable / swappable infrastructure components configured via Python

The easiest way to add a model is to **mirror the structure** of the existing Llama 3 integration under `torchtitan/models/llama3/`. This guide will reference those files conceptually; adjust details to your actual codebase.

---

## 1. **Get familiar with the existing model definition**

   Read or skim at least these files:

   - `torchtitan/models/llama3/model.py` — Llama 3.1 model definition
   - `torchtitan/models/llama3/state_dict_adapter.py` — helper functions for converting to/from HF checkpoint format
   - `torchtitan/models/llama3/parallelize.py` — how Data/Tensor/Context Parallel, activation checkpointing, and `torch.compile` are applied
   - `torchtitan/models/llama3/__init__.py` — `model_registry(flavor)` returning a `ModelSpec` that bundles the model config, parallelize fn, and loss fn
   - `torchtitan/models/llama3/config_registry.py` — Python functions returning a `Trainer.Config`; each function is a named run preset selected via `--config <function_name>`

   Your new model should "plug into" the same infrastructure patterns.

---

## 2. Decide Where Your Model Lives

You may add your model under `./models/<model_name>/`.  
For a new model called `mytransformer`, we recommend:

```text
  models/
    mytransformer/
      __init__.py
      config_registry.py
      model.py
      parallelize.py
```


## Registering the new model
Once your model definition is done, you need to register the model in `models/__init__.py` by adding the name of your model.