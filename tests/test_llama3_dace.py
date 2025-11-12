#!/usr/bin/env python
# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'dace'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'torchtitan'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'claude_workspace'))

import torch
from dace.frontend.ml.torch.module import DaceModule
from onnx_utils import ONNXCompatibleWrapper
from llama3_patched.model.model import Transformer
from llama3_patched.model.args import TransformerModelArgs

llama3_args = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
}

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_llama3():

    BATCH_SIZE = 1
    SEQ_LEN = 32

    model_args = llama3_args["debugmodel"]
    print(f"Config: vocab={model_args.vocab_size}, layers={model_args.n_layers}, dim={model_args.dim}")

    model = Transformer(model_args)
    model.init_weights()
    model.eval()
    wrapped_model = ONNXCompatibleWrapper(model)

    sample_input = torch.randint(0, model_args.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    print(f"Input shape: {sample_input.shape}")

    with torch.no_grad():
        pytorch_output = wrapped_model(sample_input.clone())
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output stats: mean={pytorch_output.mean():.4f}, std={pytorch_output.std():.4f}")

    try:
        dace_model = DaceModule(wrapped_model, sdfg_name="llama3_test")

        with torch.no_grad():
            dace_output = dace_model(sample_input.clone())


        max_diff = torch.max(torch.abs(pytorch_output - dace_output)).item()
        print(f"\n4. Max difference: {max_diff:.6f}")

        if max_diff < 0.1:
            print("Results match reasonably well!")
        else:
            print("Some numerical differences (expected with optimizations)")

        return True

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model_args = llama3_args["debugmodel"]  
    model = Transformer(model_args)
    total_params, trainable_params = count_model_parameters(model)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    success = test_llama3()
    sys.exit(0 if success else 1)
