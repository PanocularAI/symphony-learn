# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'dace'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'torchtitan'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

import numpy as np
import torch
from torch import nn, optim
from dace.frontend.ml.torch.module import DaceModule
from llama3_patched.model.model import Transformer
from llama3_patched.model.args import TransformerModelArgs


llama3_args = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
}


class ONNXCompatibleRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def replace_rms_norm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.RMSNorm):
            normalized_shape = module.weight.shape[0]
            eps = module.eps
            onnx_rms_norm = ONNXCompatibleRMSNorm(normalized_shape, eps)
            onnx_rms_norm.weight.data = module.weight.data.clone()
            setattr(model, name, onnx_rms_norm)
        else:
            replace_rms_norm(module)


class ONNXCompatibleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        replace_rms_norm(self.model)

    def forward(self, input_ids):
        return self.model(input_ids)


def torch_tensors_close(name, torch_v, dace_v, rtol=1e-4, atol=1e-3):
    if torch_v is None and dace_v is None:
        return
    if torch_v is None or dace_v is None:
        raise AssertionError(f"{name}: one tensor is None")
    assert torch_v.device == dace_v.device, f"{name}: tensors on different devices"
    torch_v = torch_v.detach().cpu().numpy()
    dace_v = dace_v.detach().cpu().numpy()
    np.testing.assert_allclose(dace_v, torch_v, rtol=rtol, atol=atol, err_msg=f'{name} not close')


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def training_step(dace_model, pt_model, train_batch, sdfg_name):
    dace_model.load_state_dict(pt_model.state_dict())
    for dace_value, value in zip(pt_model.state_dict().values(), dace_model.state_dict().values()):
        assert torch.allclose(dace_value, value), "State dict copy verification failed"

    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    x, y = train_batch
    train_criterion = nn.CrossEntropyLoss()

    pt_output = pt_model(x)
    pt_output_flat = pt_output.view(-1, pt_output.size(-1))
    y_flat = y.view(-1)
    pt_loss = train_criterion(pt_output_flat, y_flat)

    dace_output = dace_model(x)
    dace_output_flat = dace_output.view(-1, dace_output.size(-1))
    dace_loss = train_criterion(dace_output_flat, y_flat)

    print(f"PT loss: {pt_loss.item():.6f}, DaCe loss: {dace_loss.item():.6f}")

    diff = abs(pt_loss.item() - dace_loss.item()) / pt_loss.item()
    assert diff < 1e-3, f"Loss mismatch: relative difference {diff:.2e} exceeds tolerance 1e-3"
    print(f"Loss relative diff: {diff:.2e}")

    pt_loss.backward()
    dace_loss.backward()

    print("Validating gradients...")
    grad_count = 0
    for (name, pt_param), (dace_name, dace_param) in zip(pt_model.named_parameters(), dace_model.named_parameters()):
        if pt_param.grad is not None:
            torch_tensors_close(name, pt_param.grad, dace_param.grad)
            grad_count += 1
    print(f"Validated {grad_count} parameter gradients")

    optimizer = optim.SGD(pt_model.parameters(), lr=0.001)
    dace_optimizer = optim.SGD(dace_model.parameters(), lr=0.001)
    optimizer.step()
    dace_optimizer.step()

    print("Validating weights after optimizer step...")
    for (name, pt_param), (dace_name, dace_param) in zip(pt_model.named_parameters(), dace_model.named_parameters()):
        torch_tensors_close(name, pt_param.detach(), dace_param.detach())
    print("Weights match after optimizer step")


def test_llama3():
    BATCH_SIZE = 1
    SEQ_LEN = 32

    model_args = llama3_args["debugmodel"]
    print(f"Config: vocab={model_args.vocab_size}, layers={model_args.n_layers}, dim={model_args.dim}")

    pt_model = Transformer(model_args)
    pt_model.init_weights()
    pt_model = ONNXCompatibleWrapper(pt_model)
    pt_model.train()

    dace_model = Transformer(model_args)
    dace_model.init_weights()
    dace_model = ONNXCompatibleWrapper(dace_model)
    dace_model.train()

    sample_input = torch.randint(0, model_args.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    labels = torch.randint(0, model_args.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    print(f"Input shape: {sample_input.shape}, Labels shape: {labels.shape}")

    try:
        training_step(dace_model, pt_model, (sample_input, labels), "llama3_test")
        print("Training step completed successfully!")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_llama_with_dace(num_steps=10, batch_size=2, seq_len=64, lr=1e-4):
    model_args = llama3_args["debugmodel"]
    print(f"Config: vocab={model_args.vocab_size}, layers={model_args.n_layers}, dim={model_args.dim}")

    model = Transformer(model_args)
    model.init_weights()
    model = ONNXCompatibleWrapper(model)
    model.train()

    total_params, trainable_params = count_model_parameters(model)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    dace_model = DaceModule(model, backward=True, simplify=True, training=True, sdfg_name="llama3_train")

    optimizer = optim.AdamW(dace_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    print(f"\nTraining for {num_steps} steps (batch_size={batch_size}, seq_len={seq_len}, lr={lr})")
    print("-" * 60)

    for step in range(num_steps):
        input_ids = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), dtype=torch.long)
        labels = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), dtype=torch.long)

        optimizer.zero_grad()

        output = dace_model(input_ids)
        output_flat = output.view(-1, output.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(output_flat, labels_flat)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Step {step+1:3d}/{num_steps} | Loss: {loss.item():.4f}")

    print("-" * 60)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Loss change:  {losses[-1] - losses[0]:.4f}")

    return losses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    if args.train:
        train_llama_with_dace(
            num_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr
        )
    else:
        model_args = llama3_args["debugmodel"]
        model = Transformer(model_args)
        total_params, trainable_params = count_model_parameters(model)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        success = test_llama3()
        sys.exit(0 if success else 1)
