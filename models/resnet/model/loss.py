from typing import Callable, TypeAlias

import torch

from torchtitan.config.configs import CompileConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.float(), labels
    )


def build_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn