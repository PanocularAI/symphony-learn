# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from models.resnet.datasets.cifar10 import build_cifar_dataloader
from models.resnet.model.loss import build_cross_entropy_loss

from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_resnet
from .model.args import ResNetModelArgs
from .model.model import ResNetModel

__all__ = [
    "parallelize_resnet",
    "ResNetModelArgs",
    "ResNetModel",
    "resnet_configs",
]


resnet_configs = {
    "18": ResNetModelArgs(
        num_classes=10,
        block="ResidualBlock",
        layers=[2, 2, 2, 2]
    ),
    "34": ResNetModelArgs(
        num_classes=10,
        block="ResidualBlock",
        layers=[3, 4, 6, 3]
    ),
    "50": ResNetModelArgs(
        num_classes=10,
        block="BottleNeckBlock",
        layers=[3, 4, 6, 3]
    ),
    "152": ResNetModelArgs(
        num_classes=10,
        block="BottleNeckBlock",
        layers=[3, 8, 36, 3]
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=ResNetModel,
        model_args=resnet_configs,
        parallelize_fn=parallelize_resnet,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_cifar_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=build_cross_entropy_loss,
    )
