# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from models.resnet.model.loss import build_cross_entropy_loss
from torchtitan.protocols.model_spec import FaultTolerantModelSpec

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


def model_registry(flavor: str) -> FaultTolerantModelSpec:
    return FaultTolerantModelSpec(
        name="ft/resnet",
        flavor=flavor,
        model=resnet_configs[flavor],
        parallelize_fn=parallelize_resnet,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
        fragment_fn=None,
    )
