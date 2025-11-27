# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.ft.diloco import FaultTolerantTrainSpec, fragment_llm
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from torchtitan.experiments.gpt_oss.infra.parallelize import parallelize_gptoss
from torchtitan.experiments.gpt_oss.model.args import GptOssModelArgs
from torchtitan.experiments.gpt_oss.model.model import GptOssModel

from torchtitan.experiments.gpt_oss import gptoss_configs


def get_train_spec() -> TrainSpec:
    return FaultTolerantTrainSpec(
        model_cls=GptOssModel,
        model_args=gptoss_configs,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        fragment_fn=fragment_llm,
    )
