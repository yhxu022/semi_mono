# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmengine.registry import HOOKS


@HOOKS.register_module()
class Unsup_weight_Hook(Hook):
    def __init__(self,
                 momentum: float = 0.001,
                 interval: int = 1,
                 skip_buffer=True) -> None:
        assert 0 < momentum < 1
        self.momentum = momentum
        self.interval = interval
        self.skip_buffers = skip_buffer

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Update teacher's parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.momentum_update(model, self.momentum)

