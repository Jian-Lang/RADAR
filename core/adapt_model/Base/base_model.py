import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from utils.adapt_utils import SAM, softmax_entropy


class BaseAdaptModel(object):
    def __init__(self, src_model: nn.Module, cfg: DictConfig, **kargs):
        super(BaseAdaptModel, self).__init__()
        self.cfg = cfg
        self.src_model = copy.deepcopy(src_model)
        self._base_init(cfg)

    def _base_init_optimizer(self, params, **kargs):
        optimizer_name = kargs.pop("name")
        optimizer = None
        match optimizer_name:
            case "AdamW":
                optimizer = torch.optim.AdamW
            case "Adam":
                optimizer = torch.optim.Adam
            case "SGD":
                optimizer = torch.optim.SGD
            case "SAM":
                optimizer = SAM
                kargs["base_optimizer"] = torch.optim.SGD
            case _:
                raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")
        return optimizer(params, **kargs)

    def _init_src_model(self, model: nn.Module):
        raise NotImplementedError("Subclass must implement this method")

    def _initialize_trainable_parameters(self):
        raise NotImplementedError("Subclass must implement this method")

    def _base_init(self, cfg: DictConfig):
        self.src_model = self._init_src_model(self.src_model)
        self.base_src_model = copy.deepcopy(self.src_model)
        params, names = self._initialize_trainable_parameters()
        # logger debug trainable parameters
        logger.info(f"Trainable parameters: {names}")
        self.optimizer = self._base_init_optimizer(params=params, **cfg.opt)
        self.base_optimizer = copy.deepcopy(self.optimizer)

    def copy_model(self):
        """copy and return the whole model."""
        return copy.deepcopy(self.src_model)

    def reset(self):
        self.optimizer = copy.deepcopy(self.base_optimizer)
        self.src_model = copy.deepcopy(self.base_src_model)
