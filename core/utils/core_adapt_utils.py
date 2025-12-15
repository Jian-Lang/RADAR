import hashlib
import importlib
import os
import random
import shutil
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import is_tensor
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from transformers import BatchEncoding, BatchFeature


def load_adapt_model(model_name: str, **kargs):
    try:
        # Attempt to import the module
        module = importlib.import_module(f"adapt_model.{model_name}.{model_name}_model")
    except ImportError:
        raise ImportError(
            "Failed to import the 'adapt_model' module. Please ensure it exists and is in the correct path."
        )

    try:
        # Attempt to get the model class
        model_class = getattr(module, model_name)
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in the 'adapt_model' module.")

    try:
        # Attempt to instantiate the model
        model = model_class(**kargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating model '{model_name}': {str(e)}. Please check the provided arguments."
        )

    # Set the model name
    model.name = model_name

    return model


def copy_adapt_config_file():
    src_dir = "core/adapt_model"
    dest_dir = "core/adapt_config"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # remove all yaml files in dest_dir
    for file in os.listdir(dest_dir):
        if file.endswith(".yaml"):
            os.remove(os.path.join(dest_dir, file))

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".yaml"):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                shutil.copy2(src_file_path, dest_file_path)


def print_param_frozen(model: nn.Module):
    """
    Prints every parameter (weight and bias) in the model,
    along with whether it's frozen.
    """
    for name, param in model.named_parameters():
        status = "Frozen" if not param.requires_grad else "Trainable"
        print(f"{name:40s}: {status}")


def compute_grad_norm(grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.tensor(0.0)
    total_sq = sum(g.pow(2).sum() for g in grads.values())
    total_dim = sum(g.numel() for g in grads.values())
    rms = torch.sqrt(total_sq / total_dim)
    return rms
