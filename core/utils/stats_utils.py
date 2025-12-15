import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Convert to M (millions)
    def to_millions(num):
        return num / 1_000_000

    total_params_m = to_millions(total_params)
    trainable_params_m = to_millions(trainable_params)

    # Return total and trainable in M
    return total_params_m, trainable_params_m
