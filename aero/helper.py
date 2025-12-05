
import os
import copy
import torch
import random
import numpy as np
from torch import nn
from typing import Any, List, Dict, Tuple
import pytest # Import pytest

from mmdet.models.roi_heads import SingleRoIExtractor

SEED_FEAT = 1234
SEED_ROI = 5678

def seed_everything(seed=1029):
    """Seeds all relevant random number generators for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.enabled = False # This can sometimes slow things down, only disable if strictly necessary


def get_model_outputs(model: nn.Module, func_name: str,
                      model_inputs: Dict[str, Any]) -> Any:
    """To get outputs of pytorch model."""
    assert hasattr(model, func_name), f'Got unexpected func name: {func_name}'
    func = getattr(model, func_name)
    model_outputs = func(**copy.deepcopy(model_inputs))
    return model_outputs


def get_single_roi_extractor():
    """Initializes and returns a SingleRoIExtractor model."""
    roi_layer = dict(type='RoIAlign', output_size=7, sampling_ratio=2)
    out_channels = 1
    featmap_strides = [4, 8, 16, 32]
    model = SingleRoIExtractor(roi_layer, out_channels, featmap_strides).eval()
    return model