import os
import copy
import torch
import random
import numpy as np
from torch import nn
from typing import Any, List, Dict, Tuple
import pytest # Import pytest



from helper import (seed_everything, 
                    get_model_outputs, 
                    get_single_roi_extractor,
                    SEED_ROI,
                    SEED_FEAT)


seed_everything(SEED_FEAT)
single_roi_extractor = get_single_roi_extractor()

out_channels = single_roi_extractor.out_channels


feats = [
    torch.rand((1, out_channels, 200, 336)),
    torch.rand((1, out_channels, 100, 168)),
    torch.rand((1, out_channels, 50, 84)),
    torch.rand((1, out_channels, 25, 42)),
]

seed_everything(SEED_ROI)
rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

model_inputs = {'feats': feats, 'rois': rois}
output = get_model_outputs(single_roi_extractor, 'forward', model_inputs)

print(output.shape)
print(output)