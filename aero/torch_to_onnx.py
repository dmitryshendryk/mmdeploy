import torch

from helper import get_single_roi_extractor

single_roi_extractor = get_single_roi_extractor()

out_channels = single_roi_extractor.out_channels


feats = [
    torch.rand((1, out_channels, 200, 336)),
    torch.rand((1, out_channels, 100, 168)),
    torch.rand((1, out_channels, 50, 84)),
    torch.rand((1, out_channels, 25, 42)),
]

rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

model_inputs = {'feats': feats, 'rois': rois}


onnx_filename = "simple_model.onnx"
torch.onnx.export(single_roi_extractor,      
                  (feats, rois),             
                  onnx_filename,             
                  export_params=True,        
                  opset_version=11,          
                  do_constant_folding=True) 