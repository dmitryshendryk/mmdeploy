import onnxruntime
import numpy as np
import torch

from helper import seed_everything, SEED_FEAT, SEED_ROI

# Load the ONNX model
onnx_model_path = "simple_model.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

seed_everything(SEED_FEAT)
feats = [
    torch.rand((1, 1, 200, 336)),
    torch.rand((1, 1, 100, 168)),
    torch.rand((1, 1, 50, 84)),
    torch.rand((1, 1, 25, 42)),
]

seed_everything(SEED_ROI)
rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])
# Prepare the inputs
# Convert torch tensors to numpy arrays
feats_np = [f.numpy().astype(np.float32) for f in feats]
rois_np = rois.numpy().astype(np.float32)

# ONNX Runtime expects inputs as a dictionary with names matching the model's input names
input_names = [inp.name for inp in ort_session.get_inputs()]

# Create input dictionary
# If your model has multiple inputs (like feats list + rois), ONNX usually flattens list inputs.
# We'll assume the exported model expects them as separate inputs like feats_0, feats_1, ..., rois
onnx_inputs = {input_names[i]: feats_np[i] for i in range(len(feats_np))}
onnx_inputs[input_names[-1]] = rois_np  # rois as the last input

# Run inference

outputs = ort_session.run(None, onnx_inputs)


# outputs is a list; for SingleRoIExtractor, it should contain one tensor
print("ONNX input shape: ", onnx_inputs['onnx::ReduceSum_0'].shape)
print("ONNX output shape: ", outputs[0].shape)
print("ONNX output: ", outputs)
