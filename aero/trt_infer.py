import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import random
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")



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



# Load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize the engine! Plugin might be missing.")
        return engine

# Load your TensorRT engine
trt_engine_path = "/workspace/mmdeploy/aero-tensorrt/model.trt.engine"
engine = load_engine(trt_engine_path)
context = engine.create_execution_context()

# Set seeds and generate inputs (mimicking your ONNX example)
seed_everything(SEED_FEAT)
feats = [
    torch.rand((1, 1, 200, 336)),
    torch.rand((1, 1, 100, 168)),
    torch.rand((1, 1, 50, 84)),
    torch.rand((1, 1, 25, 42)),
]

seed_everything(SEED_ROI)
rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

# Convert all inputs to numpy float32
feats_np = [f.numpy().astype(np.float32) for f in feats]
rois_np = rois.numpy().astype(np.float32)

# Flatten all inputs for TensorRT bindings
flat_inputs = [f.ravel() for f in feats_np] + [rois_np.ravel()]

# Allocate buffers
bindings = []
inputs_buffers = []
outputs_buffers = []
stream = cuda.Stream()

for i in range(engine.num_bindings):
    dtype = trt.nptype(engine.get_binding_dtype(i))
    binding_shape = context.get_binding_shape(i)
    size = trt.volume(binding_shape)
    device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
    bindings.append(int(device_mem))

    if engine.binding_is_input(i):
        inputs_buffers.append({'host': flat_inputs.pop(0), 'device': device_mem})
    else:
        outputs_buffers.append({'host': np.empty(size, dtype=dtype), 'device': device_mem})

# Copy inputs to GPU
for inp in inputs_buffers:
    cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

# Run inference
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# Copy outputs back to CPU
for out in outputs_buffers:
    cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

stream.synchronize()

# Reshape outputs to their original shapes
tensorrt_outputs = []
for i, out in enumerate(outputs_buffers):
    shape = tuple(context.get_binding_shape(engine.num_bindings - len(outputs_buffers) + i))
    tensorrt_outputs.append(out['host'].reshape(shape))

# Print shapes and first few values for verification
print("TensorRT input shapes:")
for i, inp in enumerate(inputs_buffers):
    print(f"Input {i}: {inp['host'].shape}")

print("\nTensorRT output shapes:")
for i, out in enumerate(tensorrt_outputs):
    print(f"Output {i}: {out.shape}, sample values: {out.flatten()}")
