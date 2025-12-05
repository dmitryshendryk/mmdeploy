# MMDet-to-ONNX/TensorRT Converter

This project provides a simple pipeline for converting **MMDetection** models into **ONNX** and **TensorRT** formats for optimized inference.

## Features
- Convert MMDet checkpoints (`.pth`) to **ONNX**
- Export ONNX models to **TensorRT** engines (`.engine`)
- Configurable input shapes and dynamic axes
- Supports common MMDetection architectures (e.g., Faster R-CNN, YOLOX, Mask R-CNN)

## Requirements
- Python 3.8+
- MMDetection & MMCV
- PyTorch
- ONNX & ONNX Runtime
- TensorRT (for engine generation)

## Usage

### 1. Convert model to ONNX
```bash
python export_to_onnx.py \
  --config configs/model_config.py \
  --checkpoint checkpoints/model.pth \
  --output model.onnx \
  --input-shape 640 640
