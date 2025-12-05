#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# =============================================================================
# Configuration
# =============================================================================
CUDA=${CUDA:-"12.8"} 
# Python 3.8 is too old for modern CUDA 12 wheels. Upgraded to 3.10.
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}

# PyTorch 2.6.0 / Vision 0.21.0 are compatible with CUDA 12.x
# We will use the cu126 wheels (binary compatible with 12.8 drivers)
TORCH_VERSION=${TORCH_VERSION:-"2.6.0"} 
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.21.0"}

# Updated ONNXRuntime to a version supporting CUDA 12
ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION:-"1.19.2"}

# PPLCV and MM versions
PPLCV_VERSION=${PPLCV_VERSION:-"0.7.0"}
MMCV_VERSION=${MMCV_VERSION:-">=2.1.0"}
MMENGINE_VERSION=${MMENGINE_VERSION:-">=0.10.0"}

USE_SRC_INSIDE=${USE_SRC_INSIDE:-false}
VERSION=${VERSION:-""}

ENV_NAME="mmdeploy"
export FORCE_CUDA="1"
export DEBIAN_FRONTEND=noninteractive

# Define TensorRT Directory (Matches your installation script)
export TENSORRT_DIR=${TENSORRT_DIR:-"/workspace/tensorrt"}

# =============================================================================
# 0. Disk Space Check & Setup
# =============================================================================
WORKSPACE_DIR="/workspace"

echo "Checking disk space..."
df -h "$WORKSPACE_DIR"

if [ ! -d "$WORKSPACE_DIR" ]; then
    mkdir -p "$WORKSPACE_DIR"
fi

# =============================================================================
# 1. System Dependencies
# =============================================================================
if [ "$USE_SRC_INSIDE" = true ] ; then
    sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
fi

echo "Installing system dependencies..."
# Note: For CUDA 12, we typically use libcudnn9-cuda-12. 
# If your repo is older, this might fail, but it's correct for modern CUDA 12 images.
apt-get update && \
apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    git wget libssl-dev libopencv-dev libspdlog-dev \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    --no-install-recommends || echo "Warning: apt install of cudnn9 failed. Ensuring manual CUDNN is available later."

rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. Install Miniforge
# =============================================================================
if [ ! -d "/opt/conda" ]; then
    echo "Installing Miniforge..."
    curl -fsSL -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    chmod +x ~/miniforge.sh
    bash ~/miniforge.sh -b -p /opt/conda
    rm ~/miniforge.sh
fi

MAMBA_BIN="/opt/conda/bin/mamba"

# =============================================================================
# 3. Create Clean Environment
# =============================================================================
echo "Creating '${ENV_NAME}' environment with Python ${PYTHON_VERSION}..."

# We install basics with Mamba, but SKIP PyTorch here to save space
$MAMBA_BIN create -n ${ENV_NAME} -y \
    python=${PYTHON_VERSION} \
    conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja cmake

# Clean Mamba cache to free up space immediately
$MAMBA_BIN clean --all -y

export PATH=/opt/conda/envs/${ENV_NAME}/bin:/opt/conda/bin:$PATH

# =============================================================================
# 4. Install PyTorch (Via Pip - Saves Space) & Libraries
# =============================================================================
if [ "$USE_SRC_INSIDE" = true ] ; then
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo "Installing PyTorch via Pip (Space Optimized)..."
# Installing via Pip avoids the "Write failed" error common with large Conda packages
# Using cu126 wheels which work on CUDA 12.8 drivers
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    --index-url https://download.pytorch.org/whl/cu126

pip install --no-cache-dir openmim

echo "Installing MMCV, ONNXRuntime, and MMEngine..."
mim install --no-cache-dir "mmcv${MMCV_VERSION}" onnxruntime-gpu==${ONNXRUNTIME_VERSION} "mmengine${MMENGINE_VERSION}"

# =============================================================================
# 5. ONNXRuntime C++ & TensorRT Setup
# =============================================================================
cd "$WORKSPACE_DIR"
if [ ! -d "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" ]; then
    echo "Downloading ONNXRuntime C++ lib v${ONNXRUNTIME_VERSION}..."
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

    echo "Extracting ONNXRuntime..."
    tar --no-same-owner -zxf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
    rm onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
fi

# Copy TRT bindings
# Note: In TensorRT 10, the python bindings might be installed differently depending on your previous script.
# We attempt to copy them into the conda env.
TRT_SOURCE_SYS="${TENSORRT_DIR}/python" # Often custom installs put wheels here

echo "Checking for TensorRT Python bindings..."
# Try to install the wheel directly if it exists in the TRT directory
if [ -d "${TENSORRT_DIR}/python" ]; then
     echo "Installing TensorRT wheel from ${TENSORRT_DIR}/python..."
     pip install ${TENSORRT_DIR}/python/*cp3${PYTHON_VERSION/3./}*.whl || echo "Warning: Wheel install failed or already installed."
else
     echo "Warning: ${TENSORRT_DIR}/python not found. Assuming TensorRT is already in python path or system."
fi

# =============================================================================
# 6. Build MMDeploy
# =============================================================================
export ONNXRUNTIME_DIR="$WORKSPACE_DIR/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"

# Auto-detect CUDNN
# For CUDA 12, headers are often in /usr/include or /usr/local/cuda/include
if [ -z "$CUDNN_DIR" ]; then
    if [ -f "/usr/local/cuda/include/cudnn.h" ] || [ -f "/usr/local/cuda/include/cudnn_version.h" ]; then
        export CUDNN_DIR="/usr/local/cuda"
    elif [ -f "/usr/include/cudnn.h" ] || [ -f "/usr/include/cudnn_version.h" ]; then
        # Create a link wrapper because CMake expects a specific structure
        mkdir -p /tmp/cudnn_link/include
        mkdir -p /tmp/cudnn_link/lib
        ln -sf /usr/include/cudnn*.h /tmp/cudnn_link/include/
        ln -sf /usr/lib/x86_64-linux-gnu/libcudnn* /tmp/cudnn_link/lib/
        export CUDNN_DIR="/tmp/cudnn_link"
    fi
fi

cd "$WORKSPACE_DIR"
if [ ! -d "mmdeploy" ]; then 
    echo "Cloning MMDeploy (Main branch for TRT 10 support)..."
    git clone -b main https://github.com/dmitryshendryk/mmdeploy
fi

cd mmdeploy
if [ -n "${VERSION}" ] ; then
    git checkout tags/v${VERSION} -b tag_v${VERSION} || echo "Tag already checked out"
fi
git submodule update --init --recursive

mkdir -p build && cd build
rm -rf CMakeCache.txt

echo "Running CMake for MMDeploy..."
# Added -DCMAKE_PREFIX_PATH to ensure CMake finds the Pip-installed Torch
cmake -DMMDEPLOY_TARGET_BACKENDS="trt" \
      -DMMDEPLOY_BUILD_ONNXRUNTIME_OPS=OFF \
      -DCUDNN_DIR="${CUDNN_DIR}" \
      -DTENSORRT_DIR="${TENSORRT_DIR}" \
      -DCMAKE_PREFIX_PATH="/opt/conda/envs/${ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/torch/share/cmake" \
      ..

make -j4
cd ..
mim install -e .

# =============================================================================
# 7. Build PPL.CV
# =============================================================================
cd "$WORKSPACE_DIR"
if [ ! -d "ppl.cv" ]; then git clone https://github.com/openppl-public/ppl.cv.git; fi

cd ppl.cv
git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} || echo "Branch exists"

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

./build.sh cuda

# =============================================================================
# 8. Build MMDeploy SDK
# =============================================================================
export BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# Ensure CUDA 12 compat libs are found if needed
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

cd "$WORKSPACE_DIR/mmdeploy"
rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc
mkdir -p build && cd build

echo "Building MMDeploy SDK..."
cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpplcv_DIR="$WORKSPACE_DIR/ppl.cv/cuda-build/install/lib/cmake/ppl" \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DCUDNN_DIR="${CUDNN_DIR}" \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="trt" \
    -DMMDEPLOY_BUILD_ONNXRUNTIME_OPS=OFF \
    -DMMDEPLOY_CODEBASES=all \
    -DCMAKE_PREFIX_PATH="/opt/conda/envs/${ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/torch/share/cmake"

make -j4 && make install

export SPDLOG_LEVEL=warn

# Final Environment Setup
export LD_LIBRARY_PATH="$WORKSPACE_DIR/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# Add to bashrc if not present
if ! grep -q "conda activate ${ENV_NAME}" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# MMDeploy Environment" >> ~/.bashrc
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    echo "conda activate ${ENV_NAME}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export TENSORRT_DIR=$TENSORRT_DIR" >> ~/.bashrc
    echo "export CUDNN_DIR=$CUDNN_DIR" >> ~/.bashrc
fi

echo "=========================================================="
echo "Done! The '${ENV_NAME}' environment is created."
echo "CUDA: ${CUDA} | TensorRT: 10.9 | PyTorch: ${TORCH_VERSION}"
echo "=========================================================="