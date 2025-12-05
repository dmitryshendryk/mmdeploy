#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# =============================================================================
# Configuration
# =============================================================================
CUDA=${CUDA:-"11.8"} 
PYTHON_VERSION=${PYTHON_VERSION:-"3.8"}
# PyTorch 2.0.1 is recommended for CUDA 11.8 compatibility in this context
TORCH_VERSION=${TORCH_VERSION:-"2.0.1"} 
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.15.2"}
ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION:-"1.11.0"}
PPLCV_VERSION=${PPLCV_VERSION:-"0.7.0"}
MMCV_VERSION=${MMCV_VERSION:-">=2.0.0rc2"}
MMENGINE_VERSION=${MMENGINE_VERSION:-">=0.3.0"}
USE_SRC_INSIDE=${USE_SRC_INSIDE:-false}
VERSION=${VERSION:-""}

ENV_NAME="mmdeploy"
export FORCE_CUDA="1"
export DEBIAN_FRONTEND=noninteractive

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
apt-get update && \
apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    git wget libssl-dev libopencv-dev libspdlog-dev \
    libcudnn8 libcudnn8-dev \
    --no-install-recommends
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
echo "Creating '${ENV_NAME}' environment..."

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
# CUDA 11.8 specific index URL
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir openmim

echo "Installing MMCV, ONNXRuntime, and MMEngine..."
mim install --no-cache-dir "mmcv${MMCV_VERSION}" onnxruntime-gpu==${ONNXRUNTIME_VERSION} "mmengine${MMENGINE_VERSION}"

# =============================================================================
# 5. ONNXRuntime C++ & TensorRT
# =============================================================================
cd "$WORKSPACE_DIR"
if [ ! -d "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" ]; then
    echo "Downloading ONNXRuntime C++ lib..."
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

    echo "Extracting ONNXRuntime..."
    tar --no-same-owner -zxf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
    rm onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
fi

# Copy TRT bindings (Check Pip location first since we used Pip above)
TRT_SOURCE_PIP="/opt/conda/envs/${ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/tensorrt*"
if ! ls $TRT_SOURCE_PIP 1> /dev/null 2>&1; then
    # Fallback to system location
    TRT_SOURCE_SYS="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt*"
    if ls $TRT_SOURCE_SYS 1> /dev/null 2>&1; then
        echo "Copying TensorRT bindings from system..."
        cp -r $TRT_SOURCE_SYS "/opt/conda/envs/${ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/"
    fi
fi

# =============================================================================
# 6. Build MMDeploy
# =============================================================================
export ONNXRUNTIME_DIR="$WORKSPACE_DIR/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"
export TENSORRT_DIR=${TENSORRT_DIR:-"/workspace/tensorrt"}

# Auto-detect CUDNN
if [ -z "$CUDNN_DIR" ]; then
    if [ -f "/usr/local/cuda/include/cudnn.h" ]; then
        export CUDNN_DIR="/usr/local/cuda"
    elif [ -f "/usr/include/cudnn.h" ]; then
        mkdir -p /tmp/cudnn_link/include
        mkdir -p /tmp/cudnn_link/lib
        ln -sf /usr/include/cudnn*.h /tmp/cudnn_link/include/
        ln -sf /usr/lib/x86_64-linux-gnu/libcudnn* /tmp/cudnn_link/lib/
        export CUDNN_DIR="/tmp/cudnn_link"
    fi
fi

cd "$WORKSPACE_DIR"
if [ ! -d "mmdeploy" ]; then git clone -b main https://github.com/open-mmlab/mmdeploy; fi

cd mmdeploy
if [ -n "${VERSION}" ] ; then
    git checkout tags/v${VERSION} -b tag_v${VERSION} || echo "Tag already checked out"
fi
git submodule update --init --recursive

mkdir -p build && cd build
rm -rf CMakeCache.txt

echo "Running CMake..."
# Added -DCMAKE_PREFIX_PATH to ensure CMake finds the Pip-installed Torch
cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
      -DCUDNN_DIR="${CUDNN_DIR}" \
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
./build.sh cuda

# =============================================================================
# 8. Build MMDeploy SDK
# =============================================================================
export BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
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
    -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
    -DMMDEPLOY_CODEBASES=all \
    -DCMAKE_PREFIX_PATH="/opt/conda/envs/${ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/torch/share/cmake"

make -j4 && make install

export SPDLOG_LEVEL=warn

# Final Environment Setup
export LD_LIBRARY_PATH="$WORKSPACE_DIR/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

if ! grep -q "conda activate ${ENV_NAME}" ~/.bashrc; then
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    echo "conda activate ${ENV_NAME}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export TENSORRT_DIR=$TENSORRT_DIR" >> ~/.bashrc
    echo "export CUDNN_DIR=$CUDNN_DIR" >> ~/.bashrc
fi

echo "=========================================================="
echo "Done! The '${ENV_NAME}' environment is created."
echo "=========================================================="