#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ================= CONFIGURATION =================
TRT_MAJOR="10.9"
TRT_VERSION="10.9.0.34"
# TensorRT 10.9 is built for CUDA 12.8.
# (If you strictly need CUDA 11.8, you may need to check the archive, 
# but 12.8 is the standard for TRT 10.9).
CUDA_VERSION="12.8"
ARCH="x86_64"
OS="Linux"

## wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

# The specific filename NVIDIA provides for TRT 10.9
TRT_FILENAME="TensorRT-${TRT_VERSION}.${OS}.${ARCH}-gnu.cuda-${CUDA_VERSION}.tar.gz"
TARGET_DIR="/workspace/tensorrt"
# =================================================

echo "========================================================"
echo "Starting TensorRT ${TRT_VERSION} Installation"
echo "Target Directory: ${TARGET_DIR}"
echo "CUDA Version:     ${CUDA_VERSION}"
echo "========================================================"

# 1. Check for Permissions
if [ ! -w "/workspace" ]; then
    echo "Error: You do not have write permissions for /workspace."
    echo "Please run this script with sudo or ensure you own the directory."
    exit 1
fi

# 2. Check if the Tarball exists
if [ ! -f "${TRT_FILENAME}" ]; then
    echo ""
    echo "❌ ERROR: TensorRT Tarball not found: ${TRT_FILENAME}"
    echo "--------------------------------------------------------"
    echo "Due to NVIDIA Licensing, this script cannot download the file automatically."
    echo "You must download it manually:"
    echo "1. Visit: https://developer.nvidia.com/tensorrt/download"
    echo "2. Log in."
    echo "3. Select 'TensorRT 10.x'."
    echo "4. Locate: TensorRT ${TRT_VERSION} for Linux ${ARCH} and CUDA ${CUDA_VERSION} TAR Package"
    echo "5. Save the file to: $(pwd)/${TRT_FILENAME}"
    echo "--------------------------------------------------------"
    exit 1
fi

# 3. Clean previous installation
if [ -d "${TARGET_DIR}" ]; then
    echo "Warning: ${TARGET_DIR} already exists. Removing old files to ensure clean install..."
    rm -rf "${TARGET_DIR}"
fi
mkdir -p "${TARGET_DIR}"

# 4. Extract and Install
echo "Extracting ${TRT_FILENAME}..."
# Extract to a temporary location first to avoid nesting issues
tar -xf "${TRT_FILENAME}"

# The tarball extracts into a folder named 'TensorRT-10.9.0.34'
EXTRACTED_FOLDER="TensorRT-${TRT_VERSION}"

if [ -d "${EXTRACTED_FOLDER}" ]; then
    echo "Moving files to ${TARGET_DIR}..."
    # Move contents directly to target so 'include' is at the top level
    mv "${EXTRACTED_FOLDER}"/* "${TARGET_DIR}/"
    rm -rf "${EXTRACTED_FOLDER}" # Remove the empty shell folder
else
    echo "Error: Extraction failed. Expected folder ${EXTRACTED_FOLDER} not created."
    exit 1
fi

# 5. Install Python bindings (Optional, but recommended)
echo "Installing Python bindings..."
PIP_CMD="pip"
if command -v pip3 &> /dev/null; then PIP_CMD="pip3"; fi

# Navigate to python dir to install wheels
pushd "${TARGET_DIR}/python" > /dev/null
# Install the tensorrt wheel matching python version
# Note: TensorRT 10.x includes 'lean' and 'dispatch' wheels. 
# We install the main 'tensorrt' wheel which covers standard development needs.
${PIP_CMD} install tensorrt-*-cp$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")-*.whl || echo "Warning: Python wheel install failed (version mismatch?), skipping."
popd > /dev/null

# 6. Set Environment Variables
echo "Configuring Environment Variables..."

# Export for current session
export TENSORRT_DIR="${TARGET_DIR}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TARGET_DIR}/lib"

# Persist to .bashrc if not already there
SHELL_RC="$HOME/.bashrc"
if ! grep -q "export TENSORRT_DIR=${TARGET_DIR}" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# TensorRT Configuration" >> "$SHELL_RC"
    echo "export TENSORRT_DIR=${TARGET_DIR}" >> "$SHELL_RC"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${TARGET_DIR}/lib" >> "$SHELL_RC"
    echo "Updated $SHELL_RC"
else
    echo "$SHELL_RC already contains TensorRT paths."
fi

# 7. Verification
if [ -f "${TARGET_DIR}/include/NvInfer.h" ]; then
    echo "========================================================"
    echo "✅ SUCCESS: TensorRT installed to ${TARGET_DIR}"
    echo "   NvInfer.h found."
    echo "========================================================"
    echo "To apply changes to your current shell immediately, run:"
    echo "source ~/.bashrc"
    echo "OR"
    echo "export TENSORRT_DIR=${TARGET_DIR}"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${TARGET_DIR}/lib"
else
    echo "❌ ERROR: Installation finished but NvInfer.h is missing."
    exit 1
fi