#!/bin/bash

# Install cuDNN script for tritonparse CI environment
# This script downloads and installs cuDNN to /usr/local/cuda

set -e

# Default values
CUDNN_VERSION=${CUDNN_VERSION:-"9.10.2.21"}
CUDA_VERSION=${CUDA_VERSION:-"12.8"}
CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}

echo "Installing cuDNN..."
echo "CUDNN_VERSION: $CUDNN_VERSION"
echo "CUDA_VERSION: $CUDA_VERSION"
echo "CUDA_HOME: $CUDA_HOME"

# Check if CUDA is installed
if [ ! -d "$CUDA_HOME" ]; then
    echo "ERROR: CUDA not found in $CUDA_HOME"
    echo "Please install CUDA first"
    exit 1
fi

echo "Found CUDA installation at $CUDA_HOME"

# Detect architecture
export TARGETARCH=${TARGETARCH:-$(uname -m)}
if [ "${TARGETARCH}" = 'aarch64' ] || [ "${TARGETARCH}" = 'arm64' ]; then
    ARCH_PATH='sbsa'
else
    ARCH_PATH='x86_64'
fi

echo "Architecture: ${ARCH_PATH}"

cd /tmp

# Download cuDNN archive from NVIDIA
CUDNN_ARCHIVE="cudnn-linux-${ARCH_PATH}-${CUDNN_VERSION}_cuda12-archive.tar.xz"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${ARCH_PATH}/${CUDNN_ARCHIVE}"

echo "Downloading cuDNN from: $CUDNN_URL"
echo "Archive name: $CUDNN_ARCHIVE"

# Download cuDNN
if ! wget -q "$CUDNN_URL" -O "$CUDNN_ARCHIVE"; then
    echo "ERROR: Failed to download cuDNN from $CUDNN_URL"
    exit 1
fi

echo "cuDNN download complete, extracting archive..."

# Extract cuDNN
if ! tar -xJf "$CUDNN_ARCHIVE"; then
    echo "ERROR: Failed to extract cuDNN archive"
    exit 1
fi

# Install cuDNN to CUDA directory
echo "Installing cuDNN to $CUDA_HOME..."
EXTRACTED_DIR="cudnn-linux-${ARCH_PATH}-${CUDNN_VERSION}_cuda12-archive"

if [ -d "$EXTRACTED_DIR" ]; then
    sudo cp -a "$EXTRACTED_DIR/include/"* "$CUDA_HOME/include/"
    sudo cp -a "$EXTRACTED_DIR/lib/"* "$CUDA_HOME/lib64/"

    # Set permissions using find to avoid globbing issues
    sudo find "$CUDA_HOME/include" -name "cudnn*.h" -exec chmod a+r {} \;
    sudo find "$CUDA_HOME/lib64" -name "libcudnn*" -exec chmod a+r {} \;

    echo "cuDNN installed successfully"
else
    echo "ERROR: Extracted directory not found: $EXTRACTED_DIR"
    exit 1
fi

# Clean up downloaded files
rm -f "$CUDNN_ARCHIVE"
rm -rf "$EXTRACTED_DIR"

# Set cuDNN environment variables
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Verify cuDNN installation
echo "Verifying cuDNN installation..."

# Check for cuDNN headers
if [ -f "$CUDA_HOME/include/cudnn.h" ]; then
    echo "✓ cuDNN headers found in $CUDA_HOME/include/"
    ls -la "$CUDA_HOME/include/cudnn"*
else
    echo "⚠ cuDNN headers not found in $CUDA_HOME/include/"
fi

# Check for cuDNN libraries
if [ -f "$CUDA_HOME/lib64/libcudnn.so" ]; then
    echo "✓ cuDNN libraries found in $CUDA_HOME/lib64/"
    ls -la "$CUDA_HOME/lib64/libcudnn"*
else
    echo "⚠ cuDNN libraries not found in $CUDA_HOME/lib64/"
fi

echo "cuDNN installation completed!"
