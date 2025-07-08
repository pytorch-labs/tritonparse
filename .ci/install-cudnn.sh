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

cd /tmp

# Download cuDNN archive
CUDNN_ARCHIVE="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz"
echo "Looking for cuDNN archive: $CUDNN_ARCHIVE"

if [ -f "$CUDNN_ARCHIVE" ]; then
    echo "Found cuDNN archive, installing from file..."

    # Extract cuDNN
    echo "Extracting cuDNN archive..."
    tar -xJf "$CUDNN_ARCHIVE"

    # Install cuDNN to CUDA directory
    echo "Installing cuDNN to $CUDA_HOME..."
    sudo cp cuda/include/cudnn*.h "$CUDA_HOME/include/"
    sudo cp cuda/lib64/libcudnn* "$CUDA_HOME/lib64/"
    sudo chmod a+r "$CUDA_HOME/include/cudnn*.h" "$CUDA_HOME/lib64/libcudnn*"

    echo "cuDNN installed successfully from archive"
else
    echo "cuDNN archive not found, using conda installation as fallback..."

    # Ensure conda environment is activated
    if [ -n "$CONDA_ENV" ]; then
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV"
    fi

    # Install cuDNN via conda
    conda install -c conda-forge cudnn="$CUDNN_VERSION" -y

    echo "cuDNN installed successfully via conda"
fi

# Set cuDNN environment variables
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Verify cuDNN installation
echo "Verifying cuDNN installation..."

# Check for cuDNN headers
if [ -f "$CUDA_HOME/include/cudnn.h" ]; then
    echo "✓ cuDNN headers found in $CUDA_HOME/include/"
    ls -la "$CUDA_HOME/include/cudnn*"
else
    echo "⚠ cuDNN headers not found in $CUDA_HOME/include/"
fi

# Check for cuDNN libraries
if [ -f "$CUDA_HOME/lib64/libcudnn.so" ]; then
    echo "✓ cuDNN libraries found in $CUDA_HOME/lib64/"
    ls -la "$CUDA_HOME/lib64/libcudnn*"
else
    echo "⚠ cuDNN libraries not found in $CUDA_HOME/lib64/"
fi

echo "cuDNN installation completed!"
