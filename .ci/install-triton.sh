#!/bin/bash

# Install Triton from source
# This script clones and installs Triton from the main repository

set -e

echo "Installing Triton from source..."

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi


# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Update libstdc++ to match system version
# Otherwise, we get errors like:
# ImportError: /opt/miniconda3/envs/tritonparse/bin/../lib/libstdc++.so.6:
# version `GLIBCXX_3.4.30' not found (required by /tmp/triton/python/triton/_C/libtriton.so)
echo "Updating libstdc++ to match system version..."
conda install -y -c conda-forge libstdcxx-ng=12.3.0
# Check if the update was successful
strings /opt/miniconda3/envs/tritonparse/lib/libstdc++.so.6 | grep GLIBCXX | tail -5

# Uninstall existing pytorch-triton
echo "Uninstalling existing pytorch-triton..."
pip uninstall -y pytorch-triton || true
pip uninstall -y triton || true

# Remove existing triton installation
TRITON_PKG_DIR=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))" 2>/dev/null || echo "")
if [ -n "$TRITON_PKG_DIR" ] && [ -d "$TRITON_PKG_DIR" ]; then
    echo "Removing existing Triton installation: $TRITON_PKG_DIR"
    rm -rf "$TRITON_PKG_DIR"
fi

# Clone Triton repository
echo "Cloning Triton repository..."
cd /tmp
if [ -d "triton" ]; then
    rm -rf triton
fi
git clone https://github.com/triton-lang/triton.git
cd triton

# Install build dependencies
echo "Installing build dependencies..."
pip install ninja cmake wheel pybind11

# Install Triton requirements
echo "Installing Triton requirements..."
pip install -r python/requirements.txt

# Set environment to use clang compiler for faster compilation
echo "Setting up clang compiler for faster compilation..."
export CC=clang
export CXX=clang++
echo "Using CC: $CC"
echo "Using CXX: $CXX"

# Install Triton in editable mode with clang
echo "Installing Triton in editable mode with clang..."
pip install -e .

# Verify Triton installation
echo "Verifying Triton installation..."
python -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    echo "ERROR: Failed to import triton"
    echo "This might be due to libstdc++ version issues"
    echo "Checking system libstdc++ version:"
    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -5
    echo "Checking conda libstdc++ version:"
    strings /opt/miniconda3/envs/tritonparse/lib/libstdc++.so.6 | grep GLIBCXX | tail -5
    exit 1
}
python -c "import triton; print(f'Triton path: {triton.__file__}')"

echo "Triton installation completed successfully!"
