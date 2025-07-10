#!/bin/bash

# Install Triton from source
# This script clones and installs Triton from the main repository

set -e

echo "Installing Triton from source..."

# Set Triton version/commit for cache consistency
TRITON_COMMIT=${TRITON_COMMIT:-"main"}
echo "Target Triton commit/branch: $TRITON_COMMIT"
TRITON_CACHE_DIR="/tmp/triton-cache"
TRITON_SOURCE_DIR="/tmp/triton"

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Create cache directory
mkdir -p "$TRITON_CACHE_DIR"

# Check if Triton is already installed and working
if python -c "import triton; print(f'Triton version: {triton.__version__}')" 2>/dev/null; then
    echo "Triton is already installed, checking commit compatibility..."
    
    # Check if the cached commit matches the target commit
    if [ -f "$TRITON_CACHE_DIR/commit" ]; then
        CACHED_COMMIT=$(cat "$TRITON_CACHE_DIR/commit")
        if [ "$CACHED_COMMIT" = "$TRITON_COMMIT" ] && [ "$TRITON_COMMIT" != "main" ]; then
            echo "Triton is already installed with correct commit ($CACHED_COMMIT), skipping installation"
            exit 0
        elif [ "$TRITON_COMMIT" = "main" ]; then
            echo "Target is 'main' branch (API fallback), will reinstall to get latest"
            echo "Cached commit: $CACHED_COMMIT"
        else
            echo "Triton installed but commit mismatch: cached=$CACHED_COMMIT, target=$TRITON_COMMIT"
            echo "Will reinstall Triton..."
        fi
    else
        echo "Triton installed but no commit info found, will reinstall..."
    fi
else
    echo "Triton not installed or not working, proceeding with installation..."
fi

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

# Clone or update Triton repository
echo "Setting up Triton repository..."
if [ -d "$TRITON_SOURCE_DIR" ]; then
    echo "Using cached Triton source..."
    cd "$TRITON_SOURCE_DIR"
    # Reset to clean state and fetch latest
    git reset --hard HEAD
    git clean -fd
    git fetch origin
else
    echo "Cloning Triton repository..."
    git clone https://github.com/triton-lang/triton.git "$TRITON_SOURCE_DIR"
    cd "$TRITON_SOURCE_DIR"
fi

# Checkout specific commit for reproducibility
git checkout "$TRITON_COMMIT"
ACTUAL_COMMIT=$(git rev-parse HEAD)
echo "Using Triton commit: $ACTUAL_COMMIT"

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

# Save commit info for cache validation
echo "$ACTUAL_COMMIT" > "$TRITON_CACHE_DIR/commit"

echo "Triton installation completed successfully!"
