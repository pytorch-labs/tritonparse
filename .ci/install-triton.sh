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

# Check if we have cached source with correct commit
if [ -f "$TRITON_CACHE_DIR/commit" ] && [ -d "$TRITON_SOURCE_DIR" ]; then
    CACHED_COMMIT=$(cat "$TRITON_CACHE_DIR/commit")
    if [ "$CACHED_COMMIT" = "$TRITON_COMMIT" ] && [ "$TRITON_COMMIT" != "main" ]; then
        echo "Found cached Triton source with correct commit ($CACHED_COMMIT)"
        echo "Will use cached source and re-install to new conda environment"
        USE_CACHED_SOURCE=true
    elif [ "$TRITON_COMMIT" = "main" ]; then
        echo "Target is 'main' branch (API fallback), will rebuild from scratch"
        echo "Cached commit: $CACHED_COMMIT"
        USE_CACHED_SOURCE=false
    else
        echo "Cached source commit mismatch: cached=$CACHED_COMMIT, target=$TRITON_COMMIT"
        echo "Will rebuild from scratch"
        USE_CACHED_SOURCE=false
    fi
else
    echo "No cached source found or no commit info, will build from scratch"
    USE_CACHED_SOURCE=false
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

# Setup Triton repository based on cache status
if [ "$USE_CACHED_SOURCE" = "true" ]; then
    echo "Using cached Triton source..."
    cd "$TRITON_SOURCE_DIR"
    ACTUAL_COMMIT=$(git rev-parse HEAD)
    echo "Using cached Triton commit: $ACTUAL_COMMIT"
else
    echo "Setting up Triton repository from scratch..."
    if [ -d "$TRITON_SOURCE_DIR" ]; then
        echo "Removing existing source directory..."
        rm -rf "$TRITON_SOURCE_DIR"
    fi

    echo "Cloning Triton repository..."
    git clone https://github.com/triton-lang/triton.git "$TRITON_SOURCE_DIR"
    cd "$TRITON_SOURCE_DIR"

    # Checkout specific commit for reproducibility
    git checkout "$TRITON_COMMIT"
    ACTUAL_COMMIT=$(git rev-parse HEAD)
    echo "Using Triton commit: $ACTUAL_COMMIT"
fi

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
if [ "$USE_CACHED_SOURCE" = "true" ]; then
    echo "Installing cached Triton to new conda environment..."
    echo "This should be fast since build artifacts are cached"
else
    echo "Compiling and installing Triton from scratch..."
    echo "This will take 30-50 minutes for compilation"
fi
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
echo "$ACTUAL_COMMIT" >"$TRITON_CACHE_DIR/commit"

echo "Triton installation completed successfully!"
