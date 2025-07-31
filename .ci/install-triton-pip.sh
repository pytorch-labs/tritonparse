#!/bin/bash

# Install Triton from pip
# This script installs the latest version of Triton from PyPI.

set -e

echo "🚀 Installing Triton from pip..."
START_TIME=$(date +%s)

# Function to show elapsed time
show_elapsed() {
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "⏱️ Elapsed time: ${ELAPSED}s"
}

# Pre-flight checks
echo "🔍 Running pre-flight checks..."

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Uninstall existing triton to avoid conflicts
echo "Uninstalling existing Triton versions..."
pip uninstall -y pytorch-triton triton || true

# Install Triton from pip
echo "Installing the latest Triton from PyPI..."
pip install triton

show_elapsed

# Verify Triton installation
echo "Verifying Triton installation..."
if python -c "import triton; print(f'Triton version: {triton.__version__}')" 2>/dev/null; then
    python -c "import triton; print(f'Triton path: {triton.__file__}')"
    echo "✅ Triton installation verified successfully"
    show_elapsed
    echo "🎉 Triton installation completed successfully!"
else
    echo "❌ ERROR: Failed to import Triton"
    exit 1
fi
