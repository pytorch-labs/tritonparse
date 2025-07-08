#!/bin/bash

# Setup script for tritonparse CI environment
# This script sets up the conda environment, installs dependencies, and configures CUDA

set -e

# Default values
CONDA_ENV=${CONDA_ENV:-"tritonparse"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
CUDA_VERSION=${CUDA_VERSION:-"12.8"}

echo "Setting up tritonparse environment..."
echo "CONDA_ENV: $CONDA_ENV"
echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "CUDA_VERSION: $CUDA_VERSION"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update

# Set up LLVM 17 APT source with modern GPG key handling
echo "Setting up LLVM 17 APT source with modern GPG key handling..."

# Download and install GPG key to /usr/share/keyrings
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | \
  gpg --dearmor | sudo tee /usr/share/keyrings/llvm-archive-keyring.gpg > /dev/null

# Make sure key file is readable by _apt
sudo chmod a+r /usr/share/keyrings/llvm-archive-keyring.gpg

# Write APT source list, explicitly binding keyring file
echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" | \
  sudo tee /etc/apt/sources.list.d/llvm-toolchain-jammy-17.list

# Update package lists
sudo apt-get update

# Install clang and clangd first
echo "Installing clang and clangd..."
sudo apt-get install -y clang-17 clangd-17

# Set up clang alternatives
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-17 100

# Install CUDA and development libraries
echo "Installing CUDA and development libraries..."
sudo apt-get install -y cuda-toolkit-12.8 libstdc++6 libstdc++-12-dev libffi-dev libncurses-dev zlib1g-dev libxml2-dev git build-essential

# Verify clang installation
echo "Verifying clang installation..."
clang --version
clangd --version


# Install Miniconda if not already installed
if [ ! -d "/opt/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh
    chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -u -p /opt/miniconda3
fi

# Add conda to PATH
export PATH="/opt/miniconda3/bin:$PATH"
export CONDA_HOME="/opt/miniconda3"

# Initialize conda
conda init bash || true

# Create conda environment
echo "Creating conda environment: $CONDA_ENV"
conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y || true

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Check NVIDIA GPU information
echo "Checking NVIDIA GPU information..."
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi
else
    echo "nvidia-smi not found"
fi

# Detect CUDA version
echo "Detecting CUDA version..."
if [ -d "/usr/local/cuda" ]; then
    DETECTED_CUDA=$(ls -la /usr/local/cuda | grep -o 'cuda-[0-9.]*' | head -1 | sed 's/cuda-//')
    if [ -n "$DETECTED_CUDA" ]; then
        CUDA_VERSION="$DETECTED_CUDA"
        echo "Found CUDA version: $CUDA_VERSION"
    fi
    export CUDA_HOME="/usr/local/cuda"
else
    echo "CUDA not found in /usr/local/cuda"
fi

export CUDA_VERSION="$CUDA_VERSION"
echo "Using CUDA version: $CUDA_VERSION"

# Install cuDNN
echo "Installing cuDNN..."
bash .ci/install-cudnn.sh

# Install PyTorch nightly
echo "Installing PyTorch nightly..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
fi

echo "Setup completed successfully!"
