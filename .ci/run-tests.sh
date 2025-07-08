#!/bin/bash

# Run tritonparse tests
# This script runs the test suite with proper environment setup

set -e

# Default values
TEST_TYPE=${TEST_TYPE:-"all"}
VERBOSE=${VERBOSE:-"true"}
COVERAGE=${COVERAGE:-"false"}

echo "Running tritonparse tests..."
echo "TEST_TYPE: $TEST_TYPE"
echo "VERBOSE: $VERBOSE"
echo "COVERAGE: $COVERAGE"

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Set environment variables
# export TORCHINDUCTOR_FX_GRAPH_CACHE=0
# export TRITONPARSE_DEBUG=1

# Build pytest command
PYTEST_CMD="python -m pytest tests/test_tritonparse.py"

# Add verbose flag
if [ "$VERBOSE" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage if requested
if [ "$COVERAGE" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=tritonparse --cov-report=xml"
fi

# Run tests based on type
case "$TEST_TYPE" in
    "cpu")
        echo "Running CPU tests only..."
        $PYTEST_CMD -m "not cuda"
        ;;
    "cuda")
        echo "Running CUDA tests only..."
        export CUDA_VISIBLE_DEVICES=0
        $PYTEST_CMD -m cuda
        ;;
    "all")
        echo "Running all tests..."
        export CUDA_VISIBLE_DEVICES=0
        $PYTEST_CMD
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Available options: cpu, cuda, all"
        exit 1
        ;;
esac

echo "Tests completed successfully!" 