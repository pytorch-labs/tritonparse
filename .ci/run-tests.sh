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

# Build unittest command based on test type
case "$TEST_TYPE" in
"cpu")
    echo "Running CPU tests only..."
    if [ "$COVERAGE" = "true" ]; then
        echo "Running with coverage..."
        if [ "$VERBOSE" = "true" ]; then
            coverage run -m unittest tests.test_tritonparse.TestTritonparseCPU -v
        else
            coverage run -m unittest tests.test_tritonparse.TestTritonparseCPU
        fi
        coverage report
        coverage xml
    else
        if [ "$VERBOSE" = "true" ]; then
            python -m unittest tests.test_tritonparse.TestTritonparseCPU -v
        else
            python -m unittest tests.test_tritonparse.TestTritonparseCPU
        fi
    fi
    ;;
"cuda")
    echo "Running CUDA tests only..."
    export CUDA_VISIBLE_DEVICES=0
    if [ "$COVERAGE" = "true" ]; then
        echo "Running with coverage..."
        if [ "$VERBOSE" = "true" ]; then
            coverage run -m unittest tests.test_tritonparse.TestTritonparseCUDA -v
        else
            coverage run -m unittest tests.test_tritonparse.TestTritonparseCUDA
        fi
        coverage report
        coverage xml
    else
        if [ "$VERBOSE" = "true" ]; then
            python -m unittest tests.test_tritonparse.TestTritonparseCUDA -v
        else
            python -m unittest tests.test_tritonparse.TestTritonparseCUDA
        fi
    fi
    ;;
"all")
    echo "Running all tests..."
    export CUDA_VISIBLE_DEVICES=0
    if [ "$COVERAGE" = "true" ]; then
        echo "Running with coverage..."
        if [ "$VERBOSE" = "true" ]; then
            coverage run -m unittest tests.test_tritonparse -v
        else
            coverage run -m unittest tests.test_tritonparse
        fi
        coverage report
        coverage xml
    else
        if [ "$VERBOSE" = "true" ]; then
            python -m unittest tests.test_tritonparse -v
        else
            python -m unittest tests.test_tritonparse
        fi
    fi
    ;;
*)
    echo "Unknown test type: $TEST_TYPE"
    echo "Available options: cpu, cuda, all"
    exit 1
    ;;
esac

echo "Tests completed successfully!"
