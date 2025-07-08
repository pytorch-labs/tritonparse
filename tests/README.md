# Tritonparse Tests

This directory contains the test suite for tritonparse, including both automated unit tests and manual test examples.

## Test Structure

### Automated Tests
- `test_tritonparse.py`: Comprehensive automated test suite
- `conftest.py`: Pytest configuration and fixtures
- `__init__.py`: Makes the directory a Python package

### Manual Test Examples
- `test_add.py`: Manual test example for Triton kernel addition (not included in automated tests)

### Sample Data
- `example_output/`: Example output directory containing:
  - `logs/`: Sample log files
  - `parsed_output/`: Sample parsed output files

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[test]"
```

### Automated Tests

#### Running All Tests
```bash
# Using pytest directly
python -m pytest tests/test_tritonparse.py -v

# With print statements visible
python -m pytest tests/test_tritonparse.py -s -v

# With coverage
python -m pytest tests/test_tritonparse.py --cov=tritonparse --cov-report=html
```

#### Running Specific Test Categories
```bash
# CPU-only tests
python -m pytest tests/test_tritonparse.py -m "not cuda"

# CUDA tests only (requires CUDA)
python -m pytest tests/test_tritonparse.py -m cuda

# Specific test function
python -m pytest tests/test_tritonparse.py::test_whole_workflow -s -v
```

### Manual Test Example

The `test_add.py` file serves as a manual test example that demonstrates:

1. **Basic Triton kernel implementation**
2. **Logging setup and usage**
3. **Manual test execution**
4. **Parsing workflow**

#### Running Manual Test
```bash
# Direct execution (not through pytest)
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python tests/test_add.py
```

This will:
- Execute a simple tensor addition kernel
- Generate log files in `./logs`
- Parse the logs and output to `./parsed_output`

## Test Categories

### CPU Tests (No CUDA Required)
- `test_convert()`: Tests data conversion functionality with various data types
- `test_unified_parse()`: Tests parsing functionality with mock data

### CUDA Tests (Require GPU)
- `test_extract_python_source_info()`: Tests Python source code extraction during Triton compilation
- `test_whole_workflow()`: Tests complete workflow from kernel execution to log parsing

## Test Features

### Fixtures
- `triton_hooks_setup`: Manages Triton hooks and compilation settings
  - Saves and restores all triton knobs (compilation and runtime hooks)
  - Ensures clean state between tests
  - Automatically restores settings even if tests fail

### CUDA Device Management
- `cuda_device`: Provides CUDA device for tests
- `cuda_available`: Checks CUDA availability
- Automatic skipping of CUDA tests when CUDA is not available

### Kernel Isolation
Each test function defines its own Triton kernel to avoid compilation cache interference:
- `extract_test_kernel`: Simple multiplication kernel (x * 3.0)
- `test_kernel`: Simple addition kernel (x + 1.0)

## Environment Variables

The following environment variables are used during testing:

- `TORCHINDUCTOR_FX_GRAPH_CACHE=0`: Disable FX graph caching
- `TRITONPARSE_DEBUG=1`: Enable debug logging
- `CUDA_VISIBLE_DEVICES=0`: Use first CUDA device (in CI)

## Test Configuration

### Pytest Configuration (`conftest.py`)
- Custom markers for CUDA tests
- Automatic CUDA availability checking
- Fixtures for device management

### Test Isolation
- Each test function has its own kernel definition
- Fixtures ensure clean state between tests
- Temporary directories are automatically cleaned up

## Adding New Tests

### For Automated Tests
1. Add test functions to `test_tritonparse.py`
2. Use `@pytest.mark.cuda` for CUDA tests
3. Use `triton_hooks_setup` fixture for tests that modify Triton settings
4. Define kernels inside test functions to avoid cache interference

Example:
```python
@pytest.mark.cuda
def test_my_function(cuda_device, triton_hooks_setup):
    @triton.jit
    def my_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # Your kernel logic here
        pass
    
    # Your test code here
    pass
```

### For Manual Tests
1. Create new files following the pattern of `test_add.py`
2. Include direct execution with `if __name__ == "__main__"`
3. Document the test purpose and execution method

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- Multiple Python versions (3.9, 3.10, 3.11)
- CUDA versions (11.8, 12.1)
- Code coverage reporting
- Linting and formatting checks

## Troubleshooting

### Common Issues
1. **CUDA not available**: Tests will be skipped automatically
2. **No log files generated**: Check that Triton compilation is working
3. **Import errors**: Ensure all dependencies are installed

### Debug Mode
```bash
# Run with verbose output and print statements
python -m pytest tests/test_tritonparse.py -s -v --tb=long
```

### Manual Verification
```bash
# Run manual test to verify basic functionality
python tests/test_add.py

### Example Output
The `example_output/` directory demonstrates the expected output structure:

```
example_output/
├── logs/
│   └── dedicated_log_triton_trace_findhao_.ndjson
└── parsed_output/
    ├── dedicated_log_triton_trace_findhao__mapped.ndjson.gz
    ├── log_file_list.json
    └── f0_fc0_a0_cai-.ndjson.gz
```

These files can be used to:
- Verify parsing functionality
- Understand expected output format
- Debug parsing issues
- Test with real log data 
