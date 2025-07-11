"""
Comprehensive tests for tritonparse using unittest.
Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python -m unittest tests.test_tritonparse -v
```
"""

import os
import shutil
import tempfile
import unittest
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

import tritonparse.structured_logging
import tritonparse.utils

# @manual=//triton:triton
from triton.compiler import ASTSource, IRSource

# @manual=//triton:triton
from triton.knobs import CompileTimes
from tritonparse.structured_logging import convert, extract_python_source_info


class TestTritonparseCPU(unittest.TestCase):
    """CPU-only tests (no CUDA required)"""

    def test_convert(self):
        """Test convert function with various data types"""
        # Test with primitive types
        assert convert(42) == 42
        assert convert("hello") == "hello"
        assert convert(3.14) == 3.14
        assert convert(None) is None
        assert convert(True) is True

        # Test with a dictionary
        test_dict = {"a": 1, "b": "string", "c": 3.14}
        assert convert(test_dict) == test_dict

        # Test with a list
        test_list = [1, "string", 3.14]
        assert convert(test_list) == test_list

        # Test with a dataclass
        @dataclass
        class TestDataClass:
            x: int
            y: str
            z: float

        test_dataclass = TestDataClass(x=42, y="hello", z=3.14)
        expected_dict = {"x": 42, "y": "hello", "z": 3.14}
        assert convert(test_dataclass) == expected_dict

        # Test with nested structures
        @dataclass
        class NestedDataClass:
            name: str
            value: int

        nested_structure = {
            "simple_key": "simple_value",
            "list_key": [1, 2, NestedDataClass(name="test", value=42)],
            "dict_key": {"nested_key": NestedDataClass(name="nested", value=100)},
        }

        expected_nested = {
            "simple_key": "simple_value",
            "list_key": [1, 2, {"name": "test", "value": 42}],
            "dict_key": {"nested_key": {"name": "nested", "value": 100}},
        }

        assert convert(nested_structure) == expected_nested


class TestTritonparseCUDA(unittest.TestCase):
    """CUDA tests (require GPU)"""

    def setUp(self):
        """Set up triton hooks and compilation settings"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.cuda_device = torch.device("cuda:0")

        # Save original settings
        self.prev_listener = triton.knobs.compilation.listener
        self.prev_always_compile = triton.knobs.compilation.always_compile
        self.prev_jit_post_compile_hook = triton.knobs.runtime.jit_post_compile_hook
        self.prev_launch_enter_hook = triton.knobs.runtime.launch_enter_hook

        # Set up new settings
        triton.knobs.compilation.always_compile = True

    def tearDown(self):
        """Restore original triton settings"""
        # Always restore original settings, even if test fails
        triton.knobs.compilation.always_compile = self.prev_always_compile
        triton.knobs.compilation.listener = self.prev_listener
        triton.knobs.runtime.jit_post_compile_hook = self.prev_jit_post_compile_hook
        triton.knobs.runtime.launch_enter_hook = self.prev_launch_enter_hook

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_extract_python_source_info(self):
        """Test extract_python_source_info function"""

        # Define kernel inside the test function
        @triton.jit
        def extract_test_kernel(
            x_ptr,
            y_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = x * 3.0  # Simple operation: multiply by 3
            tl.store(y_ptr + offsets, y, mask=mask)

        trace_data = defaultdict(dict)

        def compile_listener(
            src: Union[ASTSource, IRSource],
            metadata: dict[str, str],
            metadata_group: dict[str, Any],
            times: CompileTimes,
            cache_hit: bool,
        ) -> None:
            nonlocal trace_data
            extract_python_source_info(trace_data, src)

        # Set up compilation listener
        triton.knobs.compilation.listener = compile_listener

        torch.manual_seed(0)
        size = (512, 512)
        a = torch.randn(size, device=self.cuda_device, dtype=torch.float32)

        # Use the kernel defined inside this test function
        n_elements = a.numel()
        c = torch.empty_like(a)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        extract_test_kernel[grid](a, c, n_elements, BLOCK_SIZE)

        torch.cuda.synchronize()
        assert "python_source" in trace_data
        assert "file_path" in trace_data["python_source"]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_whole_workflow(self):
        """Test unified_parse functionality"""

        # Define a simple kernel directly in the test function
        @triton.jit
        def test_kernel(
            x_ptr,
            y_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = x + 1.0  # Simple operation: add 1
            tl.store(y_ptr + offsets, y, mask=mask)

        # Simple function to run the kernel
        def run_test_kernel(x):
            n_elements = x.numel()
            y = torch.empty_like(x)
            BLOCK_SIZE = 256  # Smaller block size for simplicity
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            test_kernel[grid](x, y, n_elements, BLOCK_SIZE)
            return y

        temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory: {temp_dir}")
        temp_dir_logs = os.path.join(temp_dir, "logs")
        os.makedirs(temp_dir_logs, exist_ok=True)
        temp_dir_parsed = os.path.join(temp_dir, "parsed_output")
        os.makedirs(temp_dir_parsed, exist_ok=True)

        tritonparse.structured_logging.init(temp_dir_logs)

        # Generate some triton compilation activity to create log files
        torch.manual_seed(0)
        size = (512, 512)  # Smaller size for faster testing
        x = torch.randn(size, device=self.cuda_device, dtype=torch.float32)
        run_test_kernel(x)  # Run the simple kernel
        torch.cuda.synchronize()

        # Check that temp_dir_logs folder has content
        assert os.path.exists(
            temp_dir_logs
        ), f"Log directory {temp_dir_logs} does not exist."
        log_files = os.listdir(temp_dir_logs)
        assert (
            len(log_files) > 0
        ), f"No log files found in {temp_dir_logs}. Expected log files to be generated during Triton compilation."
        print(f"Found {len(log_files)} log files in {temp_dir_logs}: {log_files}")

        tritonparse.utils.unified_parse(
            source=temp_dir_logs, out=temp_dir_parsed, overwrite=True
        )

        # Clean up temporary directory
        try:
            # Check that parsed output directory has files
            parsed_files = os.listdir(temp_dir_parsed)
            assert len(parsed_files) > 0, "No files found in parsed output directory"
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
