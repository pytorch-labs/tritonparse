"""
Comprehensive tests for tritonparse using unittest.
Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python -m unittest tests.test_tritonparse -v
```
"""

import json
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

    def _create_test_kernel(self):
        """Create a simple test kernel for compilation testing"""
        @triton.jit
        def test_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = x + 1.0  # Simple operation: add 1
            tl.store(y_ptr + offsets, y, mask=mask)
        
        return test_kernel

    def _run_kernel(self, kernel, x):
        """Run a kernel with given input tensor"""
        n_elements = x.numel()
        y = torch.empty_like(x)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        kernel[grid](x, y, n_elements, BLOCK_SIZE)
        return y

    def _setup_test_directories(self):
        """Set up temporary directories for testing"""
        temp_dir = tempfile.mkdtemp()
        temp_dir_logs = os.path.join(temp_dir, "logs")
        temp_dir_parsed = os.path.join(temp_dir, "parsed_output")
        
        os.makedirs(temp_dir_logs, exist_ok=True)
        os.makedirs(temp_dir_parsed, exist_ok=True)
        
        return temp_dir, temp_dir_logs, temp_dir_parsed

    def _generate_test_data(self):
        """Generate test tensor data"""
        torch.manual_seed(0)
        size = (512, 512)  # Smaller size for faster testing
        return torch.randn(size, device=self.cuda_device, dtype=torch.float32)

    def _verify_log_directory(self, log_dir):
        """Verify that log directory exists and contains files"""
        assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."
        
        log_files = os.listdir(log_dir)
        assert len(log_files) > 0, (
            f"No log files found in {log_dir}. "
            "Expected log files to be generated during Triton compilation."
        )
        print(f"Found {len(log_files)} log files in {log_dir}: {log_files}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_whole_workflow(self):
        """Test unified_parse functionality"""
        # Set up test environment
        temp_dir, temp_dir_logs, temp_dir_parsed = self._setup_test_directories()
        print(f"Temporary directory: {temp_dir}")
        
        # Initialize logging
        tritonparse.structured_logging.init(temp_dir_logs, enable_trace_launch=True)
        
        # Generate test data and run kernels
        test_kernel = self._create_test_kernel()
        x = self._generate_test_data()
        
        # Run kernel twice to generate compilation and launch events
        self._run_kernel(test_kernel, x)
        self._run_kernel(test_kernel, x)
        torch.cuda.synchronize()
        
        # Verify log directory
        self._verify_log_directory(temp_dir_logs)

        def parse_log_line(line: str, line_num: int) -> dict | None:
            """Parse a single log line and extract event data"""
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON decode error - {e}")
                return None

        def process_event_data(event_data: dict, line_num: int, event_counts: dict) -> None:
            """Process event data and update counts"""
            try:
                event_type = event_data.get("event_type")
                if event_type is None:
                    return
                
                if event_type in event_counts:
                    event_counts[event_type] += 1
                    print(f"  Line {line_num}: event_type = '{event_type}' (count: {event_counts[event_type]})")
                else:
                    print(f"  Line {line_num}: event_type = '{event_type}' (not tracked)")
            except (KeyError, TypeError) as e:
                print(f"  Line {line_num}: Data structure error - {e}")

        def count_events_in_file(file_path: str, event_counts: dict) -> None:
            """Count events in a single log file"""
            print(f"Checking event types in: {os.path.basename(file_path)}")
            
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    event_data = parse_log_line(line, line_num)
                    if event_data:
                        process_event_data(event_data, line_num, event_counts)

        def check_event_type_counts_in_logs(log_dir: str) -> dict:
            """Count 'launch' and 'compilation' events in all log files"""
            event_counts = {"compilation": 0, "launch": 0}
            
            for log_file in os.listdir(log_dir):
                if log_file.endswith(".ndjson"):
                    log_file_path = os.path.join(log_dir, log_file)
                    count_events_in_file(log_file_path, event_counts)
            
            print(f"Event type counts: {event_counts}")
            return event_counts

        # Verify event counts
        event_counts = check_event_type_counts_in_logs(temp_dir_logs)
        self._verify_event_counts(event_counts)
        
        # Test parsing functionality
        tritonparse.utils.unified_parse(
            source=temp_dir_logs, out=temp_dir_parsed, overwrite=True
        )
        
        # Verify parsing output
        self._verify_parsing_output(temp_dir_parsed)
        
        # Clean up
        shutil.rmtree(temp_dir)

    def _verify_event_counts(self, event_counts):
        """Verify that event counts match expected values"""
        assert event_counts["compilation"] == 1, (
            f"Expected 1 'compilation' event, found {event_counts['compilation']}"
        )
        assert event_counts["launch"] == 2, (
            f"Expected 2 'launch' events, found {event_counts['launch']}"
        )
        print("✓ Verified correct event type counts: 1 compilation, 2 launch")

    def _verify_parsing_output(self, parsed_dir):
        """Verify that parsing output directory contains files"""
        parsed_files = os.listdir(parsed_dir)
        assert len(parsed_files) > 0, "No files found in parsed output directory"


if __name__ == "__main__":
    unittest.main()
