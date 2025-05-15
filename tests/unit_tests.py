from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union

import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource, IRSource
from triton.knobs import CompileTimes

from ..tritonparse.structured_logging import convert, extract_python_source_info


def test_extract_python_source_info():
    trace_data = defaultdict(dict)

    @triton.jit
    def add_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a + b
        tl.store(c_ptr + offsets, c, mask=mask)

    def tensor_add(a, b):
        n_elements = a.numel()
        c = torch.empty_like(a)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
        return c

    def compile_listener(
        src: Union[ASTSource, IRSource],
        metadata: dict[str, str],
        metadata_group: dict[str, Any],
        times: CompileTimes,
        cache_hit: bool,
    ) -> None:
        nonlocal trace_data
        extract_python_source_info(trace_data, src)

    triton.knobs.compilation.always_compile = True
    triton.knobs.compilation.listener = compile_listener
    torch.manual_seed(0)
    size = (1024, 1024)
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    tensor_add(a, b)
    assert "python_source" in trace_data
    assert "file_path" in trace_data["python_source"]


def test_convert():
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
