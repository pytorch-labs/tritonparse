from collections import defaultdict
from typing import Any, Union

import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource, IRSource
from triton.knobs import CompileTimes

from .tritonparse.structured_logging.py import extract_python_source_info


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
