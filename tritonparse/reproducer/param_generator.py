"""Parameter generator: produce deterministic allocation code from a bundle.

This module reduces LLM burden by emitting Python code that:
- selects a device
- seeds RNG
- allocates tensors with the exact shape/dtype/device/stride
- prepares scalar/constexpr kwargs

The generated code is intended to be inserted into the final repro script.
"""

import json
from typing import Any, Dict, List, Optional


def _torch_dtype_expr(dtype: str) -> str:
    mapping = {
        "float16": "torch.float16",
        "bfloat16": "torch.bfloat16",
        "float32": "torch.float32",
        "float": "torch.float32",
        "float64": "torch.float64",
        "half": "torch.float16",
        "bf16": "torch.bfloat16",
        "fp16": "torch.float16",
        "fp32": "torch.float32",
        "fp64": "torch.float64",
        "int8": "torch.int8",
        "int16": "torch.int16",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "long": "torch.int64",
        "bool": "torch.bool",
    }
    return mapping.get(str(dtype).lower(), "torch.float32")


def _compute_storage_numel(shape: List[int], stride: Optional[List[int]]) -> int:
    if not shape:
        return 1
    if not stride:
        # contiguous default
        numel = 1
        for s in shape:
            numel *= int(s)
        return numel
    # minimal storage size (in elements) to support the given logical shape/stride
    max_index = 0
    for dim, (sz, st) in enumerate(zip(shape, stride)):
        if sz <= 0:
            continue
        max_index = max(max_index, (int(sz) - 1) * int(st))
    return int(max_index) + 1


def _emit_tensor_alloc(name: str, spec: Dict[str, Any]) -> str:
    shape = spec.get("shape") or []
    dtype = _torch_dtype_expr(spec.get("dtype"))
    device = spec.get("device") or "cuda:0"
    stride = spec.get("stride")

    # ensure ints
    shape = [int(s) for s in shape]
    if stride is not None:
        stride_list = [int(x) for x in stride]
    else:
        stride_list = None

    lines: List[str] = []
    # allocate backing storage
    storage_numel = _compute_storage_numel(shape, stride_list)
    lines.append(
        f"# {name}: shape={shape}, dtype={dtype}, device={device}, stride={stride_list}"
    )
    lines.append(
        f"_storage_{name} = torch.empty(({storage_numel},), dtype={dtype}, device=device)"
    )
    if stride_list:
        # Create an as_strided view over the 1D storage
        sizes_expr = str(tuple(shape))
        strides_expr = str(tuple(stride_list))
        lines.append(
            f"{name} = _storage_{name}.as_strided(size={sizes_expr}, stride={strides_expr})"
        )
    else:
        # contiguous allocation
        size_expr = str(tuple(shape))
        lines.append(f"{name} = torch.empty({size_expr}, dtype={dtype}, device=device)")
    return "\n".join(lines)


def _emit_scalar(name: str, spec: Dict[str, Any]) -> str:
    value = spec.get("value")
    # Preserve JSON-serializable value as-is
    return f"{name} = {json.dumps(value)}"


def generate_allocation_snippet(bundle: Dict[str, Any]) -> str:
    """Generate a self-contained code snippet that:
    - imports torch
    - sets device
    - seeds RNG
    - allocates tensors and defines scalars for all args
    Returns Python source as a string.
    """
    tensor_args: Dict[str, Any] = bundle.get("tensor_args", {}) or {}
    args_all: Dict[str, Any] = bundle.get("args", {}) or {}

    # Pick device from any tensor arg, fallback to cuda:0
    device = "cuda:0"
    for spec in tensor_args.values():
        dev = spec.get("device")
        if dev:
            device = str(dev)
            break

    lines: List[str] = []
    lines.append("import torch")
    lines.append(f"device = '{device}'")
    lines.append("torch.manual_seed(0)")
    lines.append("if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)")
    lines.append("")

    # Emit tensors first for names with type==tensor in args_all
    for name, spec in args_all.items():
        if isinstance(spec, dict) and spec.get("type") == "tensor":
            lines.append(_emit_tensor_alloc(name, spec))
            lines.append("")

    # Emit non-tensor scalars next
    for name, spec in args_all.items():
        if not isinstance(spec, dict) or spec.get("type") == "tensor":
            continue
        lines.append(_emit_scalar(name, spec))
    return "\n".join(lines)


def generate_kwargs_dict(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Return a kwargs dict derived from bundle['launch']['kwargs'] suitable for kernel call."""
    launch = bundle.get("launch", {}) or {}
    kwargs = launch.get("kwargs", {}) or {}
    return kwargs
