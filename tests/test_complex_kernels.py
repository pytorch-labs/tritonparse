"""
A more complex test case involving two distinct Triton kernels, one of which uses autotuning.
This test is designed to validate the launch_diff functionality with multiple, varied launches.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python tests/test_complex_kernels.py
```
"""

import os

import torch
import triton
import triton.language as tl

import tritonparse.structured_logging
import tritonparse.utils

# Initialize logging
log_path = "./logs_complex"
tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"


# Kernel 1: Autotuned Matmul (simplified configs for small scale)
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
            },
            num_stages=1,
            num_warps=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
            },
            num_stages=1,
            num_warps=1,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
            },
            num_stages=1,
            num_warps=1,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP_SIZE_M=8,
        ACTIVATION=None,
    )
    return c


# Kernel 2: Fused element-wise operation
@triton.jit
def fused_op_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    n_elements,
    scale_factor: float,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    result = a * b * scale_factor + c
    if ACTIVATION == "relu":
        result = tl.where(result > 0, result, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)


def fused_op(a, b, c, scale_factor: float, activation: str):
    n_elements = a.numel()
    output = torch.empty_like(a)
    BLOCK_SIZE = 8  # Reduced from 1024 for small scale testing
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_op_kernel[grid](
        a,
        b,
        c,
        output,
        n_elements,
        scale_factor,
        ACTIVATION=activation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def test_complex_kernels():
    """Main test function to run both kernels with varied parameters."""
    torch.manual_seed(0)

    # --- Matmul Launches (3 times with different configs) ---
    print("--- Testing Matmul Kernel (3 launches) ---")
    # Launch 1
    a1 = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    b1 = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    c1 = matmul(a1, b1)
    c1.sum()  # Synchronize
    print("Matmul Launch 1 (16x16 @ 16x16) done.")

    # Launch 2
    a2 = torch.randn((32, 16), device="cuda", dtype=torch.float16)
    b2 = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    c2 = matmul(a2, b2)
    c2.sum()  # Synchronize
    print("Matmul Launch 2 (32x16 @ 16x32) done.")

    # Launch 3
    a3 = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    b3 = torch.randn((32, 16), device="cuda", dtype=torch.float16)
    c3 = matmul(a3, b3)
    c3.sum()  # Synchronize
    print("Matmul Launch 3 (16x32 @ 32x16) done.")

    # --- Fused Op Launches (4 times with different parameters) ---
    print("\n--- Testing Fused Op Kernel (4 launches) ---")
    x = torch.randn((8,), device="cuda", dtype=torch.float32)
    y = torch.randn((8,), device="cuda", dtype=torch.float32)
    z = torch.randn((8,), device="cuda", dtype=torch.float32)

    # Launch 1
    print("Fused Op Launch 1: scale=1.0, activation=None")
    out1 = fused_op(x, y, z, scale_factor=1.0, activation="none")
    out1.sum()  # Synchronize

    # Launch 2
    print("Fused Op Launch 2: scale=2.5, activation=None")
    out2 = fused_op(x, y, z, scale_factor=2.5, activation="none")
    out2.sum()  # Synchronize

    # Launch 3
    print("Fused Op Launch 3: scale=1.0, activation='relu'")
    out3 = fused_op(x, y, z, scale_factor=1.0, activation="relu")
    out3.sum()  # Synchronize

    # Launch 4 (different size)
    print("Fused Op Launch 4: scale=1.0, activation='relu', different size")
    x_large = torch.randn((6,), device="cuda", dtype=torch.float32)
    y_large = torch.randn((6,), device="cuda", dtype=torch.float32)
    z_large = torch.randn((6,), device="cuda", dtype=torch.float32)
    out4 = fused_op(x_large, y_large, z_large, scale_factor=1.0, activation="relu")
    out4.sum()  # Synchronize
    print("All kernels executed.")


if __name__ == "__main__":
    test_complex_kernels()
    # Use unified_parse to process the generated logs
    tritonparse.utils.unified_parse(
        source=log_path, out="./parsed_output_complex", overwrite=True
    )
