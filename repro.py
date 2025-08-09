import torch
import triton
import triton.language as tl

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
        a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c_block = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_block, mask=c_mask)


def main():
    # Device and seeding
    device = 'cuda:0'
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Tensor allocations
    a = torch.randn((16, 16), device=device, dtype=torch.float16)
    b = torch.randn((16, 16), device=device, dtype=torch.float16)
    c = torch.empty((16, 16), device=device, dtype=torch.float16)

    # Kernel launch parameters
    grid = (1,)
    num_warps = 1
    num_stages = 1
    kwargs = {
        'M': 16,
        'N': 16,
        'K': 16,
        'stride_am': 16,
        'stride_ak': 1,
        'stride_bk': 16,
        'stride_bn': 1,
        'stride_cm': 16,
        'stride_cn': 1,
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 16,
        'BLOCK_SIZE_K': 16,
        'GROUP_SIZE_M': 1,
        'ACTIVATION': "None"
    }

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        num_warps=num_warps,
        num_stages=num_stages,
        **kwargs
    )
    
    # Synchronize and print summary
    torch.cuda.synchronize()
    
    print("--- Repro Summary ---")
    print(f"a: shape={a.shape}, dtype={a.dtype}, stride={a.stride()}, device={a.device}")
    print(f"b: shape={b.shape}, dtype={b.dtype}, stride={b.stride()}, device={b.device}")
    print(f"c: shape={c.shape}, dtype={c.dtype}, stride={c.stride()}, device={c.device}")
    print(f"c[0, 0] = {c[0, 0]}")


if __name__ == "__main__":
    main()