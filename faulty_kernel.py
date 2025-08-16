
import triton
import triton.language as tl

@triton.jit
def kernel(
    X, Yv, Yi, Bits, S,
    stride_xm, stride_ym,
    USE_PROVIDED_INDX, stride_rm, stride_rn,
    n_rows, n_expts_tot,
    BLOCK_S: tl.constexpr, s_blocks: tl.constexpr, APPLY_SOFTMAX: tl.constexpr,
    BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr
):
    # This is a simplified version of the kernel logic, designed to demonstrate the
    # out-of-bounds access caused by a misconfigured grid.

    # Get the program ID for the row dimension.
    pid_m = tl.program_id(0)

    # Calculate the starting row for this block.
    # This is the line that will cause the error when pid_m is out of bounds.
    start_m = pid_m * BLOCK_M
    
    # Create an offset range for the rows this block will process.
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # Create an offset range for the columns.
    offs_n = tl.arange(0, BLOCK_N)

    # Create pointers to the input tensor X.
    # The shape is (BLOCK_M, BLOCK_N).
    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]

    # Create a mask to avoid reading from padding rows.
    # This mask is what a *correct* kernel would use to prevent OOB access.
    # However, the OOB access happens *before* the mask is applied if the
    # pointer calculation itself goes out of bounds due to a large pid_m.
    row_mask = offs_m < n_rows

    # Load data from X.
    # When pid_m is too large, `start_m` will be >= n_rows.
    # `offs_m` will point to memory outside the tensor's allocation.
    # This `tl.load` will trigger the CUDA_ERROR_ILLEGAL_ADDRESS.
    x_data = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)

    # Dummy computation to use the loaded data.
    sum_x = tl.sum(x_data, axis=1)

    # Create pointers to the output tensor Yv.
    y_ptrs = Yv + offs_m * stride_ym

    # Write the result to Yv.
    tl.store(y_ptrs, sum_x, mask=row_mask)
