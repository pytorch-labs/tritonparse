import torch
import triton
import triton.language as tl

# The error analysis identified a tl.load operation as the likely source of error.
# The root cause, however, stems from the tensor allocation, where a strided view
# is created over a storage that is smaller than the view's dimensions imply.
# This script reproduces that exact scenario using the simplified kernel from the analysis.
# The kernel attempts to access elements near the end of the `y` tensor, which
# fall outside the bounds of its underlying, smaller storage, triggering a memory access error.

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    This kernel is a simplified version provided in the error analysis.
    It performs element-wise addition. The error is triggered when loading from y_ptr.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load from x is fine
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # LIKELY ERROR LOCATION: Loading from y will fail because its underlying storage is too small.
    # The mask protects against out-of-bounds access on the *view*, but not the *storage*.
    # The kernel will attempt to access indices up to 303, but the storage for y only has 301 elements.
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# Setup device
device = 'cuda'
if not torch.cuda.is_available():
    print("CUDA not available, skipping test.")
    exit()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# The original error occurred with tensors Yv and Yi, which had a shape of [76, 4]
# but were backed by a storage of size 301.
# 76 * 4 = 304 elements in the view, but only 301 elements in storage.
# We replicate this for a 1D tensor to be used with the simplified add_kernel.
n_elements = 304  # Total elements in the logical view
storage_size = 301 # Actual allocated elements in memory

# This tensor is allocated correctly.
x = torch.randn(n_elements, dtype=torch.float32, device=device)

# This tensor `y` is mis-configured to replicate the bug.
# It has a view of 304 elements but is backed by a storage of only 301 elements.
_storage_y = torch.empty((storage_size,), dtype=torch.float32, device=device)
y = _storage_y.as_strided(size=(n_elements,), stride=(1,))

# The output tensor is also allocated correctly.
output = torch.empty(n_elements, dtype=torch.float32, device=device)

# Kernel launch parameters
# A single block is sufficient to cover all elements and trigger the out-of-bounds access.
BLOCK_SIZE = 1024
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

# Launch the kernel. This is expected to fail with a CUDA error.
# The original compile options are used where applicable.
add_kernel[grid](
    x_ptr=x,
    y_ptr=y,
    output_ptr=output,
    n_elements=n_elements,
    BLOCK_SIZE=BLOCK_SIZE,
    num_warps=4,
    num_stages=3
)

# This summary will not be printed if the script fails as intended.
print("--- Repro Script Finished ---")
print("Summary of tensor properties:")
print(f"  x: shape={x.shape}, dtype={x.dtype}, stride={x.stride()}, storage_size={x.storage().size()}")
print(f"  y: shape={y.shape}, dtype={y.dtype}, stride={y.stride()}, storage_size={y.storage().size()}")
print(f"  output: shape={output.shape}, dtype={output.dtype}, stride={output.stride()}, storage_size={output.storage().size()}")
print("\nNOTE: If this message is printed, the script did NOT reproduce the error.")