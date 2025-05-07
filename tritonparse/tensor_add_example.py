"""
Simple Triton kernel example for tensor addition.

This example demonstrates how to create a basic Triton kernel that adds two tensors element-wise.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing C = A + B element-wise.
    
    Args:
        a_ptr: Pointer to the first input tensor (A)
        b_ptr: Pointer to the second input tensor (B)
        c_ptr: Pointer to the output tensor (C)
        n_elements: Number of elements in the tensors
        BLOCK_SIZE: Size of the processing block
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    
    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the tensor size is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute element-wise addition
    c = a + b
    
    # Store the result
    tl.store(c_ptr + offsets, c, mask=mask)


def tensor_add(a, b):
    """
    Adds two tensors element-wise using Triton kernel.
    
    Args:
        a: First input tensor
        b: Second input tensor
        
    Returns:
        c: Output tensor containing the element-wise sum
    """
    # Ensure tensors are contiguous and have the same shape
    assert a.is_contiguous(), "Tensor 'a' must be contiguous"
    assert b.is_contiguous(), "Tensor 'b' must be contiguous"
    assert a.shape == b.shape, "Tensors must have the same shape"
    assert a.device.type == 'cuda', "Tensors must be on CUDA device"
    
    # Get tensor size
    n_elements = a.numel()
    
    # Allocate output tensor
    c = torch.empty_like(a)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    add_kernel[grid](
        a,
        b,
        c,
        n_elements,
        BLOCK_SIZE,
    )
    
    return c


def test_tensor_add():
    """
    Test function to verify the correctness of the triton tensor_add function.
    """
    # Create random tensors on GPU
    torch.manual_seed(0)
    size = (1024, 1024)
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Compute reference result using PyTorch
    c_ref = a + b
    
    # Compute result using Triton kernel
    c_triton = tensor_add(a, b)
    
    # Verify the results match
    assert torch.allclose(c_ref, c_triton, rtol=1e-5, atol=1e-5)
    print("✓ Triton and PyTorch results match!")
    
    # Simple performance benchmark
    import time
    
    # Warm up
    for _ in range(10):
        _ = tensor_add(a, b)
        _ = a + b
    
    # Time Triton implementation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = tensor_add(a, b)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Time PyTorch implementation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = a + b
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    print(f"Triton: {triton_time:.4f} seconds")
    print(f"PyTorch: {pytorch_time:.4f} seconds")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    test_tensor_add() 