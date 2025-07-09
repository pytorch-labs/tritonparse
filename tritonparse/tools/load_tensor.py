#!/usr/bin/env python3
"""
Simple tensor loading utility for tritonparse saved tensors.
Usage:
import tritonparse.tools.load_tensor as load_tensor
tensor = load_tensor.load_tensor(tensor_file_path, device)
"""

import hashlib
from pathlib import Path

import torch


def load_tensor(tensor_file_path: str, device: str = None) -> torch.Tensor:
    """
    Load a tensor from its file path and verify its integrity using the hash in the filename.

    Args:
        tensor_file_path (str): Direct path to the tensor .bin file. The filename should be
                               the hash of the file contents followed by .bin extension.
        device (str, optional): Device to load the tensor to (e.g., 'cuda:0', 'cpu').
                               If None, keeps the tensor on its original device.

    Returns:
        torch.Tensor: The loaded tensor (moved to the specified device if provided)

    Raises:
        FileNotFoundError: If the tensor file doesn't exist
        RuntimeError: If the tensor cannot be loaded
        ValueError: If the computed hash doesn't match the filename hash
    """
    blob_path = Path(tensor_file_path)

    if not blob_path.exists():
        raise FileNotFoundError(f"Tensor blob not found: {blob_path}")

    # Extract expected hash from filename (remove .bin extension)
    expected_hash = blob_path.stem

    # Compute actual hash of file contents
    with open(blob_path, "rb") as f:
        file_contents = f.read()
        computed_hash = hashlib.blake2b(file_contents).hexdigest()

    # Verify hash matches filename
    if computed_hash != expected_hash:
        raise ValueError(
            f"Hash verification failed: expected '{expected_hash}' but computed '{computed_hash}'"
        )

    try:
        # Load the tensor using torch.load (tensors are saved with torch.save)
        # If device is None, keep tensor on its original device, otherwise move to specified device
        tensor = torch.load(blob_path, map_location=device)
        return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {blob_path}: {str(e)}")
