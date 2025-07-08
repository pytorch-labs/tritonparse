"""
Pytest configuration for tritonparse tests.
"""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests if CUDA is not available."""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "cuda" in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def cuda_device():
    """Get the first available CUDA device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        pytest.skip("CUDA not available")
