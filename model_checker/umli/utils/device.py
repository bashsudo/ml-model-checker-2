"""
Device Management Utilities

Provides unified device management abstracted from backend details.
"""

from typing import Any, Optional

import tensorflow as tf
import torch


def get_device(device: Optional[str] = None) -> str:
    """
    Get the appropriate device string for the current backend.

    Args:
        device: Optional device specification ('cpu', 'cuda', 'gpu', etc.)

    Returns:
        Device string appropriate for the active backend
    """
    if device is not None:
        return device

    # Try to detect available devices
    if torch.cuda.is_available():
        return "cuda"

    if tf.config.list_physical_devices("GPU"):
        return "gpu"

    return "cpu"


def to_device(x: Any, device: str) -> Any:
    """
    Move data to the specified device.

    Args:
        x: Data to move (tensor, array, etc.)
        device: Target device ('cpu', 'cuda', 'gpu', etc.)

    Returns:
        Data on the specified device
    """
    if device == "cpu":
        return _to_cpu(x)
    elif device in ("cuda", "gpu"):
        return _to_gpu(x, device)
    else:
        raise ValueError(f"Unknown device: {device}")


def _to_cpu(x: Any) -> Any:
    """Move data to CPU."""
    if hasattr(x, "cpu"):  # PyTorch tensor
        return x.cpu()
    elif hasattr(x, "numpy"):  # TensorFlow tensor
        with tf.device("/CPU:0"):
            return tf.identity(x)
    else:
        return x


def _to_gpu(x: Any, device: str) -> Any:
    """Move data to GPU."""
    if hasattr(x, "cuda"):  # PyTorch tensor
        return x.cuda()
    elif hasattr(x, "to"):  # PyTorch tensor (alternative)
        return x.to(device)
    elif hasattr(x, "__class__") and "tensorflow" in str(type(x)).lower():
        device_name = "/GPU:0" if device == "gpu" else f"/{device.upper()}:0"
        with tf.device(device_name):
            return tf.identity(x)
    else:
        # For NumPy arrays, return as-is (NumPy doesn't support GPU directly)
        return x
