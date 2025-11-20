"""
DataConverter - Unified type conversion utilities.
"""

from typing import Any, Union

import numpy as np
import torch


class DataConverter:
    """
    Utility class for converting between different numerical formats.

    Handles conversions between:
    - Python scalars
    - NumPy arrays
    - PyTorch tensors
    - Backend-specific numerical types
    """

    @staticmethod
    def to_numpy(x: Any) -> np.ndarray:
        """
        Convert input to NumPy array.

        Args:
            x: Input data (scalar, list, tensor, etc.)

        Returns:
            NumPy array
        """
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, (list, tuple)):
            return np.array(x)
        elif hasattr(x, "numpy"):  # PyTorch tensor
            return x.detach().cpu().numpy()
        elif hasattr(x, "numpy"):  # TensorFlow tensor
            return x.numpy()
        elif isinstance(x, (int, float, complex)):
            return np.array(x)
        else:
            try:
                return np.array(x)
            except Exception as e:
                raise TypeError(f"Cannot convert {type(x)} to NumPy array: {e}")

    @staticmethod
    def to_tensor(x: Any) -> Any:
        """
        Convert input to PyTorch tensor.

        Args:
            x: Input data

        Returns:
            PyTorch tensor
        """
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, (list, tuple)):
            return torch.tensor(x)
        else:
            return torch.tensor(x)

    @staticmethod
    def to_python(x: Any) -> Union[int, float, list, dict]:
        """
        Convert input to native Python types.

        Args:
            x: Input data

        Returns:
            Python native type (int, float, list, dict, etc.)
        """
        if isinstance(x, (int, float, str, bool, type(None))):
            return x
        elif isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            else:
                return x.tolist()
        elif hasattr(x, "item"):  # PyTorch scalar tensor
            return x.item()
        elif hasattr(x, "numpy"):  # TensorFlow tensor
            return DataConverter.to_python(x.numpy())
        elif isinstance(x, (list, tuple)):
            return [DataConverter.to_python(item) for item in x]
        elif isinstance(x, dict):
            return {k: DataConverter.to_python(v) for k, v in x.items()}
        else:
            try:
                return DataConverter.to_python(np.array(x))
            except Exception:
                return x

    @staticmethod
    def infer_type(x: Any) -> str:
        """
        Infer the type of the input data.

        Args:
            x: Input data

        Returns:
            String describing the type
        """
        if isinstance(x, np.ndarray):
            return "numpy"
        elif hasattr(x, "grad_fn") or (
            hasattr(x, "__class__") and "torch" in str(type(x))
        ):
            return "torch"
        elif hasattr(x, "__class__") and "tensorflow" in str(type(x)).lower():
            return "tensorflow"
        elif isinstance(x, (int, float, complex)):
            return "scalar"
        elif isinstance(x, (list, tuple)):
            return "list"
        elif isinstance(x, dict):
            return "dict"
        else:
            return "unknown"
