"""
Serialization Utilities

Provides unified saving and loading interfaces for models and configurations.
"""

import pickle
from pathlib import Path
from typing import Any, Optional

import joblib
import torch
from tensorflow import keras  # pyright: ignore[reportAttributeAccessIssue]


def save_model(model: Any, path: str, format: Optional[str] = None) -> None:
    """
    Save a model to disk using the appropriate backend-specific method.

    Args:
        model: Model to save (should have a save() method or be serializable)
        path: File path to save the model
        format: Optional format specification ('pickle', 'pytorch', 'tensorflow', etc.)
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # If model has a save method, use it
    if hasattr(model, "save"):
        model.save(path)
        return

    # Otherwise, try to infer format from extension or model type
    if format is None:
        format = _infer_format(path, model)

    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(model, f)
    elif format == "pytorch":
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)
    elif format == "tensorflow":
        model.save(path)
    elif format == "sklearn":
        joblib.dump(model, path)
    else:
        # Default to pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(
    path: str, format: Optional[str] = None, model_class: Optional[Any] = None
) -> Any:
    """
    Load a model from disk.

    Args:
        path: File path to load the model from
        format: Optional format specification
        model_class: Optional model class to instantiate (for state_dict loading)

    Returns:
        Loaded model
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if format is None:
        format = _infer_format_from_path(path)

    if format == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif format == "pytorch":
        if model_class is not None:
            model = model_class()
            model.load_state_dict(torch.load(path))
            return model
        else:
            return torch.load(path)
    elif format == "tensorflow":
        return keras.models.load_model(path)
    elif format == "sklearn":
        return joblib.load(path)
    else:
        # Default to pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def _infer_format(path: str, model: Optional[Any] = None) -> str:
    """Infer serialization format from path or model type."""
    path_lower = path.lower()

    if path_lower.endswith(".pth") or path_lower.endswith(".pt"):
        return "pytorch"
    elif path_lower.endswith(".h5") or path_lower.endswith(".keras"):
        return "tensorflow"
    elif path_lower.endswith(".pkl") or path_lower.endswith(".pickle"):
        return "pickle"
    elif path_lower.endswith(".joblib"):
        return "sklearn"
    elif model is not None:
        model_type = str(type(model))
        if "torch" in model_type.lower():
            return "pytorch"
        elif "tensorflow" in model_type.lower() or "keras" in model_type.lower():
            return "tensorflow"
        elif "sklearn" in model_type.lower():
            return "sklearn"

    return "pickle"


def _infer_format_from_path(path: str) -> str:
    """Infer format from file path extension."""
    return _infer_format(path, None)
