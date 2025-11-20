"""
CustomModelAdapter - Adapter for user-defined models.
"""

import pickle
from typing import Any, Callable, Optional

from model_checker.umli.model.unified_model import BaseModelAdapter


class CustomModelAdapter(BaseModelAdapter):
    """
    Adapter for custom/user-defined models.

    Allows users to wrap their own model implementations
    with the UnifiedModel interface.
    """

    def __init__(
        self,
        backend_model: Any,
        forward_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
        fit_fn: Optional[Callable] = None,
        save_fn: Optional[Callable] = None,
        load_fn: Optional[Callable] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize custom adapter.

        Args:
            backend_model: Custom model instance
            forward_fn: Optional custom forward function
            predict_fn: Optional custom predict function
            fit_fn: Optional custom fit function
            save_fn: Optional custom save function
            load_fn: Optional custom load function
            config: Optional configuration dictionary
        """
        super().__init__(backend_model, config)
        self.forward_fn = forward_fn or (lambda batch: backend_model(batch))
        self.predict_fn = predict_fn or (
            lambda batch: (
                backend_model.predict(batch)
                if hasattr(backend_model, "predict")
                else self.forward_fn(batch)
            )
        )
        self.fit_fn = fit_fn or (
            lambda dataset: (
                backend_model.fit(dataset) if hasattr(backend_model, "fit") else None
            )
        )
        self.save_fn = save_fn or (lambda path: self._default_save(path))
        self.load_fn = load_fn or (lambda path: self._default_load(path))

    def forward(self, batch: Any) -> Any:
        """Forward pass using custom function."""
        return self.forward_fn(batch)

    def backward(self, loss: Any) -> None:
        """Backward pass - implement if needed."""
        if hasattr(self.backend_model, "backward"):
            self.backend_model.backward(loss)
        else:
            raise NotImplementedError("Custom model does not support backward()")

    def step(self) -> None:
        """Step - implement if needed."""
        if hasattr(self.backend_model, "step"):
            self.backend_model.step()
        else:
            raise NotImplementedError("Custom model does not support step()")

    def fit(self, dataset: Any) -> None:
        """Fit using custom function."""
        self.fit_fn(dataset)

    def predict(self, batch: Any) -> Any:
        """Predict using custom function."""
        return self.predict_fn(batch)

    def save(self, path: str) -> None:
        """Save using custom function."""
        self.save_fn(path)

    def load(self, path: str) -> None:
        """Load using custom function."""
        self.load_fn(path)

    def _default_save(self, path: str) -> None:
        """Default save implementation."""
        with open(path, "wb") as f:
            pickle.dump(self.backend_model, f)

    def _default_load(self, path: str) -> None:
        """Default load implementation."""
        with open(path, "rb") as f:
            self.backend_model = pickle.load(f)
