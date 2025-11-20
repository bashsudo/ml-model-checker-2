"""
UnifiedModel Interface

Defines the protocol that all model adapters must implement.
"""

from abc import ABC
from typing import Any, Optional, Protocol


class UnifiedModel(Protocol):
    """
    Unified interface for all ML models across different backends.

    This protocol defines the standard operations that all model adapters
    must support. Not all methods are required for all model types:
    - forward: required for all models
    - backward/step: only for gradient-based models
    - fit: for stateless-fit models (e.g., sklearn)
    - predict: required for all models
    """

    def forward(self, batch: Any) -> Any:
        """
        Forward propagation through the model.

        Args:
            batch: Input data batch

        Returns:
            Model output
        """
        ...

    def backward(self, loss: Any) -> None:
        """
        Backward propagation (gradient computation).
        Only implemented for gradient-based models.

        Args:
            loss: Computed loss value
        """
        ...

    def step(self) -> None:
        """
        Update model parameters using computed gradients.
        Only implemented for gradient-based models.
        """
        ...

    def fit(self, dataset: Any) -> None:
        """
        Fit the model to a dataset (single-shot training).
        Implemented for stateless-fit models (e.g., sklearn).

        Args:
            dataset: Training dataset
        """
        ...

    def predict(self, batch: Any) -> Any:
        """
        Generate predictions for input batch.

        Args:
            batch: Input data batch

        Returns:
            Predictions
        """
        ...

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save the model
        """
        ...

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: File path to load the model from
        """
        ...

    @property
    def config(self) -> dict:
        """
        Get model configuration in a standard schema.

        Returns:
            Dictionary containing model configuration
        """
        ...


class BaseModelAdapter(ABC):
    """
    Base class for model adapters providing common functionality.
    """

    def __init__(self, backend_model: Any, config: Optional[dict] = None):
        """
        Initialize adapter with backend model.

        Args:
            backend_model: The backend-specific model instance
            config: Optional configuration dictionary
        """
        self.backend_model = backend_model
        self._config = config or {}

    @property
    def config(self) -> dict:
        """Get model configuration."""
        return self._config

    def save(self, path: str) -> None:
        """Default save implementation - should be overridden by adapters."""
        raise NotImplementedError("save() must be implemented by adapter")

    def load(self, path: str) -> None:
        """Default load implementation - should be overridden by adapters."""
        raise NotImplementedError("load() must be implemented by adapter")
