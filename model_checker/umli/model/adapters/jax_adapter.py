"""
JAXAdapter - Adapter for JAX models.
"""

import pickle
from typing import Any, Optional

import jax
import jax.numpy as jnp

from model_checker.umli.model.unified_model import BaseModelAdapter


class JAXAdapter(BaseModelAdapter):
    """
    Adapter for JAX models (gradient-based).
    """

    def __init__(
        self,
        backend_model: Any,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize JAX adapter.

        Args:
            backend_model: JAX model (typically a function or Flax module)
            optimizer: Optional optimizer (optax optimizer)
            loss_fn: Optional loss function
            config: Optional configuration dictionary
        """
        super().__init__(backend_model, config)
        self.jax = jax
        self.jnp = jnp
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.params = None  # Model parameters

    def forward(self, batch: Any) -> Any:
        """
        Forward pass through the model.

        Args:
            batch: Input data

        Returns:
            Model output
        """
        if self.params is None:
            raise ValueError("Model parameters not initialized. Call fit() first.")
        return self.backend_model(self.params, batch)

    def backward(self, loss: Any) -> None:
        """
        Backward pass (gradient computation) for JAX.
        Gradients are computed on-demand, not stored.
        """
        pass  # JAX computes gradients on-demand

    def step(self) -> None:
        """
        Update parameters using optimizer.
        Requires gradients to be computed first.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer required for step()")
        # Step is typically called within fit() for JAX models

    def fit(self, dataset: Any) -> None:
        """
        Fit JAX model to dataset.
        This is a placeholder - JAX training loops are typically custom.

        Args:
            dataset: Training dataset
        """
        raise NotImplementedError(
            "JAX training loops are typically custom. Implement training logic manually."
        )

    def predict(self, batch: Any) -> Any:
        """
        Generate predictions.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        return self.forward(batch)

    def save(self, path: str) -> None:
        """Save JAX model parameters."""
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def load(self, path: str) -> None:
        """Load JAX model parameters."""
        with open(path, "rb") as f:
            self.params = pickle.load(f)
