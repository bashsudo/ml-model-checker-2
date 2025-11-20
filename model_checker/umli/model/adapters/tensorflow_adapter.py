"""
TensorFlowAdapter - Adapter for TensorFlow/Keras models.
"""

from typing import Any, Optional

import tensorflow as tf
from tensorflow import keras  # pyright: ignore[reportAttributeAccessIssue]

from model_checker.umli.model.unified_model import BaseModelAdapter


class TensorFlowAdapter(BaseModelAdapter):
    """
    Adapter for TensorFlow/Keras models (gradient-based).
    """

    def __init__(
        self,
        backend_model: Any,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize TensorFlow adapter.

        Args:
            backend_model: TensorFlow/Keras model instance
            optimizer: Optional optimizer
            loss_fn: Optional loss function
            config: Optional configuration dictionary
        """
        super().__init__(backend_model, config)
        self.tf = tf
        if optimizer:
            backend_model.compile(optimizer=optimizer, loss=loss_fn)

    def forward(self, batch: Any) -> Any:
        """
        Forward pass through the model.

        Args:
            batch: Input data

        Returns:
            Model output
        """
        return self.backend_model(batch, training=False)

    def backward(self, loss: Any) -> None:
        """
        Backward pass is handled internally by TensorFlow.
        This is a no-op for TensorFlow models.
        """

    def step(self) -> None:
        """
        Step is handled internally by TensorFlow during fit().
        This is a no-op for TensorFlow models.
        """

    def fit(self, dataset: Any) -> None:
        """
        Fit TensorFlow model to dataset.

        Args:
            dataset: Training dataset
        """
        if hasattr(dataset, "to_numpy"):
            X, y = dataset.to_numpy()
            self.backend_model.fit(X, y, verbose=0)
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
            self.backend_model.fit(X, y, verbose=0)
        else:
            self.backend_model.fit(dataset, verbose=0)

    def predict(self, batch: Any) -> Any:
        """
        Generate predictions.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        return self.backend_model.predict(batch, verbose=0)

    def save(self, path: str) -> None:
        """Save TensorFlow model."""
        self.backend_model.save(path)

    def load(self, path: str) -> None:
        """Load TensorFlow model."""
        self.backend_model = keras.models.load_model(path)
