"""
SklearnAdapter - Adapter for scikit-learn models.
"""

from typing import Any, Optional

import joblib
import numpy as np

from model_checker.umli.model.unified_model import BaseModelAdapter


class SklearnAdapter(BaseModelAdapter):
    """
    Adapter for scikit-learn models (stateless-fit models).

    Supports models like LinearRegression, SVM, KMeans, etc.
    """

    def __init__(self, backend_model: Any, config: Optional[dict] = None):
        """
        Initialize sklearn adapter.

        Args:
            backend_model: scikit-learn model instance
            config: Optional configuration dictionary
        """
        super().__init__(backend_model, config)
        if not hasattr(backend_model, "fit"):
            raise ValueError("Backend model must have a fit() method")

    def forward(self, batch: Any) -> np.ndarray:
        """
        Forward pass (prediction) for sklearn models.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        return self.backend_model.predict(batch)

    def backward(self, loss: Any) -> None:
        """Not applicable for sklearn models."""
        raise NotImplementedError("sklearn models do not support backward()")

    def step(self) -> None:
        """Not applicable for sklearn models."""
        raise NotImplementedError("sklearn models do not support step()")

    def fit(self, dataset: Any) -> None:
        """
        Fit the sklearn model to the dataset.

        Args:
            dataset: Training dataset (can be tuple of (X, y) or UnifiedDataset)
        """
        if isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
            self.backend_model.fit(X, y)
        elif hasattr(dataset, "to_numpy"):
            X, y = dataset.to_numpy()
            self.backend_model.fit(X, y)
        else:
            raise ValueError("Dataset must be (X, y) tuple or UnifiedDataset")

    def predict(self, batch: Any) -> np.ndarray:
        """
        Generate predictions.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        return self.backend_model.predict(batch)

    def save(self, path: str) -> None:
        """Save sklearn model."""
        joblib.dump(self.backend_model, path)

    def load(self, path: str) -> None:
        """Load sklearn model."""
        self.backend_model = joblib.load(path)
