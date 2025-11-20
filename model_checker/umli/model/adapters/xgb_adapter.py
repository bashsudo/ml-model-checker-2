"""
XGBoostAdapter - Adapter for XGBoost models.
"""

from typing import Any, Optional

import xgboost as xgb

from model_checker.umli.model.unified_model import BaseModelAdapter


class XGBoostAdapter(BaseModelAdapter):
    """
    Adapter for XGBoost models (gradient boosting).
    """

    def __init__(self, backend_model: Any, config: Optional[dict] = None):
        """
        Initialize XGBoost adapter.

        Args:
            backend_model: XGBoost model instance
            config: Optional configuration dictionary
        """
        super().__init__(backend_model, config)

    def forward(self, batch: Any) -> Any:
        """
        Forward pass (prediction) for XGBoost models.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        return self.backend_model.predict(batch)

    def backward(self, loss: Any) -> None:
        """Not applicable for XGBoost models."""
        raise NotImplementedError("XGBoost models do not support backward()")

    def step(self) -> None:
        """Not applicable for XGBoost models."""
        raise NotImplementedError("XGBoost models do not support step()")

    def fit(self, dataset: Any) -> None:
        """
        Fit XGBoost model to dataset.

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

    def predict(self, batch: Any) -> Any:
        """
        Generate predictions.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        return self.backend_model.predict(batch)

    def save(self, path: str) -> None:
        """Save XGBoost model."""
        self.backend_model.save_model(path)

    def load(self, path: str) -> None:
        """Load XGBoost model."""
        self.backend_model = xgb.XGBModel()
        self.backend_model.load_model(path)
