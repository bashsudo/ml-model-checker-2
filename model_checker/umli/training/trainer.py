"""
Trainer - Training orchestration layer.
"""

from typing import Any, List, Optional

import numpy as np

from model_checker.umli.data.unified_dataset import UnifiedDataset
from model_checker.umli.model.unified_model import UnifiedModel
from model_checker.umli.training.callbacks import Callback, CallbackList


class Trainer:
    """
    Backend-agnostic trainer for unified ML models.

    Supports both stateless-fit models (e.g., sklearn) and
    gradient-based iterative models (e.g., PyTorch, TensorFlow).
    """

    def __init__(
        self,
        model: UnifiedModel,
        dataset: UnifiedDataset,
        config: Optional[dict] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: UnifiedModel instance
            dataset: UnifiedDataset instance
            config: Optional training configuration
            callbacks: Optional list of callbacks
        """
        self.model = model
        self.dataset = dataset
        self.config = config or {}
        self.callbacks = CallbackList(callbacks or [])

        # Training parameters from config
        self.epochs = self.config.get("epochs", 1)
        self.batch_size = self.config.get("batch_size", 32)
        self.validation_split = self.config.get("validation_split", 0.0)
        self.verbose = self.config.get("verbose", True)

    def train(self) -> dict:
        """
        Train the model.

        Automatically detects model type and uses appropriate training procedure:
        - Stateless-fit models: single call to fit()
        - Gradient-based models: iterative training loop

        Returns:
            Dictionary containing training history/metrics
        """
        self.callbacks.on_train_begin()

        # Check if model supports fit() (stateless-fit models)
        if hasattr(self.model, "fit") and self._is_stateless_fit_model():
            history = self._train_stateless_fit()
        else:
            # Gradient-based iterative training
            history = self._train_iterative()

        self.callbacks.on_train_end()
        return history

    def _is_stateless_fit_model(self) -> bool:
        """
        Check if model is a stateless-fit model (e.g., sklearn).

        Returns:
            True if model is stateless-fit
        """
        # Try to detect sklearn models
        model_type = str(
            type(
                self.model.backend_model
                if hasattr(self.model, "backend_model")
                else self.model
            )
        )
        return "sklearn" in model_type.lower() or "xgboost" in model_type.lower()

    def _train_stateless_fit(self) -> dict:
        """Train stateless-fit models (single fit call)."""
        self.callbacks.on_epoch_begin(0)

        # Split dataset if validation split is specified
        if self.validation_split > 0:
            train_dataset, val_dataset = self.dataset.split(
                [1 - self.validation_split, self.validation_split]
            )
        else:
            train_dataset = self.dataset
            val_dataset = None

        # Fit model
        self.model.fit(train_dataset)

        # Evaluate if validation set exists
        metrics = {}
        if val_dataset is not None:
            metrics = self.evaluate(val_dataset)

        self.callbacks.on_epoch_end(0, metrics)
        return {"metrics": metrics}

    def _train_iterative(self) -> dict:
        """Train gradient-based models (iterative training loop)."""
        history = {"loss": [], "metrics": []}

        # Split dataset if validation split is specified
        if self.validation_split > 0:
            train_dataset, val_dataset = self.dataset.split(
                [1 - self.validation_split, self.validation_split]
            )
        else:
            train_dataset = self.dataset
            val_dataset = None

        for epoch in range(self.epochs):
            self.callbacks.on_epoch_begin(epoch)

            # Training loop
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(train_dataset), self.batch_size):
                batch_end = min(i + self.batch_size, len(train_dataset))
                batch = self._get_batch(train_dataset, i, batch_end)

                self.callbacks.on_batch_begin(i // self.batch_size)

                # Forward pass
                output = self.model.forward(batch["inputs"])

                # Compute loss (simplified - assumes targets are available)
                if batch.get("targets") is not None:
                    # This is a simplified loss computation
                    # Real implementations would use proper loss functions
                    if hasattr(self.model, "loss_fn"):
                        loss = self.model.loss_fn(output, batch["targets"])
                    else:
                        # Default: mean squared error
                        loss = np.mean((output - batch["targets"]) ** 2)

                    # Backward and step if supported
                    if hasattr(self.model, "backward"):
                        self.model.backward(loss)
                    if hasattr(self.model, "step"):
                        self.model.step()

                    epoch_loss += float(loss)
                    num_batches += 1

                self.callbacks.on_batch_end(
                    i // self.batch_size, {"loss": loss if "loss" in locals() else None}
                )

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            history["loss"].append(avg_loss)

            # Validation
            metrics = {}
            if val_dataset is not None:
                metrics = self.evaluate(val_dataset)
                history["metrics"].append(metrics)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")

            self.callbacks.on_epoch_end(epoch, {"loss": avg_loss, **metrics})

        return history

    def _get_batch(self, dataset: UnifiedDataset, start: int, end: int) -> dict:
        """Get a batch of data from dataset."""
        batch_inputs = []
        batch_targets = []

        for i in range(start, end):
            record = dataset[i]
            batch_inputs.append(record["inputs"])
            if record.get("targets") is not None:
                batch_targets.append(record["targets"])

        # Convert to appropriate format
        batch_inputs = np.array(batch_inputs)
        batch_targets = np.array(batch_targets) if batch_targets else None

        return {"inputs": batch_inputs, "targets": batch_targets}

    def evaluate(self, dataset: Optional[UnifiedDataset] = None) -> dict:
        """
        Evaluate the model on a dataset.

        Args:
            dataset: Optional dataset to evaluate on (uses training dataset if None)

        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataset = dataset or self.dataset
        self.callbacks.on_evaluate_begin()

        predictions = []
        targets = []

        for i in range(len(eval_dataset)):
            record = eval_dataset[i]
            pred = self.model.predict(record["inputs"])
            predictions.append(pred)
            if record.get("targets") is not None:
                targets.append(record["targets"])

        # Compute metrics (simplified - real implementation would use proper metrics)
        metrics = {}
        if targets:
            predictions = np.array(predictions)
            targets = np.array(targets)

            # Mean squared error
            mse = np.mean((predictions - targets) ** 2)
            metrics["mse"] = float(mse)

            # Mean absolute error
            mae = np.mean(np.abs(predictions - targets))
            metrics["mae"] = float(mae)

        self.callbacks.on_evaluate_end(metrics)
        return metrics

    def predict(self, inputs: Any) -> Any:
        """
        Generate predictions for inputs.

        Args:
            inputs: Input data

        Returns:
            Predictions
        """
        return self.model.predict(inputs)
