"""
PyTorchAdapter - Adapter for PyTorch models.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from model_checker.umli.model.unified_model import BaseModelAdapter
from model_checker.umli.utils.device import get_device, to_device


class PyTorchAdapter(BaseModelAdapter):
    """
    Adapter for PyTorch neural network models (gradient-based).

    Supports training with forward, backward, and step operations.
    """

    def __init__(
        self,
        backend_model: Any,  # nn.Module when torch is available
        optimizer: Optional[
            Any
        ] = None,  # torch.optim.Optimizer when torch is available
        loss_fn: Optional[Any] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize PyTorch adapter.

        Args:
            backend_model: PyTorch nn.Module instance
            optimizer: Optional optimizer (required for training)
            loss_fn: Optional loss function
            config: Optional configuration dictionary
            device: Optional device ('cpu', 'cuda', etc.)
        """
        super().__init__(backend_model, config)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.MSELoss()
        self.device = device or get_device()
        self.backend_model.to(self.device)
        self._training_mode = False

    def forward(self, batch: Any) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: Input data (tensor or array)

        Returns:
            Model output tensor
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32)
        batch = to_device(batch, self.device)
        return self.backend_model(batch)

    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward pass (gradient computation).

        Args:
            loss: Computed loss tensor
        """
        if self.optimizer is None:
            raise ValueError("Optimizer required for backward()")
        self.optimizer.zero_grad()
        loss.backward()

    def step(self) -> None:
        """Update model parameters using computed gradients."""
        if self.optimizer is None:
            raise ValueError("Optimizer required for step()")
        self.optimizer.step()

    def fit(self, dataset: Any) -> None:
        """
        Fit is not directly applicable for PyTorch models.
        Use Trainer for iterative training.
        """
        raise NotImplementedError(
            "PyTorch models require iterative training. Use Trainer.train() instead."
        )

    def predict(self, batch: Any) -> torch.Tensor:
        """
        Generate predictions (inference mode).

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        self.backend_model.eval()
        with torch.no_grad():
            output = self.forward(batch)
        return output

    def save(self, path: str) -> None:
        """Save PyTorch model."""
        torch.save(
            {
                "model_state_dict": self.backend_model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load PyTorch model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.backend_model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "config" in checkpoint:
            self._config = checkpoint["config"]
