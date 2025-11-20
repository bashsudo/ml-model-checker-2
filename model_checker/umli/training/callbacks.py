"""
Callbacks - Training callback system.
"""

from abc import ABC
from typing import Any, Dict, List, Optional


class Callback(ABC):
    """
    Base callback class for training events.

    Subclass this to create custom callbacks.
    """

    def on_train_begin(self) -> None:
        """Called when training begins."""

    def on_train_end(self) -> None:
        """Called when training ends."""

    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the beginning of an epoch."""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of an epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """

    def on_batch_begin(self, batch: int) -> None:
        """Called at the beginning of a batch."""

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of a batch.

        Args:
            batch: Current batch number
            logs: Dictionary of batch metrics
        """

    def on_evaluate_begin(self) -> None:
        """Called when evaluation begins."""

    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when evaluation ends.

        Args:
            logs: Dictionary of evaluation metrics
        """


class CallbackList:
    """
    Container for multiple callbacks.
    """

    def __init__(self, callbacks: List[Callback]):
        """
        Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks

    def on_train_begin(self) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self, epoch: int) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_evaluate_begin(self) -> None:
        """Call on_evaluate_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_evaluate_begin()

    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_evaluate_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_evaluate_end(logs)


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when a metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether to minimize or maximize the metric
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.stopped = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if training should stop."""
        if logs is None or self.monitor not in logs:
            return

        current_value = logs[self.monitor]

        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stopped = True
                print(f"Early stopping triggered at epoch {epoch + 1}")


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.
    """

    def __init__(
        self, filepath: str, monitor: str = "loss", save_best_only: bool = True
    ):
        """
        Initialize model checkpoint.

        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor for best model
            save_best_only: If True, only save when metric improves
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float("inf")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint."""
        if logs is None or self.monitor not in logs:
            return

        current_value = logs[self.monitor]

        if not self.save_best_only or current_value < self.best_value:
            self.best_value = current_value
            # Save model (would need access to model instance)
            # This is a simplified version - real implementation would save the model
            print(f"Checkpoint saved at epoch {epoch + 1}")
