"""
UnifiedConfig Schema

Defines the standard configuration schema for models, datasets, and training.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


class ModelConfig(TypedDict, total=False):
    """Model configuration schema."""

    type: str  # Model type (e.g., 'sklearn', 'pytorch', 'tensorflow')
    name: str  # Model name/identifier
    params: Dict[str, Any]  # Model-specific parameters
    optimizer: Optional[
        Dict[str, Any]
    ]  # Optimizer configuration (for gradient-based models)
    loss_fn: Optional[str]  # Loss function name
    device: Optional[str]  # Device ('cpu', 'cuda', etc.)


class DataConfig(TypedDict, total=False):
    """Dataset configuration schema."""

    type: str  # Dataset type (e.g., 'numpy', 'torch', 'hf', 'pandas')
    path: Optional[str]  # Path to dataset file
    input_shape: Optional[tuple]  # Input shape
    output_shape: Optional[tuple]  # Output shape
    preprocessing: Optional[Dict[str, Any]]  # Preprocessing steps


class TrainingConfig(TypedDict, total=False):
    """Training configuration schema."""

    epochs: int  # Number of training epochs
    batch_size: int  # Batch size
    learning_rate: float  # Learning rate
    validation_split: float  # Validation split ratio
    verbose: bool  # Verbose output
    callbacks: Optional[list]  # List of callback configurations


class UnifiedConfig(TypedDict, total=False):
    """
    Unified configuration schema.

    Combines model, data, and training configurations.
    """

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


@dataclass
class ModelConfigDC:
    """Model configuration (dataclass version)."""

    type: str
    name: str
    params: Dict[str, Any]
    optimizer: Optional[Dict[str, Any]] = None
    loss_fn: Optional[str] = None
    device: Optional[str] = None


@dataclass
class DataConfigDC:
    """Dataset configuration (dataclass version)."""

    type: str
    path: Optional[str] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    preprocessing: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfigDC:
    """Training configuration (dataclass version)."""

    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.0
    verbose: bool = True
    callbacks: Optional[list] = None


@dataclass
class UnifiedConfigDC:
    """Unified configuration (dataclass version)."""

    model: ModelConfigDC
    data: DataConfigDC
    training: TrainingConfigDC
