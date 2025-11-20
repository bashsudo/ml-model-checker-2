"""
Bootstrap-based posterior predictive distribution approximation.

Generalized to work with any UMLI model and flexible output shapes.
"""

from typing import List, Optional, Type

import numpy as np

from model_checker.umli.data.unified_dataset import UnifiedDataset
from model_checker.umli.model.unified_model import UnifiedModel


def create_bootstrap_datasets(
    dataset: UnifiedDataset,
    n_bootstrap: int,
    rng: Optional[np.random.Generator] = None,
) -> List[UnifiedDataset]:
    """
    Create B bootstrap datasets by sampling with replacement from the original dataset.

    Each bootstrap dataset has the same size as the original dataset.

    Args:
        dataset: Original UnifiedDataset
        n_bootstrap: Number of bootstrap datasets to create
        rng: Optional random number generator for reproducibility

    Returns:
        List of B bootstrap UnifiedDataset instances
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = len(dataset)
    bootstrap_datasets = []

    # Convert to numpy for easier manipulation
    X_list = []
    y_list = []
    for i in range(n_samples):
        record = dataset[i]
        X_list.append(record["inputs"])
        if record.get("targets") is not None:
            y_list.append(record["targets"])

    X = np.array(X_list)
    y = np.array(y_list) if y_list else None

    for _ in range(n_bootstrap):
        # Sample indices with replacement
        indices = rng.integers(0, n_samples, size=n_samples)

        # Create bootstrap dataset
        X_boot = X[indices]
        y_boot = y[indices] if y is not None else None

        # Wrap in UMLI adapter
        from model_checker.umli.data.adapters import NumpyDatasetAdapter

        bootstrap_datasets.append(NumpyDatasetAdapter(X_boot, y_boot))

    return bootstrap_datasets


def _create_model_instance(
    model_type: Type[UnifiedModel], config: Optional[dict] = None
) -> UnifiedModel:
    """
    Create a new model instance from a type and optional configuration.

    Args:
        model_type: Model class/type to instantiate
        config: Optional configuration dictionary for model hyperparameters

    Returns:
        New model instance
    """
    if config is not None:
        # Try to create with config
        try:
            return model_type(**config)  # type: ignore
        except (TypeError, ValueError) as e:
            # If config doesn't work, try creating backend model with config
            # This handles adapters that wrap backend models
            raise ValueError(
                f"Cannot create instance of {model_type} with provided config: {e}"
            )
    else:
        # Try to create with no arguments
        try:
            return model_type()  # type: ignore
        except (TypeError, ValueError):
            # If that fails, raise an error
            raise ValueError(
                f"Cannot create new instance of {model_type} without config. "
                "Please provide config parameter with model hyperparameters."
            )


def bootstrap_train_models(
    model_type: Type[UnifiedModel],
    dataset: UnifiedDataset,
    n_bootstrap: int,
    config: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[UnifiedModel]:
    """
    Train models on bootstrap datasets and return trained model instances.

    Works with any UMLI model that has a fit() method.

    Args:
        model_type: Type/class of UnifiedModel to instantiate and train
        dataset: Original UnifiedDataset to bootstrap from
        n_bootstrap: Number of bootstrap iterations
        config: Optional configuration dictionary for model hyperparameters.
                If None, model_type must be instantiable with no arguments.
        rng: Optional random number generator for reproducibility

    Returns:
        List of trained UnifiedModel instances, one for each bootstrap dataset
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create bootstrap datasets
    bootstrap_datasets = create_bootstrap_datasets(dataset, n_bootstrap, rng)

    trained_models = []

    # Create new model instances for each bootstrap
    for boot_dataset in bootstrap_datasets:
        new_model = _create_model_instance(model_type, config)
        new_model.fit(boot_dataset)
        trained_models.append(new_model)

    return trained_models
