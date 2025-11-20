from typing import Optional

import numpy as np


def generate_fake_distribution(
    num_classes: int = 10, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a random "fake" probability distribution over digits.

    Args:
        num_classes: Number of classes (default 10 for digits 0-9)
        rng: Optional random number generator

    Returns:
        Valid probability distribution (sums to 1, all non-negative)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random non-negative integers for each class
    fake_counts = rng.integers(0, 100, size=num_classes)
    # Normalize to get valid probability distribution
    fake_dist = fake_counts / fake_counts.sum()
    return fake_dist


def add_noise_to_distribution(
    dist: np.ndarray,
    noise_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add noise to a probability distribution while keeping it valid.

    Uses Dirichlet noise to ensure the result is still a valid probability distribution.

    Args:
        dist: Input probability distribution
        noise_scale: Scale of noise (larger = more noise)
        rng: Optional random number generator

    Returns:
        Noisy probability distribution (still sums to 1, all non-negative)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Use Dirichlet noise: sample from Dirichlet(alpha) where alpha is proportional to dist
    # Higher noise_scale means lower concentration (more noise)
    concentration = 1.0 / (noise_scale + 1e-6)
    alpha = dist * concentration + 1e-6  # Add small epsilon to avoid zeros
    noisy_dist = rng.dirichlet(alpha)
    return noisy_dist
