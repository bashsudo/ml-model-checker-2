from typing import Tuple

import numpy as np
from scipy.special import gammaln


def dirichlet_log_likelihood(x: np.ndarray, alpha: np.ndarray) -> float:
    """
    Compute log-likelihood of x under Dirichlet(alpha).

    Args:
        x: Point to evaluate (probability distribution)
        alpha: Dirichlet concentration parameters

    Returns:
        Log-likelihood value
    """
    # Ensure x is valid probability distribution
    x = np.clip(x, 1e-10, 1.0)
    x = x / x.sum()

    # Log-likelihood: log Γ(α₀) - Σ log Γ(αᵢ) + Σ (αᵢ - 1) log xᵢ
    alpha_0 = alpha.sum()
    log_likelihood = (
        gammaln(alpha_0) - np.sum(gammaln(alpha)) + np.sum((alpha - 1) * np.log(x))
    )
    return log_likelihood


def dirichlet_mean_variance(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and variance of Dirichlet distribution.

    Args:
        alpha: Dirichlet concentration parameters

    Returns:
        Tuple of (mean, variance) arrays
    """
    alpha_0 = alpha.sum()
    mean = alpha / alpha_0
    variance = (alpha * (alpha_0 - alpha)) / (alpha_0**2 * (alpha_0 + 1))
    return mean, variance


def check_plausibility(
    candidate: np.ndarray,
    alpha: np.ndarray,
    k: float = 3.0,
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Check if a candidate distribution is plausible under Dirichlet(alpha).

    Uses k standard deviations from the mean as the threshold.

    Args:
        candidate: Candidate probability distribution to test
        alpha: Dirichlet concentration parameters
        k: Number of standard deviations for threshold (default 3)

    Returns:
        Tuple of (is_plausible, log_likelihood, mean, std_dev)
    """
    mean, variance = dirichlet_mean_variance(alpha)
    std_dev = np.sqrt(variance)

    # Check if candidate is within k standard deviations for each component
    deviations = np.abs(candidate - mean) / (std_dev + 1e-10)
    max_deviation = np.max(deviations)
    is_plausible = bool(max_deviation <= k)

    # Also compute log-likelihood
    log_likelihood = dirichlet_log_likelihood(candidate, alpha)

    return is_plausible, log_likelihood, mean, std_dev
