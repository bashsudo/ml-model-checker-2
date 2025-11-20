"""
Bayesian Prediction with Uncertainty Quantification

Provides functions for making uncertainty-aware predictions using
multiple stochastic forward passes.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_prediction(
    model: nn.Module, x: torch.Tensor, samples: int = 20, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make Bayesian prediction with uncertainty estimation.

    Performs multiple stochastic forward passes to estimate predictive
    mean and uncertainty.

    Args:
        model: BayesianNetwork model
        x: Input tensor of shape (batch_size, 784)
        samples: Number of stochastic forward passes
        device: Device to run on

    Returns:
        Tuple of (mean_probabilities, std_probabilities)
        - mean_probabilities: Expected probability per class (batch_size, 10)
        - std_probabilities: Standard deviation representing uncertainty (batch_size, 10)
    """
    model.eval()
    outputs = []

    x = x.to(device)

    with torch.no_grad():
        # Take multiple stochastic forward passes to estimate uncertainty
        for _ in range(samples):
            out = model(x)
            probs = F.softmax(out, dim=1)
            outputs.append(probs)

        # Stack outputs into a single tensor (samples × batch × classes)
        stack = torch.stack(outputs)

        # Mean represents expected probability per class (posterior predictive mean)
        mean = stack.mean(dim=0)

        # Standard deviation represents model uncertainty (posterior variance)
        std = stack.std(dim=0)

    return mean, std


def predict_digit(
    model: nn.Module, x: torch.Tensor, samples: int = 20, device: str = "cpu"
) -> Tuple[int, float, float]:
    """
    Predict a single digit with confidence and uncertainty.

    Args:
        model: BayesianNetwork model
        x: Input tensor of shape (1, 784) - single flattened image
        samples: Number of stochastic forward passes
        device: Device to run on

    Returns:
        Tuple of (predicted_digit, confidence, uncertainty)
        - predicted_digit: Most likely digit (0-9)
        - confidence: Confidence in prediction (0-1)
        - uncertainty: Standard deviation of prediction (uncertainty measure)
    """
    mean, std = make_prediction(model, x, samples, device)

    # Extract most probable prediction and its confidence
    pred = torch.argmax(mean, dim=1).item()
    confidence = mean[0, pred].item()  # type: ignore

    # Use max std as uncertainty measure
    uncertainty = std[0, pred].item()  # type: ignore

    return pred, confidence, uncertainty  # type: ignore


def get_top_predictions(
    model: nn.Module,
    x: torch.Tensor,
    k: int = 2,
    samples: int = 20,
    device: str = "cpu",
) -> list:
    """
    Get top-k predictions with confidence scores.

    Args:
        model: BayesianNetwork model
        x: Input tensor of shape (1, 784)
        k: Number of top predictions to return
        samples: Number of stochastic forward passes
        device: Device to run on

    Returns:
        List of tuples (digit, confidence) sorted by confidence
    """
    mean, _ = make_prediction(model, x, samples, device)

    values, indices = torch.topk(mean, k=k, dim=1)

    predictions = [(indices[0, i].item(), values[0, i].item()) for i in range(k)]

    return predictions
