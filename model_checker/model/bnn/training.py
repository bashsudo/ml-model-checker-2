"""
Bayesian Training with KL Divergence Regularization

Implements Bayes by Backprop training algorithm with KL divergence
regularization between posterior and prior distributions.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_checker.model.bnn.bayesian_layer import BayesianLayer


def log_gaussian(
    x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probability of x under Gaussian(mu, exp(logvar)).

    Args:
        x: Value to evaluate
        mu: Mean of Gaussian
        logvar: Log variance of Gaussian

    Returns:
        Log probability
    """
    return -0.5 * (math.log(2 * math.pi) + logvar + ((x - mu) ** 2) / torch.exp(logvar))


def log_mixture_prior(
    x: torch.Tensor, pi: float = 0.5, sigma1: float = 0.1, sigma2: float = 1.0
) -> torch.Tensor:
    """
    Evaluate log probability of weight under two-component Gaussian mixture prior.

    Args:
        x: Weight value
        pi: Mixture weight (default 0.5)
        sigma1: Standard deviation of first Gaussian component
        sigma2: Standard deviation of second Gaussian component

    Returns:
        Log probability under mixture prior
    """
    gaussian1 = torch.exp(-0.5 * (x / sigma1) ** 2) / (sigma1 * math.sqrt(2 * math.pi))
    gaussian2 = torch.exp(-0.5 * (x / sigma2) ** 2) / (sigma2 * math.sqrt(2 * math.pi))
    mix = pi * gaussian1 + (1 - pi) * gaussian2
    return torch.log(mix + 1e-8)


def kl_divergence(
    mu_q: torch.Tensor, logvar_q: torch.Tensor, sampled_w: torch.Tensor
) -> torch.Tensor:
    """
    Compute Monte Carlo estimate of KL divergence KL(q||p).

    Args:
        mu_q: Posterior mean
        logvar_q: Posterior log variance
        sampled_w: Sampled weights from forward pass

    Returns:
        KL divergence estimate
    """
    # Log probability of sampled weights under learned posterior
    log_q = log_gaussian(sampled_w, mu_q, logvar_q).sum()

    # Log probability of sampled weights under prior (mixture of Gaussians)
    log_p = log_mixture_prior(sampled_w).sum()

    # KL estimate = log_q - log_p
    return log_q - log_p


def compute_kl_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute total KL divergence loss for all Bayesian layers.

    Args:
        model: BayesianNetwork model

    Returns:
        Total KL divergence loss
    """
    kl_loss = 0.0
    bayesian_layers = [model.bayes_layer1, model.bayes_layer2]

    for layer in bayesian_layers:
        if isinstance(layer, BayesianLayer):
            # KL divergence for weights
            if layer.sampled_weight is not None:
                kl_loss += kl_divergence(
                    layer.weight, layer.weight_log_var, layer.sampled_weight
                )

            # KL divergence for biases
            if layer.sampled_bias is not None:
                kl_loss += kl_divergence(
                    layer.bias, layer.bias_log_var, layer.sampled_bias
                )

    return kl_loss  # type: ignore


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
    kl_weight: Optional[float] = None,
) -> float:
    """
    Train model for one epoch with Bayesian KL regularization.

    Args:
        model: BayesianNetwork model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to run on ('cpu' or 'cuda')
        kl_weight: Weight for KL divergence term. If None, uses 1/N where N is dataset size.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Set KL weight if not provided (use 1/N where N is dataset size)
    if kl_weight is None:
        kl_weight = 1.0 / len(dataloader.dataset)  # type: ignore

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass: sample weights internally in Bayesian layers
        outputs = model(batch_x)

        # Cross-entropy loss (negative log-likelihood)
        data_loss = criterion(outputs, batch_y)

        # KL divergence term (Monte Carlo estimate)
        kl_loss = compute_kl_loss(model)

        # Total loss = Data loss + scaled KL loss
        loss = data_loss + kl_weight * kl_loss

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train_bnn(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 10,
    device: str = "cpu",
    kl_weight: Optional[float] = None,
    verbose: bool = True,
) -> list:
    """
    Train Bayesian Neural Network for multiple epochs.

    Args:
        model: BayesianNetwork model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        epochs: Number of training epochs
        device: Device to run on
        kl_weight: Weight for KL divergence term
        verbose: If True, print training progress

    Returns:
        List of average losses per epoch
    """
    model.to(device)
    losses = []

    for epoch in range(epochs):
        avg_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, kl_weight
        )
        losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses
