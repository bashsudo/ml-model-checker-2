"""
Bayesian Layer Implementation

A Bayesian fully connected layer that samples weights and biases
from learned Gaussian distributions using the reparameterization trick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLayer(nn.Module):
    """
    Bayesian fully connected layer with learnable weight and bias distributions.

    Uses the reparameterization trick to sample weights during forward pass,
    enabling Bayesian inference and uncertainty quantification.
    """

    def __init__(self, inputs: int, outputs: int, variance: float = 1.0):
        """
        Initialize Bayesian layer.

        Args:
            inputs: Number of input features
            outputs: Number of output features
            variance: Scaling factor for random noise in weight and bias sampling
        """
        super().__init__()

        # Weight mean: learnable mean of weight distribution (outputs x inputs)
        self.weight = nn.Parameter(torch.Tensor(outputs, inputs).normal_(0, 0.1))

        # Weight log variance: log variance of weights, initialized to -5 (small uncertainty)
        self.weight_log_var = nn.Parameter(torch.Tensor(outputs, inputs).fill_(-5))

        # Bias mean: learnable mean of bias distribution (one per output neuron)
        self.bias = nn.Parameter(torch.Tensor(outputs).normal_(0, 0.1))

        # Bias log variance: log variance of bias distribution
        self.bias_log_var = nn.Parameter(torch.Tensor(outputs).fill_(-5))

        # Store sampled weights/bias for KL divergence computation
        self.sampled_weight = None
        self.sampled_bias = None

        # Variance scaling factor for noise
        self.variance = variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with stochastic weight sampling.

        Args:
            x: Input tensor of shape (batch_size, inputs)

        Returns:
            Output tensor of shape (batch_size, outputs)
        """
        # Compute standard deviation from log variance: σ = exp(0.5 * logVar)
        weight_sd = torch.exp(0.5 * self.weight_log_var)
        bias_sd = torch.exp(0.5 * self.bias_log_var)

        # Sample random Gaussian noise ~ N(0, 1) and scale by variance
        weight_eps = torch.randn_like(weight_sd) * self.variance
        bias_eps = torch.randn_like(bias_sd) * self.variance

        # Apply reparameterization trick: weight = μ + σ * ε
        self.sampled_weight = self.weight + weight_sd * weight_eps
        self.sampled_bias = self.bias + bias_sd * bias_eps

        # Perform linear layer operation: y = xW^T + b
        return F.linear(x, self.sampled_weight, self.sampled_bias)
