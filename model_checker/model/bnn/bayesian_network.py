"""
Bayesian Neural Network Architecture

A neural network consisting of Bayesian layers for uncertainty quantification
in MNIST digit classification.
"""

import torch
import torch.nn as nn

from model_checker.model.bnn.bayesian_layer import BayesianLayer


class BayesianNetwork(nn.Module):
    """
    Bayesian Neural Network for MNIST classification.

    Architecture:
    - Input: 784 features (28x28 flattened MNIST images)
    - Bayesian Layer 1: 784 -> 128 hidden units
    - Bayesian Layer 2: 128 -> 64 hidden units
    - Output Layer: 64 -> 10 classes (deterministic)
    """

    def __init__(self):
        """Initialize Bayesian Neural Network."""
        super().__init__()

        # First Bayesian layer: 784 input features to 128 hidden units
        self.bayes_layer1 = BayesianLayer(784, 128)

        # Second Bayesian layer: 128 to 64 hidden units
        self.bayes_layer2 = BayesianLayer(128, 64)

        # Final deterministic output layer: 64 -> 10 (digits 0-9)
        self.out = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Bayesian network.

        Args:
            x: Input tensor of shape (batch_size, 784)

        Returns:
            Logits tensor of shape (batch_size, 10)
        """
        # Apply first Bayesian layer with ReLU activation
        x = torch.relu(self.bayes_layer1(x))

        # Apply second Bayesian layer with ReLU activation
        x = torch.relu(self.bayes_layer2(x))

        # Apply final linear output layer to get logits
        x = self.out(x)
        return x
