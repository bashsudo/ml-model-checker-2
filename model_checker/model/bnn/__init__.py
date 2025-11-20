"""
Bayesian Neural Network (BNN) for MNIST classification.

This module provides a Bayesian Neural Network implementation using UMLI,
featuring uncertainty quantification through weight sampling.
"""

from model_checker.model.bnn.bayesian_layer import BayesianLayer
from model_checker.model.bnn.bayesian_network import BayesianNetwork

__all__ = ["BayesianLayer", "BayesianNetwork"]
