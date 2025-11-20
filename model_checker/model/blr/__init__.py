"""
Bayesian Linear Regression (BLR) for N-dimensional inputs.

This module provides Bayesian Linear Regression implementation using UMLI,
with analytical posterior predictive distributions.
"""

from model_checker.model.blr.bayesian_linear_regression import (
    BayesianLinearRegressionND,
    create_blr_adapter,
)

__all__ = [
    "BayesianLinearRegressionND",
    "create_blr_adapter",
]
