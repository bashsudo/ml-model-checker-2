"""
Posterior predictive distribution approximation.

Generalized bootstrap and posterior sampling for any UMLI model.
"""

from model_checker.analysis.posterior.bootstrap import (
    bootstrap_train_models,
    create_bootstrap_datasets,
)
from model_checker.analysis.posterior.posterior import (
    approximate_posterior_predictive,
    sample_posterior_predictive,
)

__all__ = [
    "bootstrap_train_models",
    "create_bootstrap_datasets",
    "approximate_posterior_predictive",
    "sample_posterior_predictive",
]
