# Model Checker

A Python framework for Bayesian machine learning models with uncertainty quantification and distribution analysis tools. The project provides a unified interface for working with various ML backends and includes tools for analyzing model predictions using probabilistic methods.

## Overview

Model Checker is designed to help researchers and practitioners understand model uncertainty and validate predictions through probabilistic analysis. The framework includes:

- **Unified ML Interface (UMLI)**: A backend-agnostic abstraction layer that supports PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, and HuggingFace models
- **Bayesian Models**: Implementations of Bayesian Neural Networks (BNN) and Bayesian Linear Regression (BLR) with uncertainty quantification
- **Distribution Analysis**: Tools for analyzing predictions using Dirichlet distributions and bootstrap methods
- **Prototype Pipeline**: Complete workflows for training models and performing probabilistic analysis

The implementation focuses on understanding how model predictions relate to underlying probability distributions, which is useful for model validation, uncertainty quantification, and assessing prediction reliability.

## Installation

```bash
# Install from source
pip install .

# Install with PyTorch support (required for BNN)
pip install .[pytorch]

# Install with all optional dependencies
pip install .[all]
```

## Key Components

### Unified ML Interface (UMLI)

UMLI provides a consistent interface across different ML frameworks, allowing you to work with models from various backends through a unified API. This abstraction enables:

- Backend-agnostic model training and inference
- Standardized data handling across frameworks
- Flexible configuration management
- Cross-framework model analysis

### Model Implementations

- **Bayesian Neural Network (BNN)**: A neural network with probabilistic weights trained using Bayes by Backprop, providing uncertainty estimates for classification tasks (e.g., MNIST digit classification)

- **Bayesian Linear Regression (BLR)**: A linear regression model with probabilistic weights that provides analytical posterior predictive distributions

### Analysis Tools

- **Dirichlet Distribution Analysis**: Tools for modeling and analyzing probability distributions, particularly useful for categorical predictions
- **Bootstrap Methods**: Approximating posterior predictive distributions by training multiple models on resampled datasets
- **Posterior Predictive Sampling**: Generating predictions from collections of trained models with uncertainty quantification

## Usage

### Prototype Pipeline

The prototype pipeline demonstrates training a Bayesian Neural Network on MNIST and analyzing predictions using Dirichlet distributions:

```bash
# Train a BNN and analyze predictions
python -m model_checker.prototype_pipeline.main --train --digits 0 1 2 3

# Load a pre-trained model and analyze specific digits
python -m model_checker.prototype_pipeline.main --digits 5 6 7 --lambda-concentration 10.0 --k-std 3.0
```

### Common Options

- `--train`: Train a new model (otherwise load existing)
- `--digits`: Digits to analyze (default: all 0-9)
- `--lambda-concentration`: Concentration parameter for Dirichlet (default: 10.0)
- `--k-std`: Number of standard deviations for threshold (default: 3.0)
- `--epochs`: Number of training epochs (default: 10)
- `--output-dir`: Directory to save results (default: ./experiment_results)

## Project Structure

- `model_checker/umli/`: Unified ML Interface implementation
- `model_checker/model/bnn/`: Bayesian Neural Network implementation
- `model_checker/model/blr/`: Bayesian Linear Regression implementation
- `model_checker/analysis/`: Analysis tools (bootstrap, posterior sampling)
- `model_checker/prototype_pipeline/`: Example pipeline for BNN training and analysis

## Visualization

The framework includes comprehensive visualization tools for:

- Dirichlet PDF plots showing probability distributions with standard deviation thresholds
- Model prediction analysis across different digit classes
- Uncertainty quantification visualizations

All plots are automatically saved to the specified output directory for further analysis.

