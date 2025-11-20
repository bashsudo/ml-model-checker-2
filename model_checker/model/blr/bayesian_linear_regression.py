"""
Bayesian Linear Regression for N-dimensional inputs.

Implements analytical Bayesian linear regression with closed-form
posterior predictive distributions.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from model_checker.umli.model.unified_model import UnifiedModel


class BayesianLinearRegressionND:
    """
    Bayesian Linear Regression for N-dimensional inputs.

    Provides both point predictions and uncertainty estimates through
    analytical posterior predictive distributions.
    """

    def __init__(self, alpha: float = 1e-6, beta: float = 25.0):
        """
        Initialize Bayesian Linear Regression model.

        Args:
            alpha: Prior precision (inverse variance of prior on weights)
            beta: Noise precision (inverse variance of observation noise)
        """
        self.alpha = alpha
        self.beta = beta
        self.m_N = None  # posterior mean of weights (including bias)
        self.S_N = None  # posterior covariance of weights
        self._is_fitted = False

    def _coerce_X_for_fit(self, X: np.ndarray) -> np.ndarray:
        """Coerce X for training: 1D -> (n,1), 2D stays (n,d)."""
        X = np.asarray(X)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _coerce_X_for_predict(self, X: np.ndarray) -> np.ndarray:
        """Coerce X for prediction: 1D -> (1,d) treated as single feature vector."""
        X = np.asarray(X)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            # Decide whether this is a single d-dim sample or n samples of 1 feature
            feature_dim = (self.m_N.shape[0] - 1) if self.m_N is not None else None
            if feature_dim is not None and X.shape[0] == feature_dim:
                # Single sample with d features
                X = X.reshape(1, -1)
            else:
                # Multiple samples with 1 feature each
                X = X.reshape(-1, 1)
        return X

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Linear features with bias term: phi(x) = [1, x1, x2, ..., xd]."""
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Bayesian linear regression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        X = self._coerce_X_for_fit(X)
        Phi = self._design_matrix(X)
        y_vec = np.asarray(y).reshape(-1)  # shape (n,)

        # Prior covariance inverse: S_0^{-1} = alpha * I
        S_0_inv = self.alpha * np.eye(Phi.shape[1])

        # Posterior covariance inverse: S_N^{-1} = S_0^{-1} + beta * Phi^T Phi
        S_N_inv = S_0_inv + self.beta * (Phi.T @ Phi)

        # Posterior covariance: S_N = (S_N^{-1})^{-1}
        self.S_N = np.linalg.inv(S_N_inv)

        # Posterior mean: m_N = beta * S_N * Phi^T * y
        self.m_N = self.beta * (self.S_N @ (Phi.T @ y_vec))

        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions (mean of predictive distribution).

        Args:
            X: Input features of shape (n_samples, n_features) or (n_features,)

        Returns:
            Predicted mean values of shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_star = self._coerce_X_for_predict(X)
        Phi_star = self._design_matrix(X_star)
        mean = Phi_star @ self.m_N
        return mean.flatten()

    def predictive_distribution(
        self, X: np.ndarray, n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return predictive mean and variance at new inputs X.

        Args:
            X: Input features of shape (n_samples, n_features) or (n_features,)
            n_samples: Ignored for analytical models (always returns mean/variance)

        Returns:
            Tuple of (mean, variance) arrays, each of shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_star = self._coerce_X_for_predict(X)
        Phi_star = self._design_matrix(X_star)
        mean = Phi_star @ self.m_N

        # Variance for each row i: beta^{-1} + phi_i^T S_N phi_i
        var = (1.0 / self.beta) + np.sum((Phi_star @ self.S_N) * Phi_star, axis=1)

        return mean.flatten(), var.flatten()

    def predictive_logpdf(self, X_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        """
        Return log probability of observing y_star at X_star.

        Args:
            X_star: Input features
            y_star: Observed target values

        Returns:
            Log probability density values
        """
        mean, var = self.predictive_distribution(X_star)
        return norm.logpdf(y_star, loc=mean, scale=np.sqrt(var))

    def get_parameters(self) -> dict:
        """
        Get model parameters as a dictionary.

        Returns:
            Dictionary containing 'm_N' (posterior mean) and 'S_N' (posterior covariance)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        if self.m_N is None or self.S_N is None:
            raise ValueError("Model parameters not initialized")
        return {
            "m_N": self.m_N.copy(),
            "S_N": self.S_N.copy(),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def set_parameters(self, params: dict) -> None:
        """
        Set model parameters from a dictionary.

        Args:
            params: Dictionary containing 'm_N', 'S_N', 'alpha', 'beta'
        """
        if "m_N" not in params or "S_N" not in params:
            raise ValueError("params must contain 'm_N' and 'S_N'")

        self.m_N = np.asarray(params["m_N"]).copy()
        self.S_N = np.asarray(params["S_N"]).copy()

        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "beta" in params:
            self.beta = float(params["beta"])

        # Mark as fitted if parameters are set
        if self.m_N is not None and self.S_N is not None:
            self._is_fitted = True

    def infer_noise_distribution(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Infer noise distribution from training data residuals.

        For linear regression, estimates noise variance from residuals:
        sigma^2 = (1/n) * sum((y - y_pred)^2)

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with 'variance' key
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before inferring noise")

        y_pred = self.predict(X)
        residuals = y - y_pred
        variance = np.mean(residuals**2)

        return {"variance": float(variance)}

    def sample_noise(
        self,
        size: int,
        noise_params: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample noise from N(0, sigma^2) where sigma^2 is inferred from data.

        Args:
            size: Number of noise samples
            noise_params: Dictionary with 'variance' key. If None, uses self.beta.
            rng: Optional random number generator

        Returns:
            Array of noise samples
        """
        if rng is None:
            rng = np.random.default_rng()

        if noise_params is None:
            # Use beta (noise precision) if available
            if hasattr(self, "beta") and self.beta is not None:
                variance = 1.0 / self.beta
            else:
                raise ValueError(
                    "noise_params must be provided or model must have beta"
                )
        else:
            variance = noise_params.get("variance")
            if variance is None:
                raise ValueError("noise_params must contain 'variance'")

        return rng.normal(0.0, np.sqrt(variance), size=size)

    def predict_with_noise(
        self,
        X: np.ndarray,
        noise_params: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Make predictions with added noise: y = f(X) + epsilon.

        Args:
            X: Input features
            noise_params: Optional noise parameters dict. If None, uses inferred parameters.
            rng: Optional random number generator for reproducibility

        Returns:
            Predictions with noise added
        """
        if rng is None:
            rng = np.random.default_rng()

        predictions = self.predict(X)
        noise = self.sample_noise(len(predictions), noise_params, rng)
        return predictions + noise

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted


def create_blr_adapter(alpha: float = 1e-6, beta: float = 25.0) -> UnifiedModel:
    """
    Create a UMLI adapter for BayesianLinearRegressionND.

    Args:
        alpha: Prior precision
        beta: Noise precision

    Returns:
        UnifiedModel adapter wrapping BayesianLinearRegressionND
    """
    blr = BayesianLinearRegressionND(alpha=alpha, beta=beta)

    def forward_fn(batch):
        """Forward pass (prediction)."""
        return blr.predict(batch)

    def predict_fn(batch):
        """Predict function."""
        return blr.predict(batch)

    def fit_fn(dataset):
        """Fit function - handles UMLI datasets."""
        if hasattr(dataset, "to_numpy"):
            X, y = dataset.to_numpy()
            blr.fit(X, y)
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
            blr.fit(X, y)
        else:
            raise ValueError("Dataset must be (X, y) tuple or UnifiedDataset")

    def save_fn(path: str) -> None:
        """Save model parameters."""
        import pickle

        params = blr.get_parameters()
        with open(path, "wb") as f:
            pickle.dump(params, f)

    def load_fn(path: str) -> None:
        """Load model parameters."""
        import pickle

        with open(path, "rb") as f:
            params = pickle.load(f)
        blr.set_parameters(params)

    from model_checker.umli.model.adapters import CustomModelAdapter

    return CustomModelAdapter(
        backend_model=blr,
        forward_fn=forward_fn,
        predict_fn=predict_fn,
        fit_fn=fit_fn,
        save_fn=save_fn,
        load_fn=load_fn,
        config={"alpha": alpha, "beta": beta},
    )
