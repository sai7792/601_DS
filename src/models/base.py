"""
Base Model Interface
====================

Abstract base class for all prediction models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BasePriceModel(ABC):
    """Abstract base class for flight price prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self._metrics: Dict[str, float] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasePriceModel":
        """
        Train the model on flight data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target prices (n_samples,)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict flight prices.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted prices (n_samples,)
        """
        pass

    @abstractmethod
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict prices with confidence intervals.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True prices

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)

        mae = np.mean(np.abs(predictions - y))
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y - predictions) / y)) * 100

        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self._metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        }

        return self._metrics

    @property
    def metrics(self) -> Dict[str, float]:
        """Get the latest evaluation metrics."""
        return self._metrics.copy()
