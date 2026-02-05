"""
Flight Price Predictor
======================

Main interface combining ensemble and neural network models
for robust flight price prediction.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import joblib
from pathlib import Path

try:
    from .ensemble import EnsembleModel
    from .neural_network import NeuralNetworkModel, HAS_TORCH
    from .base import BasePriceModel
    from ..data.processor import FlightDataProcessor
    from ..data.validator import FlightDataValidator, FlightPrediction
except ImportError:
    from models.ensemble import EnsembleModel
    from models.neural_network import NeuralNetworkModel, HAS_TORCH
    from models.base import BasePriceModel
    from data.processor import FlightDataProcessor
    from data.validator import FlightDataValidator, FlightPrediction


class FlightPricePredictor:
    """
    Main flight price prediction interface.

    Combines multiple models for robust predictions:
    - Ensemble of gradient boosting models
    - Neural network (if PyTorch available)
    - Automatic model selection based on confidence
    """

    def __init__(self, use_neural_network: bool = True):
        self.processor = FlightDataProcessor()
        self.validator = FlightDataValidator()

        self.ensemble_model = EnsembleModel(use_stacking=True)
        self.neural_model: Optional[NeuralNetworkModel] = None

        if use_neural_network and HAS_TORCH:
            self.neural_model = NeuralNetworkModel()

        self.is_fitted = False
        self._training_metrics: Dict[str, Any] = {}

    def fit(self, data_path: Optional[str] = None,
            n_samples: int = 5000) -> "FlightPricePredictor":
        """
        Train all models on flight data.

        Args:
            data_path: Path to CSV file with flight data
            n_samples: Number of samples if generating synthetic data

        Returns:
            Self for method chaining
        """
        # Load or generate data
        if data_path:
            df = self.processor.load_data(data_path)
        else:
            df = self.processor.create_sample_data(n_samples=n_samples)

        # Preprocess
        X, y = self.processor.preprocess(df, fit=True)

        # Split data
        n_train = int(len(X) * 0.8)
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        ensemble_metrics = self.ensemble_model.evaluate(X_test, y_test)

        self._training_metrics["ensemble"] = ensemble_metrics

        # Train neural network if available
        if self.neural_model is not None:
            self.neural_model.fit(X_train, y_train)
            nn_metrics = self.neural_model.evaluate(X_test, y_test)
            self._training_metrics["neural_network"] = nn_metrics

        self.is_fitted = True
        return self

    def predict(self, origin: str, destination: str,
                airline: str) -> FlightPrediction:
        """
        Predict flight price for a single route.

        Args:
            origin: Origin airport code (e.g., "JFK")
            destination: Destination airport code (e.g., "LAX")
            airline: Airline name (e.g., "Delta")

        Returns:
            FlightPrediction with price and confidence
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")

        # Validate inputs
        flight_input = self.validator.validate_input(origin, destination, airline)

        # Create DataFrame for prediction
        import pandas as pd
        df = pd.DataFrame([{
            "origin": flight_input.origin,
            "destination": flight_input.destination,
            "airline": flight_input.airline
        }])

        # Preprocess
        X, _ = self.processor.preprocess(df, fit=False)

        # Get predictions from all models
        ensemble_pred, ensemble_conf = self.ensemble_model.predict_with_confidence(X)

        model_used = "ensemble"
        final_pred = ensemble_pred[0]
        final_conf = ensemble_conf[0]

        # Use neural network if available and more confident
        if self.neural_model is not None:
            nn_pred, nn_conf = self.neural_model.predict_with_confidence(X)

            # Combine predictions weighted by confidence
            total_conf = ensemble_conf[0] + nn_conf[0]
            final_pred = (
                ensemble_pred[0] * ensemble_conf[0] +
                nn_pred[0] * nn_conf[0]
            ) / total_conf
            final_conf = (ensemble_conf[0] + nn_conf[0]) / 2
            model_used = "ensemble+neural"

        # Calculate price range
        uncertainty = (1 - final_conf) * 0.2  # 20% max range
        price_min = final_pred * (1 - uncertainty)
        price_max = final_pred * (1 + uncertainty)

        return FlightPrediction(
            origin=flight_input.origin,
            destination=flight_input.destination,
            airline=flight_input.airline,
            predicted_price=round(final_pred, 2),
            confidence=round(final_conf, 3),
            price_range=(round(price_min, 2), round(price_max, 2)),
            model_used=model_used
        )

    def predict_batch(self, flights: List[Dict[str, str]]) -> List[FlightPrediction]:
        """
        Predict prices for multiple flights.

        Args:
            flights: List of dicts with origin, destination, airline

        Returns:
            List of FlightPrediction objects
        """
        return [
            self.predict(f["origin"], f["destination"], f["airline"])
            for f in flights
        ]

    def compare_airlines(self, origin: str, destination: str) -> List[FlightPrediction]:
        """
        Compare prices across all airlines for a route.

        Args:
            origin: Origin airport code
            destination: Destination airport code

        Returns:
            List of predictions sorted by price
        """
        predictions = []

        for airline in self.validator.VALID_AIRLINES:
            try:
                pred = self.predict(origin, destination, airline)
                predictions.append(pred)
            except Exception:
                continue

        return sorted(predictions, key=lambda p: p.predicted_price)

    def get_cheapest_route(self, origin: str, airline: str) -> List[FlightPrediction]:
        """
        Find cheapest destinations from an origin with a given airline.

        Args:
            origin: Origin airport code
            airline: Airline name

        Returns:
            List of predictions sorted by price
        """
        predictions = []

        for dest in self.validator.VALID_AIRPORTS:
            if dest == origin.upper():
                continue
            try:
                pred = self.predict(origin, dest, airline)
                predictions.append(pred)
            except Exception:
                continue

        return sorted(predictions, key=lambda p: p.predicted_price)

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get metrics from the training phase."""
        return self._training_metrics.copy()

    def save(self, path: str) -> None:
        """Save the trained predictor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted predictor")

        save_dict = {
            "processor": self.processor,
            "ensemble_model": self.ensemble_model,
            "neural_model": self.neural_model,
            "training_metrics": self._training_metrics
        }

        joblib.dump(save_dict, path)

    @classmethod
    def load(cls, path: str) -> "FlightPricePredictor":
        """Load a trained predictor from disk."""
        save_dict = joblib.load(path)

        predictor = cls(use_neural_network=False)
        predictor.processor = save_dict["processor"]
        predictor.ensemble_model = save_dict["ensemble_model"]
        predictor.neural_model = save_dict["neural_model"]
        predictor._training_metrics = save_dict["training_metrics"]
        predictor.is_fitted = True

        return predictor
