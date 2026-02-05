"""
Tests for FlightPricePredictor
==============================

Integration tests for the main price prediction interface.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.price_predictor import FlightPricePredictor
from data.validator import FlightPrediction


class TestFlightPricePredictorInitialization:
    """Tests for FlightPricePredictor initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        predictor = FlightPricePredictor(use_neural_network=False)

        assert predictor.is_fitted is False
        assert predictor.processor is not None
        assert predictor.ensemble_model is not None

    def test_initialization_without_neural_network(self):
        """Test initialization without neural network."""
        predictor = FlightPricePredictor(use_neural_network=False)

        assert predictor.neural_model is None


class TestFlightPricePredictorFit:
    """Tests for FlightPricePredictor training."""

    def test_fit_with_synthetic_data(self):
        """Test fitting with synthetic data."""
        predictor = FlightPricePredictor(use_neural_network=False)

        predictor.fit(n_samples=200)

        assert predictor.is_fitted is True

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        predictor = FlightPricePredictor(use_neural_network=False)

        result = predictor.fit(n_samples=100)

        assert result is predictor

    def test_fit_produces_metrics(self):
        """Test that fitting produces training metrics."""
        predictor = FlightPricePredictor(use_neural_network=False)

        predictor.fit(n_samples=200)
        metrics = predictor.get_training_metrics()

        assert "ensemble" in metrics
        assert "mae" in metrics["ensemble"]


class TestFlightPricePredictorPredict:
    """Tests for FlightPricePredictor prediction."""

    @pytest.fixture
    def trained_predictor(self):
        """Provide a trained predictor."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=300)
        return predictor

    def test_predict_basic(self, trained_predictor):
        """Test basic prediction."""
        result = trained_predictor.predict("JFK", "LAX", "Delta")

        assert isinstance(result, FlightPrediction)
        assert result.origin == "JFK"
        assert result.destination == "LAX"
        assert result.airline == "Delta"
        assert result.predicted_price > 0

    def test_predict_returns_confidence(self, trained_predictor):
        """Test that prediction includes confidence."""
        result = trained_predictor.predict("JFK", "LAX", "Delta")

        assert 0 <= result.confidence <= 1

    def test_predict_returns_price_range(self, trained_predictor):
        """Test that prediction includes price range."""
        result = trained_predictor.predict("JFK", "LAX", "Delta")

        assert result.price_range[0] <= result.predicted_price
        assert result.price_range[1] >= result.predicted_price

    def test_predict_lowercase_input(self, trained_predictor):
        """Test prediction with lowercase input."""
        result = trained_predictor.predict("jfk", "lax", "Delta")

        assert result.origin == "JFK"
        assert result.destination == "LAX"

    def test_predict_before_fit_fails(self):
        """Test that prediction before fitting fails."""
        predictor = FlightPricePredictor(use_neural_network=False)

        with pytest.raises(ValueError, match="not fitted"):
            predictor.predict("JFK", "LAX", "Delta")

    def test_predict_invalid_same_airports_fails(self, trained_predictor):
        """Test that same origin and destination fails."""
        with pytest.raises(ValueError):
            trained_predictor.predict("JFK", "JFK", "Delta")


class TestFlightPricePredictorBatchPrediction:
    """Tests for batch prediction."""

    @pytest.fixture
    def trained_predictor(self):
        """Provide a trained predictor."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)
        return predictor

    def test_predict_batch(self, trained_predictor):
        """Test batch prediction."""
        flights = [
            {"origin": "JFK", "destination": "LAX", "airline": "Delta"},
            {"origin": "SFO", "destination": "ORD", "airline": "United"},
            {"origin": "MIA", "destination": "ATL", "airline": "Southwest"}
        ]

        results = trained_predictor.predict_batch(flights)

        assert len(results) == 3
        assert all(isinstance(r, FlightPrediction) for r in results)

    def test_predict_batch_empty(self, trained_predictor):
        """Test batch prediction with empty list."""
        results = trained_predictor.predict_batch([])

        assert len(results) == 0


class TestFlightPricePredictorComparison:
    """Tests for comparison methods."""

    @pytest.fixture
    def trained_predictor(self):
        """Provide a trained predictor."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)
        return predictor

    def test_compare_airlines(self, trained_predictor):
        """Test airline comparison."""
        results = trained_predictor.compare_airlines("JFK", "LAX")

        assert len(results) > 0
        # Results should be sorted by price
        prices = [r.predicted_price for r in results]
        assert prices == sorted(prices)

    def test_get_cheapest_route(self, trained_predictor):
        """Test finding cheapest routes."""
        results = trained_predictor.get_cheapest_route("JFK", "Delta")

        assert len(results) > 0
        # Results should be sorted by price
        prices = [r.predicted_price for r in results]
        assert prices == sorted(prices)

    def test_compare_airlines_different_prices(self, trained_predictor):
        """Test that different airlines have different prices."""
        results = trained_predictor.compare_airlines("JFK", "LAX")

        if len(results) > 1:
            prices = [r.predicted_price for r in results]
            # Prices should not all be identical
            assert len(set(prices)) > 1


class TestFlightPricePredictorPersistence:
    """Tests for model saving and loading."""

    @pytest.fixture
    def trained_predictor(self):
        """Provide a trained predictor."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)
        return predictor

    def test_save_model(self, trained_predictor):
        """Test model saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            trained_predictor.save(path)

            assert os.path.exists(path)

    def test_load_model(self, trained_predictor):
        """Test model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            trained_predictor.save(path)

            loaded = FlightPricePredictor.load(path)

            assert loaded.is_fitted is True

    def test_loaded_model_predicts(self, trained_predictor):
        """Test that loaded model can predict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            trained_predictor.save(path)

            loaded = FlightPricePredictor.load(path)
            result = loaded.predict("JFK", "LAX", "Delta")

            assert result.predicted_price > 0

    def test_loaded_model_same_predictions(self, trained_predictor):
        """Test that loaded model gives same predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")

            original_pred = trained_predictor.predict("JFK", "LAX", "Delta")
            trained_predictor.save(path)

            loaded = FlightPricePredictor.load(path)
            loaded_pred = loaded.predict("JFK", "LAX", "Delta")

            assert abs(original_pred.predicted_price - loaded_pred.predicted_price) < 0.01

    def test_save_unfitted_fails(self):
        """Test that saving unfitted model fails."""
        predictor = FlightPricePredictor(use_neural_network=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")

            with pytest.raises(ValueError, match="unfitted"):
                predictor.save(path)


class TestFlightPricePredictorMetrics:
    """Tests for training metrics."""

    def test_get_training_metrics_after_fit(self):
        """Test getting training metrics."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)

        metrics = predictor.get_training_metrics()

        assert "ensemble" in metrics
        assert metrics["ensemble"]["mae"] > 0
        assert metrics["ensemble"]["r2"] <= 1

    def test_get_training_metrics_before_fit(self):
        """Test getting metrics before fitting."""
        predictor = FlightPricePredictor(use_neural_network=False)

        metrics = predictor.get_training_metrics()

        assert metrics == {}
