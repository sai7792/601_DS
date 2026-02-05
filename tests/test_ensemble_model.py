"""
Tests for Ensemble Model
========================

Unit tests for the ensemble machine learning model.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ensemble import EnsembleModel, HAS_XGBOOST, HAS_LIGHTGBM, HAS_CATBOOST
from data.processor import FlightDataProcessor


class TestEnsembleModelInitialization:
    """Tests for EnsembleModel initialization."""

    def test_basic_initialization(self):
        """Test basic model initialization."""
        model = EnsembleModel()

        assert model.name == "EnsembleModel"
        assert model.is_fitted is False
        assert len(model.models) > 0

    def test_initialization_with_stacking(self):
        """Test initialization with stacking enabled."""
        model = EnsembleModel(use_stacking=True)
        assert model.use_stacking is True

    def test_initialization_without_stacking(self):
        """Test initialization with stacking disabled."""
        model = EnsembleModel(use_stacking=False)
        assert model.use_stacking is False

    def test_base_models_available(self):
        """Test that base sklearn models are always available."""
        model = EnsembleModel()

        assert "gradient_boosting" in model.models
        assert "random_forest" in model.models
        assert "adaboost" in model.models

    def test_initial_weights_uniform(self):
        """Test that initial weights are uniform."""
        model = EnsembleModel()
        n_models = len(model.models)

        expected_weight = 1.0 / n_models
        for weight in model.weights.values():
            assert abs(weight - expected_weight) < 1e-6


class TestEnsembleModelTraining:
    """Tests for EnsembleModel training."""

    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=200, seed=42)
        X, y = processor.preprocess(df, fit=True)
        return X, y

    def test_fit_basic(self, training_data):
        """Test basic model fitting."""
        X, y = training_data
        model = EnsembleModel()

        model.fit(X, y)

        assert model.is_fitted is True

    def test_fit_returns_self(self, training_data):
        """Test that fit returns self for chaining."""
        X, y = training_data
        model = EnsembleModel()

        result = model.fit(X, y)

        assert result is model

    def test_fit_updates_weights(self, training_data):
        """Test that fitting updates model weights."""
        X, y = training_data
        model = EnsembleModel()

        initial_weights = model.weights.copy()
        model.fit(X, y)

        # Weights should be updated (unlikely to remain exactly uniform)
        weights_changed = any(
            abs(model.weights[k] - initial_weights[k]) > 1e-10
            for k in model.weights
        )
        assert weights_changed

    def test_fit_with_stacking_creates_meta_model(self, training_data):
        """Test that stacking creates meta model."""
        X, y = training_data
        model = EnsembleModel(use_stacking=True)

        model.fit(X, y)

        assert model.meta_model is not None


class TestEnsembleModelPrediction:
    """Tests for EnsembleModel prediction."""

    @pytest.fixture
    def fitted_model(self):
        """Provide a fitted model."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=200, seed=42)
        X, y = processor.preprocess(df, fit=True)

        model = EnsembleModel(use_stacking=True)
        model.fit(X, y)

        return model, X, y

    def test_predict_basic(self, fitted_model):
        """Test basic prediction."""
        model, X, y = fitted_model

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_predict_returns_positive_values(self, fitted_model):
        """Test that predictions are reasonable (positive prices)."""
        model, X, y = fitted_model

        predictions = model.predict(X)

        # Most predictions should be positive
        assert (predictions > 0).mean() > 0.95

    def test_predict_before_fit_fails(self):
        """Test that predicting before fitting raises error."""
        model = EnsembleModel()
        X = np.random.randn(10, 7)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_predict_with_confidence(self, fitted_model):
        """Test prediction with confidence scores."""
        model, X, y = fitted_model

        predictions, confidence = model.predict_with_confidence(X)

        assert len(predictions) == len(X)
        assert len(confidence) == len(X)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_confidence_varies_by_sample(self, fitted_model):
        """Test that confidence varies across samples."""
        model, X, y = fitted_model

        _, confidence = model.predict_with_confidence(X)

        # Confidence should not be identical for all samples
        assert confidence.std() > 0


class TestEnsembleModelEvaluation:
    """Tests for EnsembleModel evaluation."""

    @pytest.fixture
    def fitted_model_with_test_data(self):
        """Provide fitted model with train/test split."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=300, seed=42)
        X, y = processor.preprocess(df, fit=True)

        # Split
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        model = EnsembleModel()
        model.fit(X_train, y_train)

        return model, X_test, y_test

    def test_evaluate_returns_metrics(self, fitted_model_with_test_data):
        """Test that evaluation returns all expected metrics."""
        model, X_test, y_test = fitted_model_with_test_data

        metrics = model.evaluate(X_test, y_test)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics

    def test_metrics_are_reasonable(self, fitted_model_with_test_data):
        """Test that metrics are in reasonable ranges."""
        model, X_test, y_test = fitted_model_with_test_data

        metrics = model.evaluate(X_test, y_test)

        assert metrics["mae"] >= 0
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["r2"] <= 1

    def test_rmse_equals_sqrt_mse(self, fitted_model_with_test_data):
        """Test that RMSE = sqrt(MSE)."""
        model, X_test, y_test = fitted_model_with_test_data

        metrics = model.evaluate(X_test, y_test)

        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-6

    def test_metrics_property(self, fitted_model_with_test_data):
        """Test that metrics property returns last evaluation."""
        model, X_test, y_test = fitted_model_with_test_data

        model.evaluate(X_test, y_test)
        metrics = model.metrics

        assert len(metrics) > 0


class TestEnsembleModelFeatureImportance:
    """Tests for feature importance."""

    @pytest.fixture
    def fitted_model(self):
        """Provide a fitted model."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=200, seed=42)
        X, y = processor.preprocess(df, fit=True)

        model = EnsembleModel()
        model.fit(X, y)

        return model, X.shape[1]

    def test_get_feature_importance(self, fitted_model):
        """Test feature importance retrieval."""
        model, n_features = fitted_model

        importances = model.get_feature_importance()

        assert len(importances) > 0

    def test_feature_importance_sums_to_one(self, fitted_model):
        """Test that feature importances sum to 1 for each model."""
        model, n_features = fitted_model

        importances = model.get_feature_importance()

        for model_name, imp in importances.items():
            assert abs(imp.sum() - 1.0) < 0.01

    def test_feature_importance_length(self, fitted_model):
        """Test that importance length matches features."""
        model, n_features = fitted_model

        importances = model.get_feature_importance()

        for model_name, imp in importances.items():
            assert len(imp) == n_features

    def test_get_model_weights(self, fitted_model):
        """Test model weights retrieval."""
        model, _ = fitted_model

        weights = model.get_model_weights()

        assert len(weights) == len(model.models)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
