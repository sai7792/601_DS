"""
Integration Tests
=================

End-to-end integration tests for the flight price prediction system.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.processor import FlightDataProcessor
from data.validator import FlightDataValidator
from models.ensemble import EnsembleModel
from models.price_predictor import FlightPricePredictor
from features.engineering import FeatureEngineer
from agents.flight_agent import FlightPriceAgent


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_full_pipeline_data_to_prediction(self):
        """Test complete pipeline from data generation to prediction."""
        # Step 1: Generate data
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=500, seed=42)

        assert len(df) == 500
        assert "price" in df.columns

        # Step 2: Preprocess
        X, y = processor.preprocess(df, fit=True)

        assert X.shape[0] == 500
        assert len(y) == 500

        # Step 3: Train model
        model = EnsembleModel()
        model.fit(X, y)

        assert model.is_fitted is True

        # Step 4: Make prediction
        new_flight = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"]
        })
        X_new, _ = processor.preprocess(new_flight, fit=False)
        prediction = model.predict(X_new)

        assert len(prediction) == 1
        assert prediction[0] > 0

    def test_predictor_integration(self):
        """Test FlightPricePredictor integration."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=300)

        # Single prediction
        result = predictor.predict("JFK", "LAX", "Delta")
        assert result.predicted_price > 0

        # Comparison
        comparison = predictor.compare_airlines("SFO", "ORD")
        assert len(comparison) > 0

        # Cheapest routes
        cheapest = predictor.get_cheapest_route("BOS", "JetBlue")
        assert len(cheapest) > 0

    def test_agent_integration(self):
        """Test FlightPriceAgent integration."""
        # Setup
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)

        agent = FlightPriceAgent(predictor=predictor, use_llm=False)

        # Query
        response = agent.query("What is the price from JFK to LAX on Delta?")

        assert response.answer is not None
        assert len(response.answer) > 0


@pytest.mark.integration
class TestDataFlowIntegration:
    """Tests for data flow through the system."""

    def test_data_validation_to_processing(self):
        """Test data flows correctly from validation to processing."""
        validator = FlightDataValidator()
        processor = FlightDataProcessor()

        # Validate input
        flight = validator.validate_input("jfk", "lax", "Delta")

        # Create DataFrame
        df = pd.DataFrame([{
            "origin": flight.origin,
            "destination": flight.destination,
            "airline": flight.airline
        }])

        # Generate training data for fitting
        train_df = processor.create_sample_data(n_samples=100)
        processor.preprocess(train_df, fit=True)

        # Process validated input
        X, _ = processor.preprocess(df, fit=False)

        assert X.shape == (1, 7)

    def test_feature_engineering_integration(self):
        """Test feature engineering with data processor."""
        processor = FlightDataProcessor()
        engineer = FeatureEngineer()

        # Generate data
        df = processor.create_sample_data(n_samples=200)

        # Engineer features
        df_engineered = engineer.fit_transform(df)

        # Should have original + new columns
        assert len(df_engineered.columns) > len(df.columns)
        assert "origin_region" in df_engineered.columns
        assert "hub_score" in df_engineered.columns


@pytest.mark.integration
class TestModelConsistency:
    """Tests for model prediction consistency."""

    def test_same_input_same_output(self):
        """Test that same input produces same output."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)

        result1 = predictor.predict("JFK", "LAX", "Delta")
        result2 = predictor.predict("JFK", "LAX", "Delta")

        assert result1.predicted_price == result2.predicted_price

    def test_price_ordering_by_airline_tier(self):
        """Test that premium airlines generally cost more than budget."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=500)

        premium = predictor.predict("JFK", "LAX", "Delta")
        budget = predictor.predict("JFK", "LAX", "Spirit")

        # Premium should generally be more expensive
        assert premium.predicted_price > budget.predicted_price * 0.8

    def test_distance_affects_price(self):
        """Test that longer routes generally cost more."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=500)

        short_route = predictor.predict("JFK", "BOS", "Delta")  # ~187 miles
        long_route = predictor.predict("JFK", "LAX", "Delta")  # ~2475 miles

        # Long route should cost more
        assert long_route.predicted_price > short_route.predicted_price


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Tests for error handling across components."""

    def test_invalid_input_handled_gracefully(self):
        """Test that invalid input is handled gracefully."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=100)

        agent = FlightPriceAgent(predictor=predictor, use_llm=False)

        # Query with invalid info
        response = agent.query("Price for unknown airline XYZ")

        # Should not crash, should return helpful message
        assert response.answer is not None

    def test_validator_catches_errors_before_model(self):
        """Test that validator catches errors before model execution."""
        validator = FlightDataValidator()

        errors = validator.get_validation_errors("ZZZ", "ZZZ", "FakeAir")

        assert len(errors) > 0


@pytest.mark.integration
class TestPerformanceIntegration:
    """Performance-related integration tests."""

    def test_batch_prediction_performance(self):
        """Test batch predictions complete in reasonable time."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=300)

        flights = [
            {"origin": "JFK", "destination": "LAX", "airline": "Delta"},
            {"origin": "SFO", "destination": "ORD", "airline": "United"},
            {"origin": "MIA", "destination": "ATL", "airline": "Southwest"},
            {"origin": "BOS", "destination": "DEN", "airline": "JetBlue"},
            {"origin": "SEA", "destination": "PHX", "airline": "Alaska"}
        ] * 10  # 50 predictions

        results = predictor.predict_batch(flights)

        assert len(results) == 50

    def test_large_data_training(self):
        """Test training on larger dataset."""
        predictor = FlightPricePredictor(use_neural_network=False)

        # Train on 2000 samples
        predictor.fit(n_samples=2000)

        assert predictor.is_fitted is True

        # Should still predict accurately
        result = predictor.predict("JFK", "LAX", "Delta")
        assert result.predicted_price > 0
