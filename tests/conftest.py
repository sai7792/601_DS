"""
Pytest Configuration and Fixtures
=================================

Shared fixtures for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_flight_data() -> pd.DataFrame:
    """Generate sample flight data for testing."""
    np.random.seed(42)

    data = [
        {"origin": "JFK", "destination": "LAX", "airline": "Delta", "price": 350.0},
        {"origin": "JFK", "destination": "LAX", "airline": "United", "price": 325.0},
        {"origin": "JFK", "destination": "LAX", "airline": "Southwest", "price": 275.0},
        {"origin": "LAX", "destination": "ORD", "airline": "American", "price": 280.0},
        {"origin": "LAX", "destination": "ORD", "airline": "Spirit", "price": 150.0},
        {"origin": "SFO", "destination": "JFK", "airline": "Delta", "price": 380.0},
        {"origin": "SFO", "destination": "JFK", "airline": "JetBlue", "price": 320.0},
        {"origin": "ATL", "destination": "MIA", "airline": "Delta", "price": 180.0},
        {"origin": "ATL", "destination": "MIA", "airline": "Frontier", "price": 95.0},
        {"origin": "ORD", "destination": "DEN", "airline": "United", "price": 220.0},
        {"origin": "BOS", "destination": "SFO", "airline": "Alaska", "price": 340.0},
        {"origin": "SEA", "destination": "LAX", "airline": "Alaska", "price": 185.0},
        {"origin": "DFW", "destination": "JFK", "airline": "American", "price": 290.0},
        {"origin": "MIA", "destination": "ORD", "airline": "Spirit", "price": 125.0},
        {"origin": "DEN", "destination": "SEA", "airline": "Southwest", "price": 165.0},
    ]

    return pd.DataFrame(data)


@pytest.fixture
def large_sample_data() -> pd.DataFrame:
    """Generate larger sample data for model training."""
    from data.processor import FlightDataProcessor

    processor = FlightDataProcessor()
    return processor.create_sample_data(n_samples=500, seed=42)


@pytest.fixture
def sample_prediction_input() -> Dict[str, str]:
    """Sample input for price prediction."""
    return {
        "origin": "JFK",
        "destination": "LAX",
        "airline": "Delta"
    }


@pytest.fixture
def invalid_prediction_inputs() -> list:
    """Invalid inputs for testing error handling."""
    return [
        {"origin": "JFK", "destination": "JFK", "airline": "Delta"},  # Same airport
        {"origin": "XXX", "destination": "LAX", "airline": "Delta"},  # Invalid origin
        {"origin": "JFK", "destination": "LAX", "airline": "FakeAir"},  # Invalid airline
        {"origin": "", "destination": "LAX", "airline": "Delta"},  # Empty origin
    ]


@pytest.fixture
def trained_processor():
    """Provide a fitted data processor."""
    from data.processor import FlightDataProcessor

    processor = FlightDataProcessor()
    df = processor.create_sample_data(n_samples=100, seed=42)
    processor.preprocess(df, fit=True)
    return processor


@pytest.fixture
def trained_ensemble_model(large_sample_data):
    """Provide a trained ensemble model."""
    from data.processor import FlightDataProcessor
    from models.ensemble import EnsembleModel

    processor = FlightDataProcessor()
    X, y = processor.preprocess(large_sample_data, fit=True)

    model = EnsembleModel(use_stacking=True)
    model.fit(X, y)

    return model, processor


@pytest.fixture
def feature_engineer():
    """Provide a feature engineer instance."""
    from features.engineering import FeatureEngineer, FeatureConfig

    config = FeatureConfig(
        include_interactions=True,
        include_route_stats=True
    )
    return FeatureEngineer(config)


@pytest.fixture
def flight_agent(trained_ensemble_model):
    """Provide a configured flight agent."""
    from models.price_predictor import FlightPricePredictor
    from agents.flight_agent import FlightPriceAgent

    # Create and train predictor
    predictor = FlightPricePredictor(use_neural_network=False)
    predictor.fit(n_samples=200)

    return FlightPriceAgent(predictor=predictor, use_llm=False)
