"""
Tests for FlightDataProcessor
=============================

Unit tests for data loading, preprocessing, and transformation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.processor import FlightDataProcessor


class TestFlightDataProcessor:
    """Tests for FlightDataProcessor class."""

    def test_initialization(self):
        """Test processor initializes correctly."""
        processor = FlightDataProcessor()
        assert processor.fitted is False
        assert processor._encoder_mappings == {}

    def test_create_sample_data(self):
        """Test synthetic data generation."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=100, seed=42)

        assert len(df) == 100
        assert "origin" in df.columns
        assert "destination" in df.columns
        assert "airline" in df.columns
        assert "price" in df.columns
        assert "distance" in df.columns

    def test_create_sample_data_reproducibility(self):
        """Test that same seed produces same data."""
        processor = FlightDataProcessor()

        df1 = processor.create_sample_data(n_samples=50, seed=123)
        df2 = processor.create_sample_data(n_samples=50, seed=123)

        pd.testing.assert_frame_equal(df1, df2)

    def test_create_sample_data_different_seeds(self):
        """Test that different seeds produce different data."""
        processor = FlightDataProcessor()

        df1 = processor.create_sample_data(n_samples=50, seed=1)
        df2 = processor.create_sample_data(n_samples=50, seed=2)

        assert not df1.equals(df2)

    def test_create_sample_data_valid_airports(self):
        """Test that generated data uses valid airport codes."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=100)

        valid_airports = set(FlightDataProcessor.AIRPORT_TIERS.keys())
        assert df["origin"].isin(valid_airports).all()
        assert df["destination"].isin(valid_airports).all()

    def test_create_sample_data_valid_airlines(self):
        """Test that generated data uses valid airline names."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=100)

        valid_airlines = set(FlightDataProcessor.AIRLINE_TIERS.keys())
        assert df["airline"].isin(valid_airlines).all()

    def test_create_sample_data_different_airports(self):
        """Test that origin and destination are never the same."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=200)

        assert (df["origin"] != df["destination"]).all()

    def test_create_sample_data_positive_prices(self):
        """Test that all prices are positive."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=100)

        assert (df["price"] >= 50).all()  # Minimum price is $50

    def test_preprocess_basic(self, sample_flight_data):
        """Test basic preprocessing."""
        processor = FlightDataProcessor()
        X, y = processor.preprocess(sample_flight_data, fit=True)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(sample_flight_data)
        assert len(y) == len(sample_flight_data)
        assert processor.fitted is True

    def test_preprocess_feature_count(self, sample_flight_data):
        """Test that preprocessing produces expected number of features."""
        processor = FlightDataProcessor()
        X, _ = processor.preprocess(sample_flight_data, fit=True)

        # Expected: origin_enc, dest_enc, airline_enc, distance, origin_tier, dest_tier, airline_tier
        expected_features = 7
        assert X.shape[1] == expected_features

    def test_preprocess_without_fitting_fails(self, sample_flight_data):
        """Test that preprocessing without fitting raises error."""
        processor = FlightDataProcessor()

        with pytest.raises(ValueError, match="not fitted"):
            processor.preprocess(sample_flight_data, fit=False)

    def test_preprocess_missing_columns_fails(self):
        """Test that missing required columns raises error."""
        processor = FlightDataProcessor()
        df = pd.DataFrame({"origin": ["JFK"], "destination": ["LAX"]})  # Missing airline

        with pytest.raises(ValueError, match="Missing required columns"):
            processor.preprocess(df, fit=True)

    def test_preprocess_without_price_column(self):
        """Test preprocessing data without price column."""
        processor = FlightDataProcessor()
        df = pd.DataFrame({
            "origin": ["JFK", "LAX"],
            "destination": ["LAX", "ORD"],
            "airline": ["Delta", "United"]
        })

        X, y = processor.preprocess(df, fit=True)

        assert X is not None
        assert y is None

    def test_preprocess_inference_mode(self, sample_flight_data):
        """Test preprocessing in inference mode (fit=False)."""
        processor = FlightDataProcessor()

        # First fit
        processor.preprocess(sample_flight_data, fit=True)

        # Then inference
        new_data = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"]
        })

        X, y = processor.preprocess(new_data, fit=False)
        assert X.shape[0] == 1

    def test_get_distance_known_route(self):
        """Test distance lookup for known routes."""
        processor = FlightDataProcessor()

        distance = processor._get_distance("JFK", "LAX")
        assert distance == 2475

    def test_get_distance_reverse_route(self):
        """Test distance lookup for reverse route."""
        processor = FlightDataProcessor()

        distance = processor._get_distance("LAX", "JFK")
        assert distance == 2475

    def test_get_feature_names(self):
        """Test feature names are returned correctly."""
        processor = FlightDataProcessor()
        names = processor.get_feature_names()

        assert "origin_encoded" in names
        assert "destination_encoded" in names
        assert "airline_encoded" in names
        assert "distance" in names

    def test_airport_tiers_values(self):
        """Test that airport tier values are in expected range."""
        for code, tier in FlightDataProcessor.AIRPORT_TIERS.items():
            assert 0.5 <= tier <= 1.5, f"Airport {code} has unexpected tier {tier}"

    def test_airline_tiers_values(self):
        """Test that airline tier values are in expected range."""
        for airline, tier in FlightDataProcessor.AIRLINE_TIERS.items():
            assert 0.5 <= tier <= 1.5, f"Airline {airline} has unexpected tier {tier}"


class TestDataProcessorEdgeCases:
    """Edge case tests for FlightDataProcessor."""

    def test_single_row_preprocessing(self):
        """Test preprocessing with single row."""
        processor = FlightDataProcessor()
        df = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"],
            "price": [300.0]
        })

        X, y = processor.preprocess(df, fit=True)

        assert X.shape == (1, 7)
        assert len(y) == 1

    def test_unknown_airport_handling(self):
        """Test handling of unknown airport codes."""
        processor = FlightDataProcessor()

        # First fit with known data
        df_train = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"],
            "price": [300.0]
        })
        processor.preprocess(df_train, fit=True)

        # Then test with unknown airport
        df_test = pd.DataFrame({
            "origin": ["ZZZ"],  # Unknown
            "destination": ["LAX"],
            "airline": ["Delta"]
        })

        X, _ = processor.preprocess(df_test, fit=False)
        # Should have -1 for unknown encoding
        assert X[0, 0] == -1

    def test_large_dataset_performance(self):
        """Test that large datasets can be processed."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=10000, seed=42)

        X, y = processor.preprocess(df, fit=True)

        assert X.shape[0] == 10000

    def test_dtype_consistency(self):
        """Test that output dtypes are consistent."""
        processor = FlightDataProcessor()
        df = processor.create_sample_data(n_samples=100)

        X, y = processor.preprocess(df, fit=True)

        assert X.dtype == np.float32
