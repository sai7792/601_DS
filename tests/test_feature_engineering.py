"""
Tests for Feature Engineering
=============================

Unit tests for the feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.engineering import FeatureEngineer, FeatureConfig


class TestFeatureEngineerInitialization:
    """Tests for FeatureEngineer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        engineer = FeatureEngineer()

        assert engineer.config.include_interactions is True
        assert engineer.config.include_route_stats is True
        assert engineer._fitted is False

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = FeatureConfig(
            include_interactions=False,
            include_route_stats=False
        )
        engineer = FeatureEngineer(config)

        assert engineer.config.include_interactions is False
        assert engineer.config.include_route_stats is False

    def test_regions_defined(self):
        """Test that regions are properly defined."""
        assert "Northeast" in FeatureEngineer.REGIONS
        assert "West" in FeatureEngineer.REGIONS

        # Check specific airports
        assert "JFK" in FeatureEngineer.REGIONS["Northeast"]
        assert "LAX" in FeatureEngineer.REGIONS["West"]


class TestFeatureEngineerFit:
    """Tests for FeatureEngineer fit method."""

    @pytest.fixture
    def sample_data_with_price(self):
        """Sample data with prices for fitting."""
        return pd.DataFrame({
            "origin": ["JFK", "LAX", "ORD", "ATL", "JFK"],
            "destination": ["LAX", "ORD", "ATL", "MIA", "LAX"],
            "airline": ["Delta", "United", "American", "Delta", "Southwest"],
            "price": [350.0, 280.0, 200.0, 150.0, 275.0]
        })

    def test_fit_basic(self, sample_data_with_price):
        """Test basic fitting."""
        engineer = FeatureEngineer()
        result = engineer.fit(sample_data_with_price)

        assert engineer._fitted is True
        assert result is engineer  # Returns self

    def test_fit_without_price_fails(self):
        """Test that fitting without price column fails."""
        df = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"]
        })
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="price"):
            engineer.fit(df)

    def test_fit_computes_route_stats(self, sample_data_with_price):
        """Test that fitting computes route statistics."""
        config = FeatureConfig(include_route_stats=True)
        engineer = FeatureEngineer(config)

        engineer.fit(sample_data_with_price)

        assert len(engineer._route_stats) > 0
        assert "JFK_LAX" in engineer._route_stats


class TestFeatureEngineerTransform:
    """Tests for FeatureEngineer transform method."""

    @pytest.fixture
    def fitted_engineer(self):
        """Provide a fitted feature engineer."""
        df = pd.DataFrame({
            "origin": ["JFK", "LAX", "ORD", "ATL", "SFO"] * 10,
            "destination": ["LAX", "ORD", "ATL", "MIA", "JFK"] * 10,
            "airline": ["Delta", "United", "American", "Delta", "Alaska"] * 10,
            "price": [350.0, 280.0, 200.0, 150.0, 380.0] * 10
        })
        engineer = FeatureEngineer()
        engineer.fit(df)
        return engineer

    @pytest.fixture
    def test_data(self):
        """Test data for transformation."""
        return pd.DataFrame({
            "origin": ["JFK", "LAX", "ORD"],
            "destination": ["LAX", "ORD", "ATL"],
            "airline": ["Delta", "United", "American"]
        })

    def test_transform_adds_region_features(self, fitted_engineer, test_data):
        """Test that transform adds region features."""
        result = fitted_engineer.transform(test_data)

        assert "origin_region" in result.columns
        assert "dest_region" in result.columns
        assert "is_cross_region" in result.columns

    def test_transform_adds_route_type_features(self, fitted_engineer, test_data):
        """Test that transform adds route type features."""
        result = fitted_engineer.transform(test_data)

        assert "is_transcontinental" in result.columns
        assert "is_hub_to_hub" in result.columns

    def test_transform_adds_hub_features(self, fitted_engineer, test_data):
        """Test that transform adds hub features."""
        result = fitted_engineer.transform(test_data)

        assert "origin_connections" in result.columns
        assert "dest_connections" in result.columns
        assert "hub_score" in result.columns

    def test_transform_preserves_original_columns(self, fitted_engineer, test_data):
        """Test that original columns are preserved."""
        result = fitted_engineer.transform(test_data)

        assert "origin" in result.columns
        assert "destination" in result.columns
        assert "airline" in result.columns

    def test_transform_does_not_modify_input(self, fitted_engineer, test_data):
        """Test that transform doesn't modify input DataFrame."""
        original_columns = test_data.columns.tolist()
        fitted_engineer.transform(test_data)

        assert test_data.columns.tolist() == original_columns


class TestRegionFeatures:
    """Tests for region-based features."""

    @pytest.fixture
    def engineer(self):
        """Provide a feature engineer."""
        df = pd.DataFrame({
            "origin": ["JFK", "LAX"],
            "destination": ["LAX", "JFK"],
            "airline": ["Delta", "Delta"],
            "price": [350.0, 350.0]
        })
        engineer = FeatureEngineer()
        engineer.fit(df)
        return engineer

    def test_northeast_airports_classified(self, engineer):
        """Test Northeast airports are correctly classified."""
        df = pd.DataFrame({
            "origin": ["JFK", "BOS", "EWR"],
            "destination": ["LAX", "LAX", "LAX"],
            "airline": ["Delta", "Delta", "Delta"]
        })

        result = engineer.transform(df)

        assert (result["origin_region"] == "Northeast").all()

    def test_west_airports_classified(self, engineer):
        """Test West airports are correctly classified."""
        df = pd.DataFrame({
            "origin": ["LAX", "SFO", "SEA"],
            "destination": ["JFK", "JFK", "JFK"],
            "airline": ["Delta", "Delta", "Delta"]
        })

        result = engineer.transform(df)

        assert (result["origin_region"] == "West").all()

    def test_cross_region_detection(self, engineer):
        """Test cross-region flight detection."""
        df = pd.DataFrame({
            "origin": ["JFK", "LAX"],  # Northeast, West
            "destination": ["LAX", "SFO"],  # West, West
            "airline": ["Delta", "Delta"]
        })

        result = engineer.transform(df)

        assert result.iloc[0]["is_cross_region"] == 1  # JFK -> LAX
        assert result.iloc[1]["is_cross_region"] == 0  # LAX -> SFO


class TestRouteTypeFeatures:
    """Tests for route type features."""

    @pytest.fixture
    def engineer(self):
        """Provide a feature engineer."""
        df = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"],
            "price": [350.0]
        })
        engineer = FeatureEngineer()
        engineer.fit(df)
        return engineer

    def test_transcontinental_detection(self, engineer):
        """Test transcontinental route detection."""
        df = pd.DataFrame({
            "origin": ["JFK", "LAX", "ORD"],
            "destination": ["LAX", "JFK", "DEN"],
            "airline": ["Delta", "Delta", "Delta"]
        })

        result = engineer.transform(df)

        assert result.iloc[0]["is_transcontinental"] == 1  # JFK -> LAX
        assert result.iloc[1]["is_transcontinental"] == 1  # LAX -> JFK
        assert result.iloc[2]["is_transcontinental"] == 0  # ORD -> DEN

    def test_hub_to_hub_detection(self, engineer):
        """Test hub-to-hub route detection."""
        df = pd.DataFrame({
            "origin": ["JFK", "JFK", "BOS"],
            "destination": ["LAX", "BOS", "PHL"],
            "airline": ["Delta", "Delta", "Delta"]
        })

        result = engineer.transform(df)

        assert result.iloc[0]["is_hub_to_hub"] == 1  # JFK -> LAX (both major)
        assert result.iloc[1]["is_hub_to_hub"] == 0  # JFK -> BOS (BOS not major)
        assert result.iloc[2]["is_hub_to_hub"] == 0  # BOS -> PHL


class TestInteractionFeatures:
    """Tests for interaction features."""

    def test_interaction_features_created(self):
        """Test that interaction features are created."""
        df = pd.DataFrame({
            "origin": ["JFK", "LAX"],
            "destination": ["LAX", "JFK"],
            "airline": ["Delta", "Delta"],
            "price": [350.0, 350.0]
        })

        config = FeatureConfig(include_interactions=True)
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(df)

        assert "hub_transcon_interaction" in result.columns
        assert "tier_interaction" in result.columns

    def test_interaction_features_not_created_when_disabled(self):
        """Test that interaction features aren't created when disabled."""
        df = pd.DataFrame({
            "origin": ["JFK"],
            "destination": ["LAX"],
            "airline": ["Delta"],
            "price": [350.0]
        })

        config = FeatureConfig(include_interactions=False)
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(df)

        assert "hub_transcon_interaction" not in result.columns


class TestFeatureNames:
    """Tests for feature name retrieval."""

    def test_get_feature_names(self):
        """Test feature names list."""
        engineer = FeatureEngineer()
        names = engineer.get_feature_names()

        assert len(names) > 0
        assert "origin_region" in names
        assert "hub_score" in names

    def test_get_categorical_features(self):
        """Test categorical feature names."""
        engineer = FeatureEngineer()
        categorical = engineer.get_categorical_features()

        assert "origin_region" in categorical
        assert "dest_region" in categorical

    def test_get_numerical_features(self):
        """Test numerical feature names."""
        engineer = FeatureEngineer()
        numerical = engineer.get_numerical_features()

        assert "hub_score" in numerical
        assert "is_cross_region" in numerical
        assert "origin_region" not in numerical
