"""
Feature Engineering
===================

Advanced feature engineering for flight price prediction.
Creates derived features from raw flight data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    include_interactions: bool = True
    include_route_stats: bool = True
    include_embeddings: bool = False
    polynomial_degree: int = 2


class FeatureEngineer:
    """
    Feature engineering pipeline for flight price prediction.

    Creates advanced features including:
    - Route-based features
    - Airport hub features
    - Airline category features
    - Interaction features
    - Route statistics
    """

    # Regional groupings
    REGIONS = {
        "Northeast": ["JFK", "BOS", "EWR", "PHL", "BWI"],
        "Southeast": ["ATL", "MIA", "CLT", "TPA", "MCO"],
        "Midwest": ["ORD", "DTW", "MSP", "STL", "IND", "MCI"],
        "Southwest": ["DFW", "IAH", "AUS", "PHX", "LAS"],
        "West": ["LAX", "SFO", "SEA", "PDX", "SAN", "DEN", "SLC"]
    }

    # Airport to region mapping
    AIRPORT_REGIONS = {}
    for region, airports in REGIONS.items():
        for airport in airports:
            AIRPORT_REGIONS[airport] = region

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._route_stats: Dict[str, Dict] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit the feature engineer on training data.

        Computes statistics needed for feature creation.

        Args:
            df: Training DataFrame with origin, destination, airline, price

        Returns:
            Self for method chaining
        """
        if "price" not in df.columns:
            raise ValueError("Training data must include 'price' column")

        # Compute route statistics
        if self.config.include_route_stats:
            self._compute_route_stats(df)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform flight data into engineered features.

        Args:
            df: DataFrame with origin, destination, airline

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Basic features
        df = self._add_region_features(df)
        df = self._add_route_type_features(df)
        df = self._add_hub_features(df)

        # Route statistics (if fitted)
        if self.config.include_route_stats and self._fitted:
            df = self._add_route_stats_features(df)

        # Interaction features
        if self.config.include_interactions:
            df = self._add_interaction_features(df)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _compute_route_stats(self, df: pd.DataFrame) -> None:
        """Compute statistics for each route."""
        # Route-level stats
        route_groups = df.groupby(["origin", "destination"])
        for (origin, dest), group in route_groups:
            route_key = f"{origin}_{dest}"
            self._route_stats[route_key] = {
                "mean_price": group["price"].mean(),
                "std_price": group["price"].std(),
                "min_price": group["price"].min(),
                "max_price": group["price"].max(),
                "count": len(group)
            }

        # Airline-route stats
        airline_route_groups = df.groupby(["origin", "destination", "airline"])
        for (origin, dest, airline), group in airline_route_groups:
            route_key = f"{origin}_{dest}_{airline}"
            self._route_stats[route_key] = {
                "mean_price": group["price"].mean(),
                "std_price": group["price"].std(),
                "count": len(group)
            }

    def _add_region_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regional features."""
        df["origin_region"] = df["origin"].map(self.AIRPORT_REGIONS).fillna("Other")
        df["dest_region"] = df["destination"].map(self.AIRPORT_REGIONS).fillna("Other")

        # Cross-region indicator
        df["is_cross_region"] = (df["origin_region"] != df["dest_region"]).astype(int)

        # Region pair (for encoding)
        df["region_pair"] = df["origin_region"] + "_to_" + df["dest_region"]

        return df

    def _add_route_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add route type features."""
        # Transcontinental (coast to coast)
        east_coast = {"JFK", "BOS", "EWR", "PHL", "MIA", "ATL"}
        west_coast = {"LAX", "SFO", "SEA", "PDX", "SAN"}

        df["is_transcontinental"] = (
            (df["origin"].isin(east_coast) & df["destination"].isin(west_coast)) |
            (df["origin"].isin(west_coast) & df["destination"].isin(east_coast))
        ).astype(int)

        # Hub-to-hub
        major_hubs = {"JFK", "LAX", "ORD", "ATL", "DFW"}
        df["is_hub_to_hub"] = (
            df["origin"].isin(major_hubs) & df["destination"].isin(major_hubs)
        ).astype(int)

        return df

    def _add_hub_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hub connectivity features."""
        # Count of connections (simplified as hub importance)
        hub_connections = {
            "ATL": 100, "ORD": 90, "DFW": 85, "DEN": 80, "LAX": 85,
            "JFK": 80, "SFO": 75, "SEA": 70, "MIA": 75, "BOS": 65,
            "PHX": 60, "IAH": 65, "LAS": 55, "MCO": 50, "EWR": 70
        }

        df["origin_connections"] = df["origin"].map(hub_connections).fillna(30)
        df["dest_connections"] = df["destination"].map(hub_connections).fillna(30)

        # Combined hub score
        df["hub_score"] = (df["origin_connections"] + df["dest_connections"]) / 2

        return df

    def _add_route_stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pre-computed route statistics as features."""
        route_means = []
        route_stds = []

        for _, row in df.iterrows():
            route_key = f"{row['origin']}_{row['destination']}"
            stats = self._route_stats.get(route_key, {})
            route_means.append(stats.get("mean_price", 0))
            route_stds.append(stats.get("std_price", 0))

        df["route_mean_price"] = route_means
        df["route_std_price"] = route_stds

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables."""
        # Hub score * transcontinental
        if "hub_score" in df.columns and "is_transcontinental" in df.columns:
            df["hub_transcon_interaction"] = df["hub_score"] * df["is_transcontinental"]

        # Origin tier * destination tier
        try:
            from ..data.processor import FlightDataProcessor
        except ImportError:
            from data.processor import FlightDataProcessor
        origin_tier = df["origin"].map(FlightDataProcessor.AIRPORT_TIERS).fillna(1.0)
        dest_tier = df["destination"].map(FlightDataProcessor.AIRPORT_TIERS).fillna(1.0)
        df["tier_interaction"] = origin_tier * dest_tier

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        features = [
            "origin_region", "dest_region", "is_cross_region", "region_pair",
            "is_transcontinental", "is_hub_to_hub",
            "origin_connections", "dest_connections", "hub_score"
        ]

        if self.config.include_route_stats:
            features.extend(["route_mean_price", "route_std_price"])

        if self.config.include_interactions:
            features.extend(["hub_transcon_interaction", "tier_interaction"])

        return features

    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names."""
        return ["origin_region", "dest_region", "region_pair"]

    def get_numerical_features(self) -> List[str]:
        """Get list of numerical feature names."""
        return [f for f in self.get_feature_names()
                if f not in self.get_categorical_features()]
