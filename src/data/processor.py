"""
Flight Data Processor
=====================

Handles loading, cleaning, and preprocessing of flight data for price prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path


class FlightDataProcessor:
    """Process flight data for price prediction models."""

    # Major airport hubs with base pricing factors
    AIRPORT_TIERS = {
        # Tier 1: Major International Hubs
        "JFK": 1.3, "LAX": 1.25, "ORD": 1.2, "ATL": 1.15, "DFW": 1.2,
        "SFO": 1.25, "MIA": 1.2, "SEA": 1.15, "BOS": 1.2, "DEN": 1.1,
        # Tier 2: Large Regional Airports
        "PHX": 1.05, "IAH": 1.1, "LAS": 1.05, "MCO": 1.0, "EWR": 1.2,
        "MSP": 1.0, "DTW": 1.0, "PHL": 1.05, "CLT": 1.0, "SLC": 0.95,
        # Tier 3: Medium Airports
        "BWI": 0.95, "SAN": 1.0, "TPA": 0.95, "PDX": 1.0, "STL": 0.9,
        "BNA": 0.95, "AUS": 1.0, "RDU": 0.9, "MCI": 0.85, "IND": 0.85,
    }

    # Airline pricing tiers
    AIRLINE_TIERS = {
        # Premium Carriers
        "Delta": 1.15, "United": 1.1, "American": 1.1,
        # Mid-Tier
        "Alaska": 1.0, "JetBlue": 0.95, "Southwest": 0.9,
        # Budget Carriers
        "Spirit": 0.7, "Frontier": 0.7, "Allegiant": 0.75,
    }

    # Route distance estimates (in miles) - simplified
    ROUTE_DISTANCES = {
        ("JFK", "LAX"): 2475, ("JFK", "SFO"): 2586, ("JFK", "MIA"): 1090,
        ("LAX", "JFK"): 2475, ("LAX", "ORD"): 1745, ("LAX", "SEA"): 954,
        ("ORD", "LAX"): 1745, ("ORD", "JFK"): 740, ("ORD", "MIA"): 1197,
        ("ATL", "JFK"): 760, ("ATL", "LAX"): 1946, ("ATL", "ORD"): 606,
        ("DFW", "JFK"): 1391, ("DFW", "LAX"): 1235, ("DFW", "ORD"): 802,
        ("SFO", "JFK"): 2586, ("SFO", "LAX"): 337, ("SFO", "SEA"): 679,
        ("MIA", "JFK"): 1090, ("MIA", "LAX"): 2342, ("MIA", "ORD"): 1197,
        ("SEA", "LAX"): 954, ("SEA", "SFO"): 679, ("SEA", "JFK"): 2421,
        ("BOS", "JFK"): 187, ("BOS", "LAX"): 2611, ("BOS", "MIA"): 1258,
        ("DEN", "JFK"): 1626, ("DEN", "LAX"): 862, ("DEN", "ORD"): 888,
    }

    def __init__(self):
        self.fitted = False
        self._encoder_mappings: Dict[str, Dict[str, int]] = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load flight data from CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        return df

    def create_sample_data(self, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic flight data for training/testing."""
        np.random.seed(seed)

        airports = list(self.AIRPORT_TIERS.keys())
        airlines = list(self.AIRLINE_TIERS.keys())

        data = []
        for _ in range(n_samples):
            origin = np.random.choice(airports)
            dest = np.random.choice([a for a in airports if a != origin])
            airline = np.random.choice(airlines)

            # Calculate base price using distance and factors
            distance = self._get_distance(origin, dest)
            base_price = self._calculate_base_price(origin, dest, airline, distance)

            # Add noise
            noise = np.random.normal(0, base_price * 0.15)
            price = max(50, base_price + noise)

            data.append({
                "origin": origin,
                "destination": dest,
                "airline": airline,
                "distance": distance,
                "price": round(price, 2)
            })

        return pd.DataFrame(data)

    def _get_distance(self, origin: str, dest: str) -> float:
        """Get or estimate distance between airports."""
        if (origin, dest) in self.ROUTE_DISTANCES:
            return self.ROUTE_DISTANCES[(origin, dest)]
        if (dest, origin) in self.ROUTE_DISTANCES:
            return self.ROUTE_DISTANCES[(dest, origin)]
        # Estimate based on random factor for unknown routes
        return np.random.uniform(300, 2500)

    def _calculate_base_price(self, origin: str, dest: str,
                              airline: str, distance: float) -> float:
        """Calculate base price using pricing factors."""
        # Base rate per mile
        rate_per_mile = 0.12

        # Get pricing factors
        origin_factor = self.AIRPORT_TIERS.get(origin, 1.0)
        dest_factor = self.AIRPORT_TIERS.get(dest, 1.0)
        airline_factor = self.AIRLINE_TIERS.get(airline, 1.0)

        # Calculate price
        base = distance * rate_per_mile
        airport_multiplier = (origin_factor + dest_factor) / 2

        return base * airport_multiplier * airline_factor + 50  # $50 minimum fee

    def preprocess(self, df: pd.DataFrame,
                   fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess flight data for ML models.

        Args:
            df: DataFrame with origin, destination, airline columns
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            Tuple of (features, labels) where labels is None if 'price' not in df
        """
        df = df.copy()

        # Validate required columns
        required = ["origin", "destination", "airline"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Encode categorical features
        if fit:
            self._fit_encoders(df)
            self.fitted = True
        elif not self.fitted:
            raise ValueError("Processor not fitted. Call preprocess with fit=True first.")

        # Apply encoding
        features = self._encode_features(df)

        # Extract labels if present
        labels = None
        if "price" in df.columns:
            labels = df["price"].values

        return features, labels

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """Fit label encoders for categorical columns."""
        for col in ["origin", "destination", "airline"]:
            unique_values = sorted(df[col].unique())
            self._encoder_mappings[col] = {v: i for i, v in enumerate(unique_values)}

    def _encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode categorical features to numerical."""
        features = []

        for col in ["origin", "destination", "airline"]:
            encoded = df[col].map(self._encoder_mappings[col])
            # Handle unknown values
            encoded = encoded.fillna(-1).astype(int)
            features.append(encoded.values.reshape(-1, 1))

        # Add distance if available
        if "distance" in df.columns:
            features.append(df["distance"].values.reshape(-1, 1))
        else:
            # Estimate distances
            distances = df.apply(
                lambda row: self._get_distance(row["origin"], row["destination"]),
                axis=1
            )
            features.append(distances.values.reshape(-1, 1))

        # Add airport tier features
        origin_tiers = df["origin"].map(self.AIRPORT_TIERS).fillna(1.0)
        dest_tiers = df["destination"].map(self.AIRPORT_TIERS).fillna(1.0)
        airline_tiers = df["airline"].map(self.AIRLINE_TIERS).fillna(1.0)

        features.extend([
            origin_tiers.values.reshape(-1, 1),
            dest_tiers.values.reshape(-1, 1),
            airline_tiers.values.reshape(-1, 1)
        ])

        return np.hstack(features).astype(np.float32)

    def get_feature_names(self) -> List[str]:
        """Get names of features in order."""
        return [
            "origin_encoded", "destination_encoded", "airline_encoded",
            "distance", "origin_tier", "destination_tier", "airline_tier"
        ]
