"""
Flight Data Validator
=====================

Validates flight data using Pydantic models for type safety and data integrity.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class FlightInput(BaseModel):
    """Validated flight input for price prediction."""

    origin: str = Field(..., min_length=3, max_length=4, description="Origin airport code")
    destination: str = Field(..., min_length=3, max_length=4, description="Destination airport code")
    airline: str = Field(..., min_length=2, description="Airline name")

    @field_validator("origin", "destination")
    @classmethod
    def uppercase_airport(cls, v: str) -> str:
        """Convert airport codes to uppercase."""
        return v.upper()

    @field_validator("origin", "destination")
    @classmethod
    def validate_different_airports(cls, v: str, info) -> str:
        """Ensure origin and destination are different."""
        # This runs for each field, actual comparison in model_validator
        return v

    def model_post_init(self, __context) -> None:
        """Validate that origin and destination are different."""
        if self.origin == self.destination:
            raise ValueError("Origin and destination must be different airports")


class FlightPrediction(BaseModel):
    """Flight price prediction result."""

    origin: str
    destination: str
    airline: str
    predicted_price: float = Field(..., ge=0, description="Predicted price in USD")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence 0-1")
    price_range: tuple[float, float] = Field(..., description="Price range (min, max)")
    model_used: str = Field(..., description="Model that generated prediction")


class FlightDataValidator:
    """Validates flight data for the prediction system."""

    VALID_AIRPORTS = {
        "JFK", "LAX", "ORD", "ATL", "DFW", "SFO", "MIA", "SEA", "BOS", "DEN",
        "PHX", "IAH", "LAS", "MCO", "EWR", "MSP", "DTW", "PHL", "CLT", "SLC",
        "BWI", "SAN", "TPA", "PDX", "STL", "BNA", "AUS", "RDU", "MCI", "IND"
    }

    VALID_AIRLINES = {
        "Delta", "United", "American", "Alaska", "JetBlue",
        "Southwest", "Spirit", "Frontier", "Allegiant"
    }

    def validate_input(self, origin: str, destination: str,
                       airline: str) -> FlightInput:
        """
        Validate flight input parameters.

        Raises:
            ValueError: If validation fails
        """
        return FlightInput(origin=origin, destination=destination, airline=airline)

    def validate_airport(self, code: str) -> bool:
        """Check if airport code is valid."""
        return code.upper() in self.VALID_AIRPORTS

    def validate_airline(self, name: str) -> bool:
        """Check if airline name is valid."""
        return name in self.VALID_AIRLINES

    def get_validation_errors(self, origin: str, destination: str,
                              airline: str) -> List[str]:
        """Get list of validation errors for inputs."""
        errors = []

        if not self.validate_airport(origin):
            errors.append(f"Unknown origin airport: {origin}")
        if not self.validate_airport(destination):
            errors.append(f"Unknown destination airport: {destination}")
        if origin.upper() == destination.upper():
            errors.append("Origin and destination must be different")
        if not self.validate_airline(airline):
            errors.append(f"Unknown airline: {airline}")

        return errors
