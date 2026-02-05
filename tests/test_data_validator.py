"""
Tests for FlightDataValidator
=============================

Unit tests for input validation using Pydantic models.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.validator import FlightDataValidator, FlightInput, FlightPrediction


class TestFlightInput:
    """Tests for FlightInput Pydantic model."""

    def test_valid_input(self):
        """Test valid flight input."""
        flight = FlightInput(origin="JFK", destination="LAX", airline="Delta")

        assert flight.origin == "JFK"
        assert flight.destination == "LAX"
        assert flight.airline == "Delta"

    def test_lowercase_airport_converted(self):
        """Test that lowercase airport codes are converted to uppercase."""
        flight = FlightInput(origin="jfk", destination="lax", airline="Delta")

        assert flight.origin == "JFK"
        assert flight.destination == "LAX"

    def test_mixed_case_airport_converted(self):
        """Test mixed case airport codes are converted."""
        flight = FlightInput(origin="JfK", destination="LaX", airline="Delta")

        assert flight.origin == "JFK"
        assert flight.destination == "LAX"

    def test_same_origin_destination_fails(self):
        """Test that same origin and destination raises error."""
        with pytest.raises(ValueError, match="different"):
            FlightInput(origin="JFK", destination="JFK", airline="Delta")

    def test_empty_origin_fails(self):
        """Test that empty origin fails validation."""
        with pytest.raises(ValueError):
            FlightInput(origin="", destination="LAX", airline="Delta")

    def test_short_airport_code_fails(self):
        """Test that airport codes less than 3 chars fail."""
        with pytest.raises(ValueError):
            FlightInput(origin="JF", destination="LAX", airline="Delta")

    def test_long_airport_code_fails(self):
        """Test that airport codes more than 4 chars fail."""
        with pytest.raises(ValueError):
            FlightInput(origin="JFKXX", destination="LAX", airline="Delta")

    def test_short_airline_fails(self):
        """Test that very short airline name fails."""
        with pytest.raises(ValueError):
            FlightInput(origin="JFK", destination="LAX", airline="D")


class TestFlightPrediction:
    """Tests for FlightPrediction Pydantic model."""

    def test_valid_prediction(self):
        """Test valid prediction creation."""
        pred = FlightPrediction(
            origin="JFK",
            destination="LAX",
            airline="Delta",
            predicted_price=350.00,
            confidence=0.85,
            price_range=(300.00, 400.00),
            model_used="ensemble"
        )

        assert pred.predicted_price == 350.00
        assert pred.confidence == 0.85

    def test_negative_price_fails(self):
        """Test that negative price fails validation."""
        with pytest.raises(ValueError):
            FlightPrediction(
                origin="JFK",
                destination="LAX",
                airline="Delta",
                predicted_price=-50.00,
                confidence=0.85,
                price_range=(300.00, 400.00),
                model_used="ensemble"
            )

    def test_confidence_out_of_range_fails(self):
        """Test that confidence outside 0-1 fails."""
        with pytest.raises(ValueError):
            FlightPrediction(
                origin="JFK",
                destination="LAX",
                airline="Delta",
                predicted_price=350.00,
                confidence=1.5,  # Invalid
                price_range=(300.00, 400.00),
                model_used="ensemble"
            )


class TestFlightDataValidator:
    """Tests for FlightDataValidator class."""

    def test_validate_input_success(self):
        """Test successful input validation."""
        validator = FlightDataValidator()
        result = validator.validate_input("JFK", "LAX", "Delta")

        assert result.origin == "JFK"
        assert result.destination == "LAX"

    def test_validate_input_converts_case(self):
        """Test that validation converts airport case."""
        validator = FlightDataValidator()
        result = validator.validate_input("jfk", "lax", "Delta")

        assert result.origin == "JFK"
        assert result.destination == "LAX"

    def test_validate_airport_valid(self):
        """Test valid airport code validation."""
        validator = FlightDataValidator()

        assert validator.validate_airport("JFK") is True
        assert validator.validate_airport("LAX") is True
        assert validator.validate_airport("ORD") is True

    def test_validate_airport_invalid(self):
        """Test invalid airport code validation."""
        validator = FlightDataValidator()

        assert validator.validate_airport("ZZZ") is False
        assert validator.validate_airport("XXX") is False

    def test_validate_airline_valid(self):
        """Test valid airline validation."""
        validator = FlightDataValidator()

        assert validator.validate_airline("Delta") is True
        assert validator.validate_airline("United") is True
        assert validator.validate_airline("Southwest") is True

    def test_validate_airline_invalid(self):
        """Test invalid airline validation."""
        validator = FlightDataValidator()

        assert validator.validate_airline("FakeAir") is False
        assert validator.validate_airline("UnknownCarrier") is False

    def test_get_validation_errors_none(self):
        """Test no errors for valid input."""
        validator = FlightDataValidator()
        errors = validator.get_validation_errors("JFK", "LAX", "Delta")

        assert len(errors) == 0

    def test_get_validation_errors_invalid_origin(self):
        """Test error for invalid origin."""
        validator = FlightDataValidator()
        errors = validator.get_validation_errors("ZZZ", "LAX", "Delta")

        assert len(errors) == 1
        assert "origin" in errors[0].lower()

    def test_get_validation_errors_invalid_destination(self):
        """Test error for invalid destination."""
        validator = FlightDataValidator()
        errors = validator.get_validation_errors("JFK", "ZZZ", "Delta")

        assert len(errors) == 1
        assert "destination" in errors[0].lower()

    def test_get_validation_errors_same_airports(self):
        """Test error for same origin and destination."""
        validator = FlightDataValidator()
        errors = validator.get_validation_errors("JFK", "JFK", "Delta")

        assert len(errors) == 1
        assert "different" in errors[0].lower()

    def test_get_validation_errors_invalid_airline(self):
        """Test error for invalid airline."""
        validator = FlightDataValidator()
        errors = validator.get_validation_errors("JFK", "LAX", "FakeAir")

        assert len(errors) == 1
        assert "airline" in errors[0].lower()

    def test_get_validation_errors_multiple(self):
        """Test multiple validation errors."""
        validator = FlightDataValidator()
        errors = validator.get_validation_errors("ZZZ", "ZZZ", "FakeAir")

        assert len(errors) >= 2  # At least origin invalid and airline invalid

    def test_valid_airports_list(self):
        """Test that VALID_AIRPORTS contains expected airports."""
        validator = FlightDataValidator()

        expected = {"JFK", "LAX", "ORD", "ATL", "DFW", "SFO", "MIA"}
        assert expected.issubset(validator.VALID_AIRPORTS)

    def test_valid_airlines_list(self):
        """Test that VALID_AIRLINES contains expected airlines."""
        validator = FlightDataValidator()

        expected = {"Delta", "United", "American", "Southwest", "Spirit"}
        assert expected.issubset(validator.VALID_AIRLINES)
