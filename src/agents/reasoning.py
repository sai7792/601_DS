"""
Price Reasoning Engine
======================

Provides explanations and reasoning for flight price predictions.
Uses rule-based and learned patterns to explain pricing.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PriceFactor(Enum):
    """Factors that influence flight prices."""
    DISTANCE = "distance"
    ORIGIN_HUB = "origin_hub"
    DESTINATION_HUB = "destination_hub"
    AIRLINE_TIER = "airline_tier"
    ROUTE_POPULARITY = "route_popularity"
    COMPETITION = "competition"


@dataclass
class PriceExplanation:
    """Structured explanation of a price prediction."""
    base_price: float
    final_price: float
    factors: List[Dict[str, Any]]
    summary: str
    confidence_explanation: str
    recommendations: List[str]


class PriceReasoningEngine:
    """
    Engine for explaining and reasoning about flight prices.

    Provides:
    - Factor-based price breakdowns
    - Natural language explanations
    - Price optimization recommendations
    - Comparative analysis reasoning
    """

    # Price factor weights for explanation
    FACTOR_WEIGHTS = {
        PriceFactor.DISTANCE: 0.35,
        PriceFactor.ORIGIN_HUB: 0.15,
        PriceFactor.DESTINATION_HUB: 0.15,
        PriceFactor.AIRLINE_TIER: 0.25,
        PriceFactor.ROUTE_POPULARITY: 0.05,
        PriceFactor.COMPETITION: 0.05
    }

    # Hub descriptions
    HUB_DESCRIPTIONS = {
        "JFK": "New York JFK, a major international hub",
        "LAX": "Los Angeles International, a Pacific gateway",
        "ORD": "Chicago O'Hare, a central US hub",
        "ATL": "Atlanta Hartsfield, the world's busiest airport",
        "DFW": "Dallas-Fort Worth, a major Southern hub",
        "SFO": "San Francisco International, a tech corridor hub",
        "MIA": "Miami International, a Latin America gateway",
        "SEA": "Seattle-Tacoma, a Pacific Northwest hub",
        "BOS": "Boston Logan, a Northeast hub",
        "DEN": "Denver International, a mountain region hub"
    }

    # Airline descriptions
    AIRLINE_DESCRIPTIONS = {
        "Delta": "premium full-service carrier with extensive network",
        "United": "major full-service carrier with global reach",
        "American": "largest US carrier by fleet size",
        "Alaska": "known for West Coast routes and reliability",
        "JetBlue": "affordable premium with good legroom",
        "Southwest": "budget-friendly with no change fees",
        "Spirit": "ultra-low-cost with basic fares",
        "Frontier": "ultra-low-cost budget carrier",
        "Allegiant": "budget carrier focusing on leisure routes"
    }

    def __init__(self, processor=None):
        self.processor = processor

    def explain_prediction(self, origin: str, destination: str,
                           airline: str, predicted_price: float,
                           confidence: float) -> PriceExplanation:
        """
        Generate a detailed explanation for a price prediction.

        Args:
            origin: Origin airport code
            destination: Destination airport code
            airline: Airline name
            predicted_price: The predicted price
            confidence: Model confidence (0-1)

        Returns:
            PriceExplanation with detailed breakdown
        """
        factors = []
        base_price = 50.0  # Minimum base

        # Distance factor
        distance = self._get_distance(origin, destination)
        distance_contribution = distance * 0.12
        factors.append({
            "factor": PriceFactor.DISTANCE.value,
            "value": distance,
            "contribution": distance_contribution,
            "explanation": f"Route distance of {distance:.0f} miles"
        })

        # Origin hub factor
        origin_tier = self._get_airport_tier(origin)
        origin_contribution = base_price * (origin_tier - 1) * 0.5
        factors.append({
            "factor": PriceFactor.ORIGIN_HUB.value,
            "value": origin_tier,
            "contribution": origin_contribution,
            "explanation": self._explain_airport(origin)
        })

        # Destination hub factor
        dest_tier = self._get_airport_tier(destination)
        dest_contribution = base_price * (dest_tier - 1) * 0.5
        factors.append({
            "factor": PriceFactor.DESTINATION_HUB.value,
            "value": dest_tier,
            "contribution": dest_contribution,
            "explanation": self._explain_airport(destination)
        })

        # Airline tier factor
        airline_tier = self._get_airline_tier(airline)
        airline_contribution = predicted_price * (airline_tier - 1) * 0.3
        factors.append({
            "factor": PriceFactor.AIRLINE_TIER.value,
            "value": airline_tier,
            "contribution": airline_contribution,
            "explanation": self._explain_airline(airline)
        })

        # Generate summary
        summary = self._generate_summary(
            origin, destination, airline,
            predicted_price, distance, airline_tier
        )

        # Confidence explanation
        confidence_explanation = self._explain_confidence(confidence)

        # Recommendations
        recommendations = self._generate_recommendations(
            origin, destination, airline, predicted_price, airline_tier
        )

        return PriceExplanation(
            base_price=base_price,
            final_price=predicted_price,
            factors=factors,
            summary=summary,
            confidence_explanation=confidence_explanation,
            recommendations=recommendations
        )

    def _get_distance(self, origin: str, destination: str) -> float:
        """Get estimated distance between airports."""
        if self.processor:
            return self.processor._get_distance(origin, destination)
        return 1000.0  # Default estimate

    def _get_airport_tier(self, code: str) -> float:
        """Get airport pricing tier."""
        if self.processor:
            return self.processor.AIRPORT_TIERS.get(code.upper(), 1.0)
        return 1.0

    def _get_airline_tier(self, airline: str) -> float:
        """Get airline pricing tier."""
        if self.processor:
            return self.processor.AIRLINE_TIERS.get(airline, 1.0)
        return 1.0

    def _explain_airport(self, code: str) -> str:
        """Generate explanation for airport's impact on price."""
        tier = self._get_airport_tier(code)
        description = self.HUB_DESCRIPTIONS.get(
            code.upper(),
            f"{code} airport"
        )

        if tier >= 1.2:
            return f"{description} - Major hub with premium pricing"
        elif tier >= 1.0:
            return f"{description} - Standard regional pricing"
        else:
            return f"{description} - Smaller airport with lower fees"

    def _explain_airline(self, airline: str) -> str:
        """Generate explanation for airline's impact on price."""
        tier = self._get_airline_tier(airline)
        description = self.AIRLINE_DESCRIPTIONS.get(
            airline,
            "airline"
        )

        if tier >= 1.1:
            return f"{airline} is a {description} - Premium pricing"
        elif tier >= 0.9:
            return f"{airline} is a {description} - Mid-range pricing"
        else:
            return f"{airline} is a {description} - Budget pricing"

    def _generate_summary(self, origin: str, destination: str,
                          airline: str, price: float,
                          distance: float, airline_tier: float) -> str:
        """Generate a natural language summary."""
        price_assessment = "moderate"
        if price > 400:
            price_assessment = "premium"
        elif price < 150:
            price_assessment = "budget-friendly"

        airline_type = "premium" if airline_tier >= 1.1 else (
            "mid-tier" if airline_tier >= 0.9 else "budget"
        )

        return (
            f"The predicted price of ${price:.2f} for the {origin} to {destination} "
            f"route on {airline} reflects a {price_assessment} fare. "
            f"This is a {distance:.0f}-mile route served by a {airline_type} carrier. "
            f"The price accounts for airport fees, route distance, and airline service level."
        )

    def _explain_confidence(self, confidence: float) -> str:
        """Explain the confidence level."""
        if confidence >= 0.85:
            return (
                f"High confidence ({confidence:.0%}): The model has strong agreement "
                "across all algorithms. This route has predictable pricing patterns."
            )
        elif confidence >= 0.7:
            return (
                f"Good confidence ({confidence:.0%}): The prediction is reliable. "
                "Some variation possible due to market factors."
            )
        elif confidence >= 0.5:
            return (
                f"Moderate confidence ({confidence:.0%}): The prediction provides "
                "a reasonable estimate but actual prices may vary more significantly."
            )
        else:
            return (
                f"Lower confidence ({confidence:.0%}): This route may have unusual "
                "pricing patterns. Consider the price range for planning."
            )

    def _generate_recommendations(self, origin: str, destination: str,
                                   airline: str, price: float,
                                   airline_tier: float) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Airline recommendation
        if airline_tier >= 1.1:
            recommendations.append(
                f"Consider budget carriers like Spirit or Frontier for savings "
                f"of 30-40% if flexible on amenities."
            )
        elif airline_tier <= 0.8:
            recommendations.append(
                f"You've chosen a budget carrier. Consider mid-tier options like "
                f"JetBlue or Southwest for better experience at moderate cost increase."
            )

        # Route recommendations
        if price > 300:
            recommendations.append(
                "For this longer route, consider connecting flights through "
                "hub airports for potential savings."
            )

        # General tips
        recommendations.append(
            "Flight prices can vary. Use this prediction as a baseline "
            "when comparing actual booking options."
        )

        return recommendations

    def compare_options(self, predictions: List[Dict[str, Any]]) -> str:
        """
        Generate a comparison analysis for multiple flight options.

        Args:
            predictions: List of prediction results

        Returns:
            Natural language comparison
        """
        if not predictions:
            return "No options to compare."

        sorted_preds = sorted(predictions, key=lambda x: x.get("predicted_price", 0))

        cheapest = sorted_preds[0]
        most_expensive = sorted_preds[-1]

        price_range = most_expensive["predicted_price"] - cheapest["predicted_price"]
        savings_pct = (price_range / most_expensive["predicted_price"]) * 100

        analysis = (
            f"Comparing {len(predictions)} options:\n\n"
            f"Best Value: {cheapest['airline']} at ${cheapest['predicted_price']:.2f}\n"
            f"Premium Option: {most_expensive['airline']} at ${most_expensive['predicted_price']:.2f}\n\n"
            f"Price Range: ${price_range:.2f} ({savings_pct:.0f}% potential savings)\n\n"
        )

        if savings_pct > 30:
            analysis += (
                "Significant price variation exists. Budget carriers offer "
                "substantial savings for flexible travelers."
            )
        else:
            analysis += (
                "Prices are relatively consistent across carriers. "
                "Consider factors like service quality and loyalty programs."
            )

        return analysis
