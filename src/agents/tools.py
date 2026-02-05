"""
Flight Agent Tools
==================

Tool definitions for the AI agent to interact with the flight price prediction system.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ToolCategory(Enum):
    """Categories of available tools."""
    PREDICTION = "prediction"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    SEARCH = "search"


@dataclass
class Tool:
    """Definition of a tool available to the agent."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Dict[str, Any]]
    required_params: List[str]
    func: Optional[Callable] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to function calling schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }


class FlightTools:
    """
    Collection of tools for the flight price agent.

    These tools allow the agent to:
    - Predict flight prices
    - Compare airlines and routes
    - Analyze pricing patterns
    - Search for optimal flights
    """

    def __init__(self, predictor=None):
        self.predictor = predictor
        self._tools: Dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""

        # Price prediction tool
        self._tools["predict_price"] = Tool(
            name="predict_price",
            description="Predict the price of a flight given origin, destination, and airline",
            category=ToolCategory.PREDICTION,
            parameters={
                "origin": {
                    "type": "string",
                    "description": "Origin airport code (e.g., JFK, LAX)"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination airport code (e.g., SFO, ORD)"
                },
                "airline": {
                    "type": "string",
                    "description": "Airline name (e.g., Delta, United, Southwest)"
                }
            },
            required_params=["origin", "destination", "airline"],
            func=self._predict_price
        )

        # Compare airlines tool
        self._tools["compare_airlines"] = Tool(
            name="compare_airlines",
            description="Compare prices across all airlines for a specific route",
            category=ToolCategory.COMPARISON,
            parameters={
                "origin": {
                    "type": "string",
                    "description": "Origin airport code"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination airport code"
                }
            },
            required_params=["origin", "destination"],
            func=self._compare_airlines
        )

        # Find cheapest destinations tool
        self._tools["find_cheapest_destinations"] = Tool(
            name="find_cheapest_destinations",
            description="Find the cheapest destinations from a given origin with a specific airline",
            category=ToolCategory.SEARCH,
            parameters={
                "origin": {
                    "type": "string",
                    "description": "Origin airport code"
                },
                "airline": {
                    "type": "string",
                    "description": "Airline name"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)"
                }
            },
            required_params=["origin", "airline"],
            func=self._find_cheapest_destinations
        )

        # Analyze route tool
        self._tools["analyze_route"] = Tool(
            name="analyze_route",
            description="Get detailed analysis of a route including all airline options and price factors",
            category=ToolCategory.ANALYSIS,
            parameters={
                "origin": {
                    "type": "string",
                    "description": "Origin airport code"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination airport code"
                }
            },
            required_params=["origin", "destination"],
            func=self._analyze_route
        )

        # Get airport info tool
        self._tools["get_airport_info"] = Tool(
            name="get_airport_info",
            description="Get information about an airport including its pricing tier",
            category=ToolCategory.ANALYSIS,
            parameters={
                "airport_code": {
                    "type": "string",
                    "description": "Airport code (e.g., JFK)"
                }
            },
            required_params=["airport_code"],
            func=self._get_airport_info
        )

        # Get airline info tool
        self._tools["get_airline_info"] = Tool(
            name="get_airline_info",
            description="Get information about an airline including its pricing tier",
            category=ToolCategory.ANALYSIS,
            parameters={
                "airline": {
                    "type": "string",
                    "description": "Airline name"
                }
            },
            required_params=["airline"],
            func=self._get_airline_info
        )

    def _predict_price(self, origin: str, destination: str,
                       airline: str) -> Dict[str, Any]:
        """Execute price prediction."""
        if self.predictor is None:
            return {"error": "Predictor not initialized"}

        try:
            prediction = self.predictor.predict(origin, destination, airline)
            return {
                "origin": prediction.origin,
                "destination": prediction.destination,
                "airline": prediction.airline,
                "predicted_price": prediction.predicted_price,
                "confidence": prediction.confidence,
                "price_range": prediction.price_range,
                "model_used": prediction.model_used
            }
        except Exception as e:
            return {"error": str(e)}

    def _compare_airlines(self, origin: str, destination: str) -> Dict[str, Any]:
        """Compare prices across airlines."""
        if self.predictor is None:
            return {"error": "Predictor not initialized"}

        try:
            predictions = self.predictor.compare_airlines(origin, destination)
            return {
                "route": f"{origin} -> {destination}",
                "comparisons": [
                    {
                        "airline": p.airline,
                        "price": p.predicted_price,
                        "confidence": p.confidence
                    }
                    for p in predictions
                ],
                "cheapest": predictions[0].airline if predictions else None,
                "most_expensive": predictions[-1].airline if predictions else None
            }
        except Exception as e:
            return {"error": str(e)}

    def _find_cheapest_destinations(self, origin: str, airline: str,
                                    limit: int = 5) -> Dict[str, Any]:
        """Find cheapest destinations."""
        if self.predictor is None:
            return {"error": "Predictor not initialized"}

        try:
            predictions = self.predictor.get_cheapest_route(origin, airline)[:limit]
            return {
                "origin": origin,
                "airline": airline,
                "destinations": [
                    {
                        "destination": p.destination,
                        "price": p.predicted_price,
                        "confidence": p.confidence
                    }
                    for p in predictions
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """Analyze a route comprehensively."""
        if self.predictor is None:
            return {"error": "Predictor not initialized"}

        try:
            # Get all airline prices
            predictions = self.predictor.compare_airlines(origin, destination)

            prices = [p.predicted_price for p in predictions]
            avg_price = sum(prices) / len(prices) if prices else 0

            # Get distance estimate
            distance = self.predictor.processor._get_distance(origin, destination)

            return {
                "route": f"{origin} -> {destination}",
                "distance_miles": distance,
                "average_price": round(avg_price, 2),
                "price_range": (min(prices), max(prices)) if prices else (0, 0),
                "cheapest_option": {
                    "airline": predictions[0].airline,
                    "price": predictions[0].predicted_price
                } if predictions else None,
                "premium_option": {
                    "airline": predictions[-1].airline,
                    "price": predictions[-1].predicted_price
                } if predictions else None,
                "all_options": [
                    {"airline": p.airline, "price": p.predicted_price}
                    for p in predictions
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_airport_info(self, airport_code: str) -> Dict[str, Any]:
        """Get airport information."""
        try:
            from ..data.processor import FlightDataProcessor
        except ImportError:
            from data.processor import FlightDataProcessor

        code = airport_code.upper()
        tier = FlightDataProcessor.AIRPORT_TIERS.get(code, 1.0)

        tier_name = "Unknown"
        if tier >= 1.2:
            tier_name = "Major International Hub"
        elif tier >= 1.0:
            tier_name = "Large Regional Airport"
        elif tier >= 0.9:
            tier_name = "Medium Airport"
        else:
            tier_name = "Smaller Airport"

        return {
            "code": code,
            "pricing_factor": tier,
            "tier": tier_name,
            "is_valid": code in FlightDataProcessor.AIRPORT_TIERS
        }

    def _get_airline_info(self, airline: str) -> Dict[str, Any]:
        """Get airline information."""
        try:
            from ..data.processor import FlightDataProcessor
        except ImportError:
            from data.processor import FlightDataProcessor

        tier = FlightDataProcessor.AIRLINE_TIERS.get(airline, 1.0)

        tier_name = "Unknown"
        if tier >= 1.1:
            tier_name = "Premium Carrier"
        elif tier >= 0.9:
            tier_name = "Mid-Tier Carrier"
        else:
            tier_name = "Budget Carrier"

        return {
            "name": airline,
            "pricing_factor": tier,
            "tier": tier_name,
            "is_valid": airline in FlightDataProcessor.AIRLINE_TIERS
        }

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Get all available tools."""
        return list(self._tools.values())

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        if tool.func is None:
            return {"error": f"Tool {tool_name} has no implementation"}

        # Validate required parameters
        missing = [p for p in tool.required_params if p not in kwargs]
        if missing:
            return {"error": f"Missing required parameters: {missing}"}

        return tool.func(**kwargs)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for function calling."""
        return [tool.to_schema() for tool in self._tools.values()]
