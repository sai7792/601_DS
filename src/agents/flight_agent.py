"""
Flight Price AI Agent
=====================

Intelligent agent for flight price predictions using tool-based reasoning.
Supports both local execution and LLM-powered natural language interaction.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import re

try:
    from .tools import FlightTools, ToolCategory
    from .reasoning import PriceReasoningEngine
    from ..models.price_predictor import FlightPricePredictor
except ImportError:
    from agents.tools import FlightTools, ToolCategory
    from agents.reasoning import PriceReasoningEngine
    from models.price_predictor import FlightPricePredictor


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class AgentResponse:
    """Response from the flight agent."""
    answer: str
    tool_calls: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    suggestions: List[str]


class FlightPriceAgent:
    """
    AI Agent for intelligent flight price predictions.

    Capabilities:
    - Natural language query understanding
    - Multi-step reasoning for complex queries
    - Tool-based execution for predictions
    - Explanatory responses with recommendations

    Can operate in two modes:
    1. Local mode: Rule-based query parsing and execution
    2. LLM mode: Uses language model for understanding (requires API key)
    """

    def __init__(self, predictor: Optional[FlightPricePredictor] = None,
                 use_llm: bool = False):
        """
        Initialize the flight price agent.

        Args:
            predictor: Trained FlightPricePredictor instance
            use_llm: Whether to use LLM for query understanding
        """
        self.predictor = predictor
        self.use_llm = use_llm
        self.tools = FlightTools(predictor)
        self.reasoning_engine = PriceReasoningEngine(
            predictor.processor if predictor else None
        )
        self.state = AgentState.IDLE
        self._conversation_history: List[Dict[str, str]] = []
        self._llm_client = None

        if use_llm:
            self._setup_llm()

    def _setup_llm(self) -> None:
        """Setup LLM client for natural language understanding."""
        try:
            import os
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._llm_client = OpenAI(api_key=api_key)
        except ImportError:
            self.use_llm = False

    def query(self, user_input: str) -> AgentResponse:
        """
        Process a natural language query about flight prices.

        Args:
            user_input: Natural language question or request

        Returns:
            AgentResponse with answer and supporting information
        """
        self.state = AgentState.THINKING

        # Store in conversation history
        self._conversation_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            if self.use_llm and self._llm_client:
                response = self._process_with_llm(user_input)
            else:
                response = self._process_locally(user_input)

            self._conversation_history.append({
                "role": "assistant",
                "content": response.answer
            })

            self.state = AgentState.IDLE
            return response

        except Exception as e:
            self.state = AgentState.ERROR
            return AgentResponse(
                answer=f"I encountered an error: {str(e)}",
                tool_calls=[],
                reasoning="Error during processing",
                confidence=0.0,
                suggestions=["Please try rephrasing your question"]
            )

    def _process_locally(self, user_input: str) -> AgentResponse:
        """Process query using local rule-based understanding."""
        user_lower = user_input.lower()
        tool_calls = []
        reasoning_steps = []

        # Parse intent and entities
        intent = self._detect_intent(user_lower)
        entities = self._extract_entities(user_input)

        reasoning_steps.append(f"Detected intent: {intent}")
        reasoning_steps.append(f"Extracted entities: {entities}")

        # Execute appropriate tools based on intent
        if intent == "predict_price":
            if all(k in entities for k in ["origin", "destination", "airline"]):
                result = self.tools.execute(
                    "predict_price",
                    origin=entities["origin"],
                    destination=entities["destination"],
                    airline=entities["airline"]
                )
                tool_calls.append({"tool": "predict_price", "result": result})

                if "error" not in result:
                    explanation = self.reasoning_engine.explain_prediction(
                        entities["origin"],
                        entities["destination"],
                        entities["airline"],
                        result["predicted_price"],
                        result["confidence"]
                    )

                    answer = self._format_price_response(result, explanation)
                    return AgentResponse(
                        answer=answer,
                        tool_calls=tool_calls,
                        reasoning="\n".join(reasoning_steps),
                        confidence=result["confidence"],
                        suggestions=explanation.recommendations
                    )
                else:
                    return AgentResponse(
                        answer=f"I couldn't predict the price: {result['error']}",
                        tool_calls=tool_calls,
                        reasoning="\n".join(reasoning_steps),
                        confidence=0.0,
                        suggestions=["Check airport codes and airline name"]
                    )
            else:
                missing = [k for k in ["origin", "destination", "airline"] if k not in entities]
                return AgentResponse(
                    answer=f"I need more information: {', '.join(missing)}",
                    tool_calls=[],
                    reasoning="\n".join(reasoning_steps),
                    confidence=0.0,
                    suggestions=[f"Please provide: {', '.join(missing)}"]
                )

        elif intent == "compare_airlines":
            if "origin" in entities and "destination" in entities:
                result = self.tools.execute(
                    "compare_airlines",
                    origin=entities["origin"],
                    destination=entities["destination"]
                )
                tool_calls.append({"tool": "compare_airlines", "result": result})

                if "error" not in result:
                    answer = self._format_comparison_response(result)
                    return AgentResponse(
                        answer=answer,
                        tool_calls=tool_calls,
                        reasoning="\n".join(reasoning_steps),
                        confidence=0.85,
                        suggestions=["Ask about a specific airline for more details"]
                    )

        elif intent == "find_cheapest":
            if "origin" in entities and "airline" in entities:
                result = self.tools.execute(
                    "find_cheapest_destinations",
                    origin=entities["origin"],
                    airline=entities["airline"],
                    limit=5
                )
                tool_calls.append({"tool": "find_cheapest_destinations", "result": result})

                if "error" not in result:
                    answer = self._format_cheapest_response(result)
                    return AgentResponse(
                        answer=answer,
                        tool_calls=tool_calls,
                        reasoning="\n".join(reasoning_steps),
                        confidence=0.8,
                        suggestions=["Ask for prices to specific destinations"]
                    )

        elif intent == "analyze_route":
            if "origin" in entities and "destination" in entities:
                result = self.tools.execute(
                    "analyze_route",
                    origin=entities["origin"],
                    destination=entities["destination"]
                )
                tool_calls.append({"tool": "analyze_route", "result": result})

                if "error" not in result:
                    answer = self._format_analysis_response(result)
                    return AgentResponse(
                        answer=answer,
                        tool_calls=tool_calls,
                        reasoning="\n".join(reasoning_steps),
                        confidence=0.85,
                        suggestions=["Compare with other routes", "Ask about specific airlines"]
                    )

        elif intent == "airport_info":
            if "airport" in entities:
                result = self.tools.execute(
                    "get_airport_info",
                    airport_code=entities["airport"]
                )
                tool_calls.append({"tool": "get_airport_info", "result": result})
                answer = self._format_airport_info(result)
                return AgentResponse(
                    answer=answer,
                    tool_calls=tool_calls,
                    reasoning="\n".join(reasoning_steps),
                    confidence=0.95,
                    suggestions=["Ask about flights from this airport"]
                )

        elif intent == "airline_info":
            if "airline" in entities:
                result = self.tools.execute(
                    "get_airline_info",
                    airline=entities["airline"]
                )
                tool_calls.append({"tool": "get_airline_info", "result": result})
                answer = self._format_airline_info(result)
                return AgentResponse(
                    answer=answer,
                    tool_calls=tool_calls,
                    reasoning="\n".join(reasoning_steps),
                    confidence=0.95,
                    suggestions=["Ask about prices on this airline"]
                )

        # Default response for unclear queries
        return AgentResponse(
            answer=self._get_help_response(),
            tool_calls=[],
            reasoning="Could not determine clear intent",
            confidence=0.0,
            suggestions=[
                "Try: 'What's the price from JFK to LAX on Delta?'",
                "Try: 'Compare airlines from SFO to ORD'",
                "Try: 'Find cheapest flights from BOS on JetBlue'"
            ]
        )

    def _detect_intent(self, text: str) -> str:
        """Detect user intent from text."""
        if any(w in text for w in ["price", "cost", "how much", "fare"]):
            if "compare" in text or "vs" in text or "versus" in text:
                return "compare_airlines"
            return "predict_price"

        if any(w in text for w in ["compare", "comparison", "difference"]):
            return "compare_airlines"

        if any(w in text for w in ["cheapest", "budget", "lowest", "save"]):
            return "find_cheapest"

        if any(w in text for w in ["analyze", "analysis", "overview", "about route"]):
            return "analyze_route"

        if any(w in text for w in ["airport info", "about airport", "which airport"]):
            return "airport_info"

        if any(w in text for w in ["airline info", "about airline", "which airline"]):
            return "airline_info"

        return "unknown"

    def _extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities (airports, airlines) from text."""
        entities = {}

        # Valid airport codes to recognize
        valid_airports = {
            "JFK", "LAX", "ORD", "ATL", "DFW", "SFO", "MIA", "SEA", "BOS", "DEN",
            "PHX", "IAH", "LAS", "MCO", "EWR", "MSP", "DTW", "PHL", "CLT", "SLC",
            "BWI", "SAN", "TPA", "PDX", "STL", "BNA", "AUS", "RDU", "MCI", "IND"
        }

        # Airport codes (3 letters, uppercase) - only match valid airport codes
        airport_pattern = r'\b([A-Z]{3})\b'
        potential_airports = re.findall(airport_pattern, text.upper())
        airports = [a for a in potential_airports if a in valid_airports]

        # Also check for common airport names
        airport_names = {
            "new york": "JFK", "los angeles": "LAX", "chicago": "ORD",
            "atlanta": "ATL", "dallas": "DFW", "san francisco": "SFO",
            "miami": "MIA", "seattle": "SEA", "boston": "BOS", "denver": "DEN"
        }

        text_lower = text.lower()
        for name, code in airport_names.items():
            if name in text_lower and code not in airports:
                airports.append(code)

        if len(airports) >= 2:
            entities["origin"] = airports[0]
            entities["destination"] = airports[1]
        elif len(airports) == 1:
            # Check context for origin vs destination
            if "from" in text_lower:
                entities["origin"] = airports[0]
            elif "to" in text_lower:
                entities["destination"] = airports[0]
            else:
                entities["airport"] = airports[0]
                entities["origin"] = airports[0]

        # Airlines
        airlines = ["Delta", "United", "American", "Alaska", "JetBlue",
                    "Southwest", "Spirit", "Frontier", "Allegiant"]

        for airline in airlines:
            if airline.lower() in text_lower:
                entities["airline"] = airline
                break

        return entities

    def _format_price_response(self, result: Dict, explanation) -> str:
        """Format price prediction response."""
        return (
            f"**Flight Price Prediction**\n\n"
            f"Route: {result['origin']} → {result['destination']}\n"
            f"Airline: {result['airline']}\n\n"
            f"**Predicted Price: ${result['predicted_price']:.2f}**\n"
            f"Price Range: ${result['price_range'][0]:.2f} - ${result['price_range'][1]:.2f}\n"
            f"Confidence: {result['confidence']:.0%}\n\n"
            f"{explanation.summary}\n\n"
            f"{explanation.confidence_explanation}"
        )

    def _format_comparison_response(self, result: Dict) -> str:
        """Format airline comparison response."""
        lines = [
            f"**Airline Comparison: {result['route']}**\n",
            "| Airline | Price | Confidence |",
            "|---------|-------|------------|"
        ]

        for comp in result["comparisons"]:
            lines.append(
                f"| {comp['airline']} | ${comp['price']:.2f} | {comp['confidence']:.0%} |"
            )

        lines.extend([
            "",
            f"**Cheapest**: {result['cheapest']}",
            f"**Most Expensive**: {result['most_expensive']}"
        ])

        return "\n".join(lines)

    def _format_cheapest_response(self, result: Dict) -> str:
        """Format cheapest destinations response."""
        lines = [
            f"**Cheapest Destinations from {result['origin']} on {result['airline']}**\n",
            "| Destination | Price | Confidence |",
            "|-------------|-------|------------|"
        ]

        for dest in result["destinations"]:
            lines.append(
                f"| {dest['destination']} | ${dest['price']:.2f} | {dest['confidence']:.0%} |"
            )

        return "\n".join(lines)

    def _format_analysis_response(self, result: Dict) -> str:
        """Format route analysis response."""
        return (
            f"**Route Analysis: {result['route']}**\n\n"
            f"Distance: {result['distance_miles']:.0f} miles\n"
            f"Average Price: ${result['average_price']:.2f}\n"
            f"Price Range: ${result['price_range'][0]:.2f} - ${result['price_range'][1]:.2f}\n\n"
            f"**Best Value**: {result['cheapest_option']['airline']} "
            f"at ${result['cheapest_option']['price']:.2f}\n"
            f"**Premium Option**: {result['premium_option']['airline']} "
            f"at ${result['premium_option']['price']:.2f}"
        )

    def _format_airport_info(self, result: Dict) -> str:
        """Format airport info response."""
        return (
            f"**Airport: {result['code']}**\n\n"
            f"Category: {result['tier']}\n"
            f"Pricing Factor: {result['pricing_factor']:.2f}x\n"
            f"Status: {'Valid' if result['is_valid'] else 'Unknown'} airport"
        )

    def _format_airline_info(self, result: Dict) -> str:
        """Format airline info response."""
        return (
            f"**Airline: {result['name']}**\n\n"
            f"Category: {result['tier']}\n"
            f"Pricing Factor: {result['pricing_factor']:.2f}x\n"
            f"Status: {'Valid' if result['is_valid'] else 'Unknown'} airline"
        )

    def _get_help_response(self) -> str:
        """Return help message."""
        return (
            "**Flight Price Agent**\n\n"
            "I can help you with:\n\n"
            "1. **Price Predictions**: 'What's the price from JFK to LAX on Delta?'\n"
            "2. **Airline Comparisons**: 'Compare airlines from SFO to ORD'\n"
            "3. **Find Deals**: 'Cheapest destinations from BOS on JetBlue'\n"
            "4. **Route Analysis**: 'Analyze the route from MIA to SEA'\n"
            "5. **Airport Info**: 'Tell me about JFK airport'\n"
            "6. **Airline Info**: 'Tell me about Southwest'\n\n"
            "Just ask a question about flight prices!"
        )

    def _process_with_llm(self, user_input: str) -> AgentResponse:
        """Process query using LLM for understanding."""
        if not self._llm_client:
            return self._process_locally(user_input)

        # Build messages for LLM
        system_prompt = (
            "You are a flight price prediction assistant. "
            "Use the available tools to answer questions about flight prices. "
            "Always extract airport codes (3 letters) and airline names from queries."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history[-5:],  # Last 5 messages for context
        ]

        try:
            response = self._llm_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=[{"type": "function", "function": schema}
                       for schema in self.tools.get_schemas()],
                tool_choice="auto"
            )

            # Process tool calls if any
            message = response.choices[0].message

            if message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    result = self.tools.execute(func_name, **func_args)
                    tool_results.append({
                        "tool": func_name,
                        "result": result
                    })

                # Get final response with tool results
                # (simplified - full implementation would make another LLM call)
                return self._process_locally(user_input)

            return AgentResponse(
                answer=message.content or "I couldn't process that request.",
                tool_calls=[],
                reasoning="LLM processing",
                confidence=0.8,
                suggestions=[]
            )

        except Exception as e:
            # Fallback to local processing
            return self._process_locally(user_input)

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools.get_all_tools()]
