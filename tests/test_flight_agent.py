"""
Tests for Flight Price AI Agent
===============================

Unit tests for the AI agent and its components.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.flight_agent import FlightPriceAgent, AgentState, AgentResponse
from agents.tools import FlightTools, Tool, ToolCategory
from agents.reasoning import PriceReasoningEngine, PriceFactor
from models.price_predictor import FlightPricePredictor


class TestFlightTools:
    """Tests for FlightTools class."""

    def test_initialization(self):
        """Test tools initialization."""
        tools = FlightTools(predictor=None)

        assert len(tools._tools) > 0

    def test_get_all_tools(self):
        """Test getting all tools."""
        tools = FlightTools(predictor=None)
        all_tools = tools.get_all_tools()

        assert len(all_tools) > 0
        assert all(isinstance(t, Tool) for t in all_tools)

    def test_get_tool_by_name(self):
        """Test getting tool by name."""
        tools = FlightTools(predictor=None)

        tool = tools.get_tool("predict_price")
        assert tool is not None
        assert tool.name == "predict_price"

    def test_get_nonexistent_tool(self):
        """Test getting nonexistent tool returns None."""
        tools = FlightTools(predictor=None)

        tool = tools.get_tool("nonexistent_tool")
        assert tool is None

    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        tools = FlightTools(predictor=None)

        prediction_tools = tools.get_tools_by_category(ToolCategory.PREDICTION)
        assert len(prediction_tools) > 0
        assert all(t.category == ToolCategory.PREDICTION for t in prediction_tools)

    def test_tool_schema_format(self):
        """Test tool schema format."""
        tools = FlightTools(predictor=None)
        schemas = tools.get_schemas()

        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_execute_without_predictor(self):
        """Test execution without predictor returns error."""
        tools = FlightTools(predictor=None)

        result = tools.execute("predict_price", origin="JFK",
                               destination="LAX", airline="Delta")

        assert "error" in result

    def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        tools = FlightTools(predictor=None)

        result = tools.execute("unknown_tool")

        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_missing_parameters(self):
        """Test executing with missing parameters."""
        tools = FlightTools(predictor=None)

        result = tools.execute("predict_price", origin="JFK")  # Missing destination, airline

        assert "error" in result
        assert "Missing" in result["error"]

    def test_get_airport_info(self):
        """Test airport info tool."""
        tools = FlightTools(predictor=None)

        result = tools.execute("get_airport_info", airport_code="JFK")

        assert result["code"] == "JFK"
        assert "pricing_factor" in result
        assert "tier" in result

    def test_get_airline_info(self):
        """Test airline info tool."""
        tools = FlightTools(predictor=None)

        result = tools.execute("get_airline_info", airline="Delta")

        assert result["name"] == "Delta"
        assert "pricing_factor" in result
        assert "tier" in result


class TestPriceReasoningEngine:
    """Tests for PriceReasoningEngine."""

    @pytest.fixture
    def engine(self):
        """Provide reasoning engine."""
        return PriceReasoningEngine(processor=None)

    def test_explain_prediction_basic(self, engine):
        """Test basic explanation generation."""
        explanation = engine.explain_prediction(
            origin="JFK",
            destination="LAX",
            airline="Delta",
            predicted_price=350.0,
            confidence=0.85
        )

        assert explanation.base_price > 0
        assert explanation.final_price == 350.0
        assert len(explanation.factors) > 0
        assert len(explanation.summary) > 0

    def test_explanation_has_all_factors(self, engine):
        """Test that explanation includes all price factors."""
        explanation = engine.explain_prediction(
            origin="JFK",
            destination="LAX",
            airline="Delta",
            predicted_price=350.0,
            confidence=0.85
        )

        factor_names = [f["factor"] for f in explanation.factors]
        assert "distance" in factor_names
        assert "origin_hub" in factor_names
        assert "destination_hub" in factor_names
        assert "airline_tier" in factor_names

    def test_high_confidence_explanation(self, engine):
        """Test explanation for high confidence."""
        explanation = engine.explain_prediction(
            origin="JFK",
            destination="LAX",
            airline="Delta",
            predicted_price=350.0,
            confidence=0.95
        )

        assert "High confidence" in explanation.confidence_explanation

    def test_low_confidence_explanation(self, engine):
        """Test explanation for low confidence."""
        explanation = engine.explain_prediction(
            origin="JFK",
            destination="LAX",
            airline="Delta",
            predicted_price=350.0,
            confidence=0.4
        )

        assert "Lower confidence" in explanation.confidence_explanation

    def test_recommendations_generated(self, engine):
        """Test that recommendations are generated."""
        explanation = engine.explain_prediction(
            origin="JFK",
            destination="LAX",
            airline="Delta",
            predicted_price=350.0,
            confidence=0.85
        )

        assert len(explanation.recommendations) > 0

    def test_compare_options_empty(self, engine):
        """Test comparison with empty list."""
        result = engine.compare_options([])
        assert "No options" in result

    def test_compare_options_basic(self, engine):
        """Test basic comparison."""
        predictions = [
            {"airline": "Delta", "predicted_price": 350.0},
            {"airline": "Spirit", "predicted_price": 150.0},
            {"airline": "United", "predicted_price": 325.0}
        ]

        result = engine.compare_options(predictions)

        assert "Spirit" in result  # Cheapest
        assert "Delta" in result  # Most expensive


class TestFlightPriceAgentInitialization:
    """Tests for FlightPriceAgent initialization."""

    def test_basic_initialization(self):
        """Test basic agent initialization."""
        agent = FlightPriceAgent(predictor=None, use_llm=False)

        assert agent.state == AgentState.IDLE
        assert agent.use_llm is False

    def test_initialization_with_predictor(self):
        """Test initialization with predictor."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=100)

        agent = FlightPriceAgent(predictor=predictor, use_llm=False)

        assert agent.predictor is not None

    def test_get_available_tools(self):
        """Test getting available tools list."""
        agent = FlightPriceAgent(predictor=None, use_llm=False)

        tools = agent.get_available_tools()

        assert len(tools) > 0
        assert "predict_price" in tools


class TestFlightPriceAgentIntentDetection:
    """Tests for intent detection."""

    @pytest.fixture
    def agent(self):
        """Provide a basic agent."""
        return FlightPriceAgent(predictor=None, use_llm=False)

    def test_detect_price_intent(self, agent):
        """Test detection of price intent."""
        intent = agent._detect_intent("what is the price from jfk to lax")
        assert intent == "predict_price"

    def test_detect_cost_intent(self, agent):
        """Test detection of cost intent."""
        intent = agent._detect_intent("how much does it cost to fly to miami")
        assert intent == "predict_price"

    def test_detect_compare_intent(self, agent):
        """Test detection of compare intent."""
        intent = agent._detect_intent("compare airlines from sfo to ord")
        assert intent == "compare_airlines"

    def test_detect_cheapest_intent(self, agent):
        """Test detection of cheapest intent."""
        intent = agent._detect_intent("find cheapest flights from boston")
        assert intent == "find_cheapest"

    def test_detect_analyze_intent(self, agent):
        """Test detection of analyze intent."""
        intent = agent._detect_intent("analyze the route from miami to seattle")
        assert intent == "analyze_route"


class TestFlightPriceAgentEntityExtraction:
    """Tests for entity extraction."""

    @pytest.fixture
    def agent(self):
        """Provide a basic agent."""
        return FlightPriceAgent(predictor=None, use_llm=False)

    def test_extract_airport_codes(self, agent):
        """Test extraction of airport codes."""
        entities = agent._extract_entities("Flight from JFK to LAX")

        assert entities.get("origin") == "JFK"
        assert entities.get("destination") == "LAX"

    def test_extract_airport_names(self, agent):
        """Test extraction of airport names."""
        entities = agent._extract_entities("Flight from new york to los angeles")

        assert entities.get("origin") == "JFK"
        assert entities.get("destination") == "LAX"

    def test_extract_airline(self, agent):
        """Test extraction of airline name."""
        entities = agent._extract_entities("Delta flight from JFK to LAX")

        assert entities.get("airline") == "Delta"

    def test_extract_all_entities(self, agent):
        """Test extraction of all entities."""
        entities = agent._extract_entities("How much is a Delta flight from JFK to LAX")

        assert entities.get("origin") == "JFK"
        assert entities.get("destination") == "LAX"
        assert entities.get("airline") == "Delta"

    def test_extract_lowercase_airline(self, agent):
        """Test extraction of lowercase airline."""
        entities = agent._extract_entities("southwest flight from denver")

        assert entities.get("airline") == "Southwest"


class TestFlightPriceAgentQuery:
    """Tests for query processing."""

    @pytest.fixture
    def trained_agent(self):
        """Provide an agent with trained predictor."""
        predictor = FlightPricePredictor(use_neural_network=False)
        predictor.fit(n_samples=200)
        return FlightPriceAgent(predictor=predictor, use_llm=False)

    def test_query_price_prediction(self, trained_agent):
        """Test price prediction query."""
        response = trained_agent.query("What is the price from JFK to LAX on Delta?")

        assert isinstance(response, AgentResponse)
        assert len(response.answer) > 0
        assert response.confidence > 0

    def test_query_compare_airlines(self, trained_agent):
        """Test airline comparison query."""
        response = trained_agent.query("Compare airlines from SFO to ORD")

        assert isinstance(response, AgentResponse)
        assert "compare" in response.answer.lower() or "airline" in response.answer.lower()

    def test_query_help_for_unclear(self, trained_agent):
        """Test help response for unclear query."""
        response = trained_agent.query("hello there")

        assert isinstance(response, AgentResponse)
        assert "help" in response.answer.lower() or "can" in response.answer.lower()

    def test_query_state_management(self, trained_agent):
        """Test that state returns to idle after query."""
        trained_agent.query("Price from JFK to LAX on Delta")

        assert trained_agent.state == AgentState.IDLE

    def test_query_stores_history(self, trained_agent):
        """Test that queries are stored in history."""
        trained_agent.query("Price from JFK to LAX on Delta")

        assert len(trained_agent._conversation_history) >= 2  # User + assistant

    def test_reset_conversation(self, trained_agent):
        """Test conversation reset."""
        trained_agent.query("Price from JFK to LAX on Delta")
        trained_agent.reset_conversation()

        assert len(trained_agent._conversation_history) == 0


class TestAgentResponseFormat:
    """Tests for AgentResponse format."""

    def test_response_has_required_fields(self):
        """Test that AgentResponse has all required fields."""
        response = AgentResponse(
            answer="Test answer",
            tool_calls=[],
            reasoning="Test reasoning",
            confidence=0.8,
            suggestions=["Suggestion 1"]
        )

        assert response.answer == "Test answer"
        assert response.tool_calls == []
        assert response.reasoning == "Test reasoning"
        assert response.confidence == 0.8
        assert len(response.suggestions) == 1
