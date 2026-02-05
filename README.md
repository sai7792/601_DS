# Flight Price Prediction System

A modern ML-powered flight price prediction system using AI agents, ensemble models, and neural networks.

## Overview

Predict flight prices based on:
- **Departure airport** (e.g., JFK, LAX, ORD)
- **Arrival airport** (e.g., SFO, MIA, ATL)
- **Airline** (e.g., Delta, United, Southwest)

No date/time required - predictions based on route characteristics and airline pricing patterns.

## Features

| Feature | Description |
|---------|-------------|
| **Ensemble ML Models** | Stacked XGBoost, LightGBM, CatBoost, Random Forest |
| **Neural Network** | PyTorch-based deep learning model |
| **AI Agent** | Natural language queries for price predictions |
| **Feature Engineering** | Route analysis, hub detection, regional pricing |
| **Data Validation** | Pydantic models for type-safe inputs |

## Installation

```bash
# Clone the repository
git clone https://github.com/sai7792/601_DS.git
cd 601_DS

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

## Quick Start

### Basic Price Prediction

```python
from src.models.price_predictor import FlightPricePredictor

# Initialize and train
predictor = FlightPricePredictor(use_neural_network=False)
predictor.fit(n_samples=2000)

# Predict a flight price
result = predictor.predict("JFK", "LAX", "Delta")
print(f"Price: ${result.predicted_price:.2f}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Range: ${result.price_range[0]:.2f} - ${result.price_range[1]:.2f}")
```

### Compare Airlines

```python
# Compare all airlines for a route
comparisons = predictor.compare_airlines("SFO", "ORD")
for pred in comparisons:
    print(f"{pred.airline}: ${pred.predicted_price:.2f}")
```

### AI Agent (Natural Language)

```python
from src.agents.flight_agent import FlightPriceAgent

agent = FlightPriceAgent(predictor=predictor)

# Ask questions in natural language
response = agent.query("What's the price from JFK to LAX on Delta?")
print(response.answer)

response = agent.query("Compare airlines from Boston to Miami")
print(response.answer)

response = agent.query("Find cheapest flights from Seattle on Alaska")
print(response.answer)
```

### Run Demo

```bash
python demo.py
```

## Project Structure

```
601_DS/
├── src/
│   ├── data/
│   │   ├── processor.py      # Data loading & preprocessing
│   │   └── validator.py      # Input validation (Pydantic)
│   ├── models/
│   │   ├── ensemble.py       # Ensemble ML models
│   │   ├── neural_network.py # PyTorch neural network
│   │   ├── price_predictor.py# Main predictor interface
│   │   └── base.py           # Base model class
│   ├── agents/
│   │   ├── flight_agent.py   # AI agent for NL queries
│   │   ├── tools.py          # Agent tools/functions
│   │   └── reasoning.py      # Price reasoning engine
│   └── features/
│       └── engineering.py    # Feature engineering pipeline
├── tests/                    # 164 comprehensive tests
├── demo.py                   # Demo script
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── pytest.ini               # Test configuration
```

## Supported Airports

| Region | Airports |
|--------|----------|
| Northeast | JFK, BOS, EWR, PHL, BWI |
| Southeast | ATL, MIA, CLT, TPA, MCO |
| Midwest | ORD, DTW, MSP, STL, IND, MCI |
| Southwest | DFW, IAH, AUS, PHX, LAS |
| West | LAX, SFO, SEA, PDX, SAN, DEN, SLC |

## Supported Airlines

- **Premium**: Delta, United, American, Alaska, JetBlue
- **Budget**: Southwest, Spirit, Frontier, Allegiant

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_price_predictor.py -v
```

### Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| Data Processor | 23 | Data generation, preprocessing |
| Data Validator | 25 | Input validation, error handling |
| Ensemble Model | 22 | Training, prediction, evaluation |
| Feature Engineering | 21 | Region/route features |
| Flight Agent | 32 | NL understanding, tool execution |
| Price Predictor | 22 | End-to-end prediction |
| Integration | 19 | Full pipeline tests |
| **Total** | **164** | |

## API Reference

### FlightPricePredictor

```python
predictor = FlightPricePredictor(use_neural_network=True)
predictor.fit(data_path=None, n_samples=5000)
result = predictor.predict(origin, destination, airline)
comparisons = predictor.compare_airlines(origin, destination)
cheapest = predictor.get_cheapest_route(origin, airline)
predictor.save("model.pkl")
predictor = FlightPricePredictor.load("model.pkl")
```

### FlightPriceAgent

```python
agent = FlightPriceAgent(predictor=predictor, use_llm=False)
response = agent.query("Your question here")
# response.answer - The answer text
# response.confidence - Prediction confidence
# response.suggestions - Follow-up suggestions
```

## Model Architecture

### Ensemble Model
- Random Forest (n_estimators=100)
- Gradient Boosting (n_estimators=100)
- XGBoost (if available)
- LightGBM (if available)
- CatBoost (if available)
- Meta-learner: Ridge Regression (stacking)

### Neural Network
- Input layer → 64 units (ReLU) → Dropout(0.2)
- Hidden layer → 32 units (ReLU) → Dropout(0.2)
- Output layer → 1 unit (price)

### Features Used
- Origin/destination airport tiers
- Airline pricing tiers
- Route distance (miles)
- Hub connectivity scores
- Regional indicators
- Transcontinental flag

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python -m pytest tests/ -v`
4. Submit a pull request
