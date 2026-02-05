#!/usr/bin/env python3
"""
Flight Price Prediction Demo
============================

Demonstrates the flight price prediction system with AI agents.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.processor import FlightDataProcessor
from data.validator import FlightDataValidator
from models.price_predictor import FlightPricePredictor
from agents.flight_agent import FlightPriceAgent


def main():
    print("=" * 60)
    print("Flight Price Prediction System Demo")
    print("=" * 60)
    print()

    # 1. Data Generation
    print("1. Generating synthetic flight data...")
    processor = FlightDataProcessor()
    df = processor.create_sample_data(n_samples=1000, seed=42)
    print(f"   Generated {len(df)} flight records")
    print(f"   Sample data:")
    print(df.head().to_string(index=False))
    print()

    # 2. Train Price Predictor
    print("2. Training price prediction models...")
    predictor = FlightPricePredictor(use_neural_network=False)
    predictor.fit(n_samples=2000)
    print("   Training complete!")

    metrics = predictor.get_training_metrics()
    print(f"   Ensemble MAE: ${metrics['ensemble']['mae']:.2f}")
    print(f"   Ensemble R²: {metrics['ensemble']['r2']:.3f}")
    print()

    # 3. Single Predictions
    print("3. Price Predictions:")
    routes = [
        ("JFK", "LAX", "Delta"),
        ("JFK", "LAX", "Spirit"),
        ("SFO", "ORD", "United"),
        ("MIA", "ATL", "Southwest"),
    ]

    for origin, dest, airline in routes:
        result = predictor.predict(origin, dest, airline)
        print(f"   {origin} → {dest} on {airline}: "
              f"${result.predicted_price:.2f} "
              f"(confidence: {result.confidence:.0%})")
    print()

    # 4. Airline Comparison
    print("4. Comparing Airlines (JFK → LAX):")
    comparison = predictor.compare_airlines("JFK", "LAX")
    for pred in comparison[:5]:
        print(f"   {pred.airline}: ${pred.predicted_price:.2f}")
    print()

    # 5. AI Agent Demo
    print("5. AI Agent Natural Language Queries:")
    agent = FlightPriceAgent(predictor=predictor, use_llm=False)

    queries = [
        "What's the price from JFK to LAX on Delta?",
        "Compare airlines from SFO to Chicago",
        "Tell me about Spirit airline",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        response = agent.query(query)
        # Print first 200 chars of answer
        answer_preview = response.answer[:300] + "..." if len(response.answer) > 300 else response.answer
        print(f"   Response: {answer_preview}")
    print()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
