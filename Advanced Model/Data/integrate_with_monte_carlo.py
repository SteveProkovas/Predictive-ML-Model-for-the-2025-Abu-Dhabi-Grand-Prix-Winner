#!/usr/bin/env python3
"""
Integrate historical patterns with Monte Carlo simulator.
"""

import json
import pandas as pd
from historical_analyzer import HistoricalPatternAnalyzer
from monte_carlo_simulator import MonteCarloSimulator, SimulationConfig


def create_historically_informed_config() -> SimulationConfig:
    """Create simulation configuration informed by historical patterns."""

    # Load historical patterns
    with open("data/historical_patterns.json", 'r') as f:
        patterns = json.load(f)

    # Create configuration with historical probabilities
    config = SimulationConfig(
        n_simulations=100000,
        safety_car_probability=patterns['safety_car_statistics']['safety_car_rate'],
        include_historical_events=True,
        random_seed=42
    )

    return config


def adjust_profiles_with_history(driver_profiles, historical_patterns):
    """Adjust driver profiles based on historical patterns."""

    # Get gap-based adjustment
    current_gap = 12  # Norris's lead
    gap_success = historical_patterns['gap_success_rates']

    # Find appropriate gap category
    if current_gap == 0:
        adjustment_factor = gap_success.get('Tied', 0.5)
    elif current_gap <= 5:
        adjustment_factor = gap_success.get('1-5 pts', 0.55)
    elif current_gap <= 10:
        adjustment_factor = gap_success.get('6-10 pts', 0.65)
    elif current_gap <= 20:
        adjustment_factor = gap_success.get('11-20 pts', 0.75)
    else:
        adjustment_factor = gap_success.get('50+ pts', 0.9)

    # Adjust leader's probabilities
    for driver, profile in driver_profiles.items():
        if driver == 'NORRIS':  # Current leader
            # Boost win probability based on historical success
            profile.base_win_prob *= (1 + (adjustment_factor - 0.5))
        elif driver == 'VERSTAPPEN':  # Main challenger
            # Slightly reduce due to historical disadvantage
            profile.base_win_prob *= (1 - (adjustment_factor - 0.5) * 0.3)

    return driver_profiles


def main():
    print("Integrating Historical Patterns with Monte Carlo Simulator")
    print("=" * 60)

    # Load historical data
    df = pd.read_csv("data/historical_deciders_raw.csv")
    analyzer = HistoricalPatternAnalyzer(df)
    patterns = analyzer.analyze_all_patterns()

    # Create historically-informed configuration
    config = create_historically_informed_config()
    print(f"Safety Car probability from history: {config.safety_car_probability:.1%}")

    # Get event probabilities for simulation
    event_probs = analyzer._get_event_adjustments()
    print(f"\nEvent probabilities for simulation:")
    for event, prob in event_probs.items():
        print(f"  {event:20} {prob:.1%}")

    # Generate recommendations for current gap
    recommendations = analyzer.generate_recommendations(12)
    print(f"\n📊 For 12-point lead (2025 scenario):")
    print(f"   Historical success rate: {recommendations['based_on_gap']['historical_success_rate']:.1%}")
    print(f"   Confidence: {recommendations['historical_confidence']:.1%}")

    # Create Monte Carlo simulator with historical patterns
    simulator = MonteCarloSimulator(config)

    print("\n✅ Historical patterns integrated successfully!")
    print("   Use these patterns in your Monte Carlo simulation for more accurate predictions.")


if __name__ == "__main__":
    main()
