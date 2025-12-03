#!/usr/bin/env python3
"""
Advanced simulation with sensitivity analysis.
"""

import numpy as np
from monte_carlo_simulator import (
    MonteCarloSimulator, SimulationConfig, DriverProfile
)


def sensitivity_analysis():
    """Run sensitivity analysis on key parameters."""

    base_config = SimulationConfig(n_simulations=50000)

    # Test different Safety Car probabilities
    sc_probabilities = [0.15, 0.25, 0.31, 0.40, 0.50]

    results = []

    for sc_prob in sc_probabilities:
        print(f"\nRunning simulation with Safety Car probability: {sc_prob:.0%}")

        config = SimulationConfig(
            n_simulations=50000,
            safety_car_probability=sc_prob
        )

        simulator = MonteCarloSimulator(config)

        current_points = {'NORRIS': 408, 'VERSTAPPEN': 396, 'PIASTRI': 392}

        profiles = {
            'NORRIS': DriverProfile(
                name='Lando Norris', team='McLaren',
                base_win_prob=0.25, base_podium_prob=0.55,
                base_dnf_prob=0.02, avg_finish=2.8, consistency_score=0.85
            ),
            'VERSTAPPEN': DriverProfile(
                name='Max Verstappen', team='Red Bull',
                base_win_prob=0.22, base_podium_prob=0.52,
                base_dnf_prob=0.03, avg_finish=3.2, consistency_score=0.82
            ),
            'PIASTRI': DriverProfile(
                name='Oscar Piastri', team='McLaren',
                base_win_prob=0.20, base_podium_prob=0.48,
                base_dnf_prob=0.025, avg_finish=3.7, consistency_score=0.78
            )
        }

        probabilities = simulator.run_simulation(
            driver_profiles=profiles,
            current_points=current_points
        )

        results.append({
            'safety_car_probability': sc_prob,
            'probabilities': probabilities,
            'norris_advantage': probabilities['NORRIS'] - probabilities['VERSTAPPEN']
        })

    # Analyze results
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\nSafety Car Probability: {result['safety_car_probability']:.0%}")
        print(f"  Norris: {result['probabilities']['NORRIS']:.2%}")
        print(f"  Verstappen: {result['probabilities']['VERSTAPPEN']:.2%}")
        print(f"  Norris advantage: {result['norris_advantage']:+.2%}")


def what_if_scenarios():
    """Run 'what-if' scenarios."""

    scenarios = [
        {
            'name': 'Norris DNF',
            'points': {'NORRIS': 408, 'VERSTAPPEN': 396, 'PIASTRI': 392},
            'profiles': {
                'NORRIS': DriverProfile('Norris', 'McLaren', 0.25, 0.55, 1.0, 99, 0.85),
                'VERSTAPPEN': DriverProfile('Verstappen', 'Red Bull', 0.30, 0.60, 0.03, 2.8, 0.85),
                'PIASTRI': DriverProfile('Piastri', 'McLaren', 0.25, 0.55, 0.025, 3.0, 0.80)
            }
        },
        {
            'name': 'Verstappen Qualifying Penalty',
            'points': {'NORRIS': 408, 'VERSTAPPEN': 396, 'PIASTRI': 392},
            'profiles': {
                'NORRIS': DriverProfile('Norris', 'McLaren', 0.30, 0.60, 0.02, 2.5, 0.85),
                'VERSTAPPEN': DriverProfile('Verstappen', 'Red Bull', 0.15, 0.45, 0.03, 5.0, 0.82),
                'PIASTRI': DriverProfile('Piastri', 'McLaren', 0.25, 0.55, 0.025, 3.0, 0.80)
            }
        }
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 40)

        simulator = MonteCarloSimulator(SimulationConfig(n_simulations=50000))
        probabilities = simulator.run_simulation(
            driver_profiles=scenario['profiles'],
            current_points=scenario['points']
        )

        for driver, prob in probabilities.items():
            print(f"  {driver}: {prob:.2%}")


if __name__ == "__main__":
    print("Running sensitivity analysis...")
    sensitivity_analysis()

    print("\n\nRunning 'what-if' scenarios...")
    what_if_scenarios()
