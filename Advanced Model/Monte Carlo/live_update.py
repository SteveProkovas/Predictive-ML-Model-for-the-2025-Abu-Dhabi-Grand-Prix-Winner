"""
Update simulation with real-time qualifying/practice data.
"""

import json
from datetime import datetime
from monte_carlo_simulator import (
    MonteCarloSimulator, SimulationConfig, DriverProfile
)


def update_with_qualifying_results(qualifying_results: Dict[str, int]):
    """
    Update driver profiles based on qualifying results.

    Args:
        qualifying_results: Dictionary of grid positions (1-20)
    """

    # Base profiles
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

    # Adjust based on grid position
    grid_position_weights = {
        1: 1.2,  # Pole position boost
        2: 1.1,
        3: 1.05,
        4: 1.0,
        5: 0.95,
        6: 0.9,
        7: 0.85,
        8: 0.8,
        9: 0.75,
        10: 0.7,
        11: 0.65,
        12: 0.6,
        13: 0.55,
        14: 0.5,
        15: 0.45,
        16: 0.4,
        17: 0.35,
        18: 0.3,
        19: 0.25,
        20: 0.2
    }

    for driver, grid_pos in qualifying_results.items():
        if driver in profiles:
            weight = grid_position_weights.get(grid_pos, 0.5)
            profiles[driver].base_win_prob *= weight
            profiles[driver].base_podium_prob *= min(1.0, weight * 1.1)

    return profiles


def main():
    # Load qualifying results (example)
    qualifying_results = {
        'NORRIS': 1,  # Pole position
        'VERSTAPPEN': 3,  # 3rd on grid
        'PIASTRI': 2  # 2nd on grid
    }

    print(f"Qualifying Results: {qualifying_results}")
    print("Updating simulation with qualifying data...")

    # Update profiles
    updated_profiles = update_with_qualifying_results(qualifying_results)

    # Run simulation
    config = SimulationConfig(n_simulations=100000)
    simulator = MonteCarloSimulator(config)

    current_points = {'NORRIS': 408, 'VERSTAPPEN': 396, 'PIASTRI': 392}

    probabilities = simulator.run_simulation(
        driver_profiles=updated_profiles,
        current_points=current_points
    )

    # Display results
    print("\nPOST-QUALIFYING PREDICTION:")
    print("=" * 40)
    for driver, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{driver:12} {prob:6.2%}")

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulator.save_results(f"results/post_qualifying_{timestamp}")


if __name__ == "__main__":
    main()
