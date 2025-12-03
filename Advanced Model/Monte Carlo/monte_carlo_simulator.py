"""
F1 CHAMPIONSHIP MONTE CARLO SIMULATOR
Core simulation engine for predicting the 2025 Abu Dhabi Grand Prix championship decider.
Integrates historical patterns, driver performance profiles, and championship rules.
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import json
from pathlib import Path
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




class OptimizedMonteCarloSimulator(MonteCarloSimulator):
    """Optimized version with hardware-specific improvements."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.batch_size = 10000  # Process simulations in batches

    def _run_optimized_simulation(self, profiles, current_points):
        """Vectorized simulation for better performance."""
        n_drivers = len(profiles)
        n_positions = 21

        # Pre-calculate cumulative distributions for all drivers
        cum_probs = {}
        profile_arrays = {}
        for driver, profile in profiles.items():
            profile_arrays[driver] = profile
            cum_probs[driver] = np.cumsum(profile)

        # Batch processing
        wins = {driver: 0 for driver in profiles}

        for batch_start in range(0, self.config.n_simulations, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.config.n_simulations)
            batch_size = batch_end - batch_start

            # Vectorized random sampling
            random_samples = np.random.random((batch_size, n_drivers))

            # Find positions for each driver in batch
            positions = {}
            for i, driver in enumerate(profiles.keys()):
                # Find position index for each random sample
                samples = random_samples[:, i]
                # Vectorized searchsorted
                pos_indices = np.searchsorted(cum_probs[driver], samples)
                positions[driver] = pos_indices

            # Process batch
            for i in range(batch_size):
                race_positions = {}
                for driver in profiles.keys():
                    pos_idx = positions[driver][i]
                    race_positions[driver] = 99 if pos_idx == 20 else pos_idx + 1

                # Determine champion
                champion = self._determine_champion_batch(race_positions, current_points)
                wins[champion] += 1

        return wins

    def _determine_champion_batch(self, positions, current_points):
        """Optimized champion determination."""
        # Simplified logic for batch processing
        pia_pos = positions.get('PIASTRI', 99)
        nor_pos = positions.get('NORRIS', 99)
        ver_pos = positions.get('VERSTAPPEN', 99)

        # Quick checks for common scenarios
        if nor_pos in [1, 2, 3]:
            return 'NORRIS'
        if ver_pos == 1 and nor_pos >= 4:
            return 'VERSTAPPEN'
        if pia_pos == 1 and nor_pos >= 6 and ver_pos >= 2:
            return 'PIASTRI'

        # Fall back to points calculation
        points_system = self.config.points_system
        pia_points = current_points['PIASTRI'] + points_system.get(pia_pos, 0)
        nor_points = current_points['NORRIS'] + points_system.get(nor_pos, 0)
        ver_points = current_points['VERSTAPPEN'] + points_system.get(ver_pos, 0)

        if ver_points > nor_points and ver_points > pia_points:
            return 'VERSTAPPEN'
        elif nor_points > ver_points and nor_points > pia_points:
            return 'NORRIS'
        else:
            return 'PIASTRI'

@dataclass
class SimulationConfig:
    """Configuration for the Monte Carlo simulation."""
    n_simulations: int = 100000
    safety_car_probability: float = 0.31
    include_historical_events: bool = True
    random_seed: Optional[int] = 42
    track_name: str = "Yas Marina Circuit"
    points_system: Dict[int, int] = None

    def __post_init__(self):
        if self.points_system is None:
            self.points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)


@dataclass
class DriverProfile:
    """Represents a driver's performance characteristics."""
    name: str
    team: str
    base_win_prob: float
    base_podium_prob: float
    base_dnf_prob: float
    avg_finish: float
    consistency_score: float  # 0-1, higher = more consistent
    track_specific_boost: float = 0.0  # +/- boost at Yas Marina
    momentum_factor: float = 1.0  # Recent form multiplier

    def to_probability_array(self) -> np.ndarray:
        """Convert profile to position probability distribution (1-20 + DNF)."""
        probs = np.zeros(21)

        # 1. Win probability (P1)
        win_prob = min(0.95, self.base_win_prob * self.momentum_factor + self.track_specific_boost)
        probs[0] = win_prob

        # 2. Podium probabilities (P2-P5)
        podium_total = self.base_podium_prob - win_prob
        podium_positions = 4

        # Create tapered distribution: P2 > P3 > P4 > P5
        podium_weights = np.array([0.35, 0.30, 0.20, 0.15])
        podium_weights = podium_weights / podium_weights.sum()

        for i in range(podium_positions):
            probs[1 + i] = podium_total * podium_weights[i]

        # 3. Points finishes (P6-P10)
        points_prob = 0.85 - self.base_podium_prob  # Points finish probability
        points_positions = 5
        points_weights = np.array([0.25, 0.22, 0.20, 0.18, 0.15])
        points_weights = points_weights / points_weights.sum()

        for i in range(points_positions):
            probs[5 + i] = points_prob * points_weights[i]

        # 4. Non-points finishes (P11-P15)
        non_points_prob = 0.95 - (0.85 + probs[20])  # Top 15 probability
        non_points_positions = 5
        if non_points_prob > 0:
            non_points_per_pos = non_points_prob / non_points_positions
            probs[10:15] = non_points_per_pos

        # 5. Backmarker finishes (P16-P20)
        remaining_prob = 1 - self.base_dnf_prob - probs.sum()
        backmarker_positions = 5

        if remaining_prob > 0:
            backmarker_per_pos = remaining_prob / backmarker_positions
            probs[15:20] = backmarker_per_pos

        # 6. DNF probability
        probs[20] = self.base_dnf_prob

        # Normalize to ensure sum = 1
        probs = probs / probs.sum()

        return probs


class ChampionshipRules:
    """Encodes all championship decision logic from the scenario screenshots."""

    def __init__(self, current_points: Dict[str, int]):
        self.current_points = current_points
        self.points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

    def determine_champion(self, positions: Dict[str, int]) -> str:
        """
        Determine 2025 champion based on finishing positions.
        Positions: 1-20 for finishing positions, >20 for DNF.
        """
        pia_pos = positions.get('PIASTRI', 99)
        nor_pos = positions.get('NORRIS', 99)
        ver_pos = positions.get('VERSTAPPEN', 99)

        # Convert DNF to 99 for processing
        pia_pos = 99 if pia_pos > 20 else pia_pos
        nor_pos = 99 if nor_pos > 20 else nor_pos
        ver_pos = 99 if ver_pos > 20 else ver_pos

        # 1. OSCAR PIASTRI's scenarios
        if pia_pos == 1 and (nor_pos >= 6 or nor_pos == 99) and (ver_pos >= 2 or ver_pos == 99):
            return 'PIASTRI'
        if pia_pos == 2 and (nor_pos >= 10 or nor_pos == 99) and (ver_pos >= 4 or ver_pos == 99):
            return 'PIASTRI'

        # 2. LANDO NORRIS's scenarios
        if nor_pos in [1, 2, 3]:
            return 'NORRIS'
        if nor_pos == 4 and (ver_pos >= 2 or ver_pos == 99):
            return 'NORRIS'
        if nor_pos == 5 and (ver_pos >= 2 or ver_pos == 99):
            return 'NORRIS'
        if nor_pos == 6 and (ver_pos >= 2 or ver_pos == 99) and (pia_pos >= 2 or pia_pos == 99):
            return 'NORRIS'
        if nor_pos == 7 and (ver_pos >= 2 or ver_pos == 99) and (pia_pos >= 2 or pia_pos == 99):
            return 'NORRIS'
        if nor_pos == 8 and (ver_pos >= 3 or ver_pos == 99) and (pia_pos >= 2 or pia_pos == 99):
            return 'NORRIS'
        if nor_pos == 9 and (ver_pos >= 4 or ver_pos == 99) and (pia_pos >= 2 or pia_pos == 99):
            return 'NORRIS'
        if nor_pos == 10 and (ver_pos >= 4 or ver_pos == 99) and (pia_pos >= 3 or pia_pos == 99):
            return 'NORRIS'
        if nor_pos == 11 and (ver_pos >= 4 or ver_pos == 99) and (pia_pos >= 3 or pia_pos == 99):
            return 'NORRIS'

        # 3. MAX VERSTAPPEN's scenarios
        if ver_pos == 1 and (nor_pos >= 4 or nor_pos == 99):
            return 'VERSTAPPEN'
        if ver_pos == 2 and (nor_pos >= 8 or nor_pos == 99) and (pia_pos >= 3 or pia_pos == 99):
            return 'VERSTAPPEN'
        if ver_pos == 3 and (nor_pos >= 9 or nor_pos == 99) and (pia_pos >= 2 or pia_pos == 99):
            return 'VERSTAPPEN'

        # 4. Calculate points if no scenario met
        pia_points = self.current_points['PIASTRI'] + self.points_system.get(pia_pos, 0)
        nor_points = self.current_points['NORRIS'] + self.points_system.get(nor_pos, 0)
        ver_points = self.current_points['VERSTAPPEN'] + self.points_system.get(ver_pos, 0)

        if ver_points > nor_points and ver_points > pia_points:
            return 'VERSTAPPEN'
        elif nor_points > ver_points and nor_points > pia_points:
            return 'NORRIS'
        elif pia_points > ver_points and pia_points > nor_points:
            return 'PIASTRI'
        else:
            # Tie - use countback (wins, then 2nd places, etc.)
            # For simplicity, return current leader
            return max(self.current_points, key=self.current_points.get)


class RaceEventSimulator:
    """Simulates race events (Safety Car, incidents, etc.) based on historical probabilities."""

    def __init__(self, config: SimulationConfig, historical_event_probs: Dict[str, float]):
        self.config = config
        self.historical_probs = historical_event_probs

        # Event definitions
        self.events = {
            'safety_car': {
                'probability': config.safety_car_probability,
                'effect': self._apply_safety_car_effect,
                'description': 'Safety Car deployment'
            },
            'mechanical_failure': {
                'probability': historical_event_probs.get('mechanical_failure', 0.087),
                'effect': self._apply_mechanical_failure,
                'description': 'Mechanical DNF'
            },
            'first_lap_incident': {
                'probability': historical_event_probs.get('first_lap_incident', 0.043),
                'effect': self._apply_first_lap_incident,
                'description': 'First lap collision'
            },
            'strategy_gamble': {
                'probability': historical_event_probs.get('strategy_gamble', 0.043),
                'effect': self._apply_strategy_gamble,
                'description': 'Alternative strategy'
            }
        }

    def simulate_race_events(self) -> Dict[str, bool]:
        """Determine which events occur in this race simulation."""
        events_occurred = {}
        for event_name, event_data in self.events.items():
            events_occurred[event_name] = np.random.random() < event_data['probability']
        return events_occurred

    def apply_event_effects(self, events_occurred: Dict[str, bool],
                            driver_profiles: Dict[str, np.ndarray],
                            current_points: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Apply event effects to driver probability profiles."""
        adjusted_profiles = {driver: profile.copy() for driver, profile in driver_profiles.items()}

        for event_name, occurred in events_occurred.items():
            if occurred and event_name in self.events:
                effect_func = self.events[event_name]['effect']
                adjusted_profiles = effect_func(adjusted_profiles, current_points)

        return adjusted_profiles

    def _apply_safety_car_effect(self, profiles: Dict[str, np.ndarray],
                                 current_points: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Safety Car increases randomness and can shuffle positions."""
        adjusted = {}

        for driver, profile in profiles.items():
            # Safety Car flattens distribution (increases variance)
            flattened = np.power(profile, 0.7)  # Exponent < 1 flattens distribution
            flattened = flattened / flattened.sum()

            # Slight benefit to trailing drivers
            if driver != max(current_points, key=current_points.get):
                # Trailing drivers get small boost to win probability
                win_boost = 0.03
                flattened[0] = min(0.95, flattened[0] * (1 + win_boost))
                # Redistribute from middle positions
                flattened[5:15] *= 0.97
                flattened = flattened / flattened.sum()

            adjusted[driver] = flattened

        return adjusted

    def _apply_mechanical_failure(self, profiles: Dict[str, np.ndarray],
                                  current_points: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Random mechanical failure for a top contender."""
        # Higher probability for cars that have been pushed hard
        contenders = list(current_points.keys())
        failure_probabilities = [0.15, 0.10, 0.08]  # Leader, 2nd, 3rd

        for driver, prob in zip(contenders, failure_probabilities):
            if np.random.random() < prob:
                # Force DNF for this driver
                profiles[driver] = np.zeros(21)
                profiles[driver][20] = 1.0
                break

        return profiles

    def _apply_first_lap_incident(self, profiles: Dict[str, np.ndarray],
                                  current_points: Dict[str, int]) -> Dict[str, np.ndarray]:
        """First lap incident affecting multiple cars."""
        # Randomly select 1-3 drivers to be affected
        drivers = list(profiles.keys())
        n_affected = np.random.randint(1, 4)
        affected_drivers = np.random.choice(drivers, n_affected, replace=False)

        for driver in affected_drivers:
            # 50% chance of DNF, 50% chance of dropping to back
            if np.random.random() < 0.5:
                profiles[driver][20] += 0.5  # DNF probability increase
            else:
                # Shift probability to back of grid
                back_boost = np.sum(profiles[driver][:10]) * 0.3
                profiles[driver][:10] *= 0.7
                profiles[driver][15:20] += back_boost / 5

            profiles[driver] = profiles[driver] / profiles[driver].sum()

        return profiles

    def _apply_strategy_gamble(self, profiles: Dict[str, np.ndarray],
                               current_points: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Alternative strategy (e.g., early pit stop, different tyre choice)."""
        # Randomly select one driver for strategy gamble
        drivers = list(profiles.keys())
        gambler = np.random.choice(drivers)

        # Strategy can either work brilliantly or fail miserably
        if np.random.random() < 0.4:  # 40% chance of success
            # Brilliant strategy: boost to podium positions
            success_boost = 0.15
            profiles[gambler][:3] *= (1 + success_boost)
            profiles[gambler][10:] *= 0.8  # Reduce backmarker probability
        else:
            # Failed strategy: drop down order
            failure_penalty = 0.2
            profiles[gambler][:5] *= (1 - failure_penalty)
            profiles[gambler][10:] *= (1 + failure_penalty)

        profiles[gambler] = profiles[gambler] / profiles[gambler].sum()
        return profiles


class MonteCarloSimulator:
    """Main Monte Carlo simulation engine for F1 championship prediction."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = None
        self.event_tracker = defaultdict(int)
        self.runtime_stats = {}

        logger.info(f"Initialized Monte Carlo Simulator with {config.n_simulations:,} simulations")
        logger.info(f"Safety Car probability: {config.safety_car_probability:.1%}")

    def run_simulation(self,
                       driver_profiles: Dict[str, DriverProfile],
                       current_points: Dict[str, int],
                       historical_event_probs: Dict[str, float] = None) -> Dict[str, float]:
        """
        Run the Monte Carlo simulation.

        Args:
            driver_profiles: Dictionary of DriverProfile objects
            current_points: Current championship points
            historical_event_probs: Historical event probabilities

        Returns:
            Dictionary of championship probabilities for each driver
        """
        logger.info("Starting Monte Carlo simulation...")
        start_time = time.time()

        # Initialize components
        rules = ChampionshipRules(current_points)
        event_simulator = RaceEventSimulator(self.config, historical_event_probs or {})

        # Convert profiles to probability arrays
        base_prob_arrays = {name: profile.to_probability_array()
                            for name, profile in driver_profiles.items()}

        # Track results
        win_counts = {driver: 0 for driver in driver_profiles.keys()}
        position_counts = {driver: [0] * 21 for driver in driver_profiles.keys()}

        # Progress tracking
        checkpoint = max(1, self.config.n_simulations // 10)

        for sim_idx in range(self.config.n_simulations):
            # Simulate race events
            events_occurred = event_simulator.simulate_race_events()

            # Apply event effects to profiles
            if self.config.include_historical_events:
                race_profiles = event_simulator.apply_event_effects(
                    events_occurred, base_prob_arrays, current_points
                )
            else:
                race_profiles = base_prob_arrays.copy()

            # Sample finishing positions for each driver
            positions = {}
            for driver, profile in race_profiles.items():
                # Sample position (0-20, where 20 = DNF)
                pos_idx = np.random.choice(range(21), p=profile)
                position_counts[driver][pos_idx] += 1

                # Convert to race position (1-20, DNF = 99)
                race_pos = 99 if pos_idx == 20 else pos_idx + 1
                positions[driver] = race_pos

            # Track events
            for event, occurred in events_occurred.items():
                if occurred:
                    self.event_tracker[event] += 1

            # Determine champion
            champion = rules.determine_champion(positions)
            win_counts[champion] += 1

            # Progress update
            if (sim_idx + 1) % checkpoint == 0:
                progress = (sim_idx + 1) / self.config.n_simulations * 100
                logger.info(f"Progress: {progress:.0f}% ({sim_idx + 1:,}/{self.config.n_simulations:,} simulations)")

        # Calculate final probabilities
        total_sims = self.config.n_simulations
        probabilities = {driver: count / total_sims for driver, count in win_counts.items()}

        # Calculate expected finishing positions
        expected_positions = {}
        for driver, counts in position_counts.items():
            expected = sum((i + 1) * count for i, count in enumerate(counts[:20])) / total_sims
            expected_positions[driver] = expected

        # Store results
        self.results = {
            'probabilities': probabilities,
            'win_counts': win_counts,
            'expected_positions': expected_positions,
            'position_distributions': position_counts,
            'event_probabilities': {event: count / total_sims
                                    for event, count in self.event_tracker.items()}
        }

        # Record runtime stats
        end_time = time.time()
        self.runtime_stats = {
            'total_simulations': total_sims,
            'execution_time': end_time - start_time,
            'simulations_per_second': total_sims / (end_time - start_time),
            'memory_usage_mb': self._get_memory_usage()
        }

        logger.info(f"Simulation completed in {self.runtime_stats['execution_time']:.2f} seconds")
        logger.info(f"Speed: {self.runtime_stats['simulations_per_second']:.0f} sims/second")

        return probabilities

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def generate_report(self) -> Dict:
        """Generate comprehensive simulation report."""
        if self.results is None:
            raise ValueError("No simulation results available. Run simulation first.")

        report = {
            'simulation_config': {
                'n_simulations': self.config.n_simulations,
                'safety_car_probability': self.config.safety_car_probability,
                'include_historical_events': self.config.include_historical_events,
                'random_seed': self.config.random_seed,
                'track': self.config.track_name
            },
            'results': self.results,
            'runtime_stats': self.runtime_stats,
            'summary': self._generate_summary()
        }

        return report

    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        probs = self.results['probabilities']
        champion = max(probs, key=probs.get)

        summary = {
            'predicted_champion': champion,
            'champion_probability': probs[champion],
            'closest_contender': None,
            'probability_gap': None,
            'key_insights': []
        }

        # Find closest contender
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) >= 2:
            summary['closest_contender'] = sorted_probs[1][0]
            summary['probability_gap'] = sorted_probs[0][1] - sorted_probs[1][1]

        # Generate insights
        if summary['probability_gap'] and summary['probability_gap'] > 0.2:
            summary['key_insights'].append(
                f"{champion} is a clear favorite with {probs[champion]:.1%} probability"
            )
        else:
            summary['key_insights'].append(
                "Championship is highly competitive with no clear favorite"
            )

        # Event insights
        event_probs = self.results['event_probabilities']
        if event_probs.get('safety_car', 0) > 0.25:
            summary['key_insights'].append(
                f"High Safety Car probability ({event_probs['safety_car']:.1%}) increases outcome variance"
            )

        return summary

    def save_results(self, output_dir: str = "results"):
        """Save simulation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        report = self.generate_report()
        report_file = output_path / f"simulation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save CSV with detailed results
        csv_data = []
        for driver, prob in report['results']['probabilities'].items():
            row = {
                'driver': driver,
                'championship_probability': prob,
                'expected_position': report['results']['expected_positions'][driver],
                'wins_in_simulation': report['results']['win_counts'][driver]
            }
            csv_data.append(row)

        df_results = pd.DataFrame(csv_data)
        csv_file = output_path / f"simulation_results_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False)

        # Save position distributions
        dist_data = []
        for driver, counts in report['results']['position_distributions'].items():
            for pos_idx, count in enumerate(counts):
                position = "DNF" if pos_idx == 20 else str(pos_idx + 1)
                dist_data.append({
                    'driver': driver,
                    'position': position,
                    'count': count,
                    'probability': count / self.config.n_simulations
                })

        df_dist = pd.DataFrame(dist_data)
        dist_file = output_path / f"position_distributions_{timestamp}.csv"
        df_dist.to_csv(dist_file, index=False)

        logger.info(f"Results saved to {output_path}/")
        logger.info(f"  - Report: {report_file.name}")
        logger.info(f"  - Results: {csv_file.name}")
        logger.info(f"  - Distributions: {dist_file.name}")

        return {
            'report_file': str(report_file),
            'results_file': str(csv_file),
            'distribution_file': str(dist_file)
        }


# Helper functions for common use cases
def create_default_driver_profiles() -> Dict[str, DriverProfile]:
    """Create default driver profiles based on 2025 data."""
    return {
        'NORRIS': DriverProfile(
            name='Lando Norris',
            team='McLaren',
            base_win_prob=0.25,
            base_podium_prob=0.55,
            base_dnf_prob=0.02,
            avg_finish=2.8,
            consistency_score=0.85,
            track_specific_boost=0.03,
            momentum_factor=1.1
        ),
        'VERSTAPPEN': DriverProfile(
            name='Max Verstappen',
            team='Red Bull',
            base_win_prob=0.22,
            base_podium_prob=0.52,
            base_dnf_prob=0.03,
            avg_finish=3.2,
            consistency_score=0.82,
            track_specific_boost=0.05,
            momentum_factor=0.95
        ),
        'PIASTRI': DriverProfile(
            name='Oscar Piastri',
            team='McLaren',
            base_win_prob=0.20,
            base_podium_prob=0.48,
            base_dnf_prob=0.025,
            avg_finish=3.7,
            consistency_score=0.78,
            track_specific_boost=0.02,
            momentum_factor=1.05
        )
    }


def load_historical_event_probs(filepath: str = "data/historical_events.json") -> Dict[str, float]:
    """Load historical event probabilities from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Historical event file not found: {filepath}. Using defaults.")
        return {
            'mechanical_failure': 0.087,
            'first_lap_incident': 0.043,
            'strategy_gamble': 0.043,
            'controversial_finish': 0.043
        }


def print_simulation_summary(results: Dict):
    """Print a formatted summary of simulation results."""
    print("=" * 70)
    print("F1 CHAMPIONSHIP SIMULATION RESULTS")
    print("=" * 70)

    if 'probabilities' in results:
        probs = results['probabilities']
        print("\nCHAMPIONSHIP PROBABILITIES:")
        print("-" * 40)

        for driver, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{driver:12} {prob:6.2%}  {bar}")

    if 'expected_positions' in results:
        print("\nEXPECTED FINISHING POSITIONS:")
        print("-" * 40)
        for driver, pos in results['expected_positions'].items():
            print(f"{driver:12} {pos:6.2f}")

    if 'event_probabilities' in results:
        print("\nRACE EVENT PROBABILITIES:")
        print("-" * 40)
        for event, prob in results['event_probabilities'].items():
            print(f"{event:20} {prob:6.2%}")

    print("=" * 70)


# Main execution for standalone use
if __name__ == "__main__":
    # Example usage
    config = SimulationConfig(
        n_simulations=100000,
        safety_car_probability=0.31,
        include_historical_events=True,
        random_seed=42
    )

    current_points = {
        'NORRIS': 408,
        'VERSTAPPEN': 396,
        'PIASTRI': 392
    }

    driver_profiles = create_default_driver_profiles()
    historical_events = load_historical_event_probs()

    simulator = MonteCarloSimulator(config)
    probabilities = simulator.run_simulation(
        driver_profiles=driver_profiles,
        current_points=current_points,
        historical_event_probs=historical_events
    )

    report = simulator.generate_report()
    print_simulation_summary(report['results'])

    # Save results
    simulator.save_results()

    # Print summary
    summary = report['summary']
    print(f"\n🎯 PREDICTED CHAMPION: {summary['predicted_champion']}")
    print(f"   Probability: {summary['champion_probability']:.1%}")

    if summary.get('closest_contender'):
        print(f"   Closest contender: {summary['closest_contender']}")
        print(f"   Probability gap: {summary['probability_gap']:.1%}")

    print("\nKey Insights:")
    for insight in summary['key_insights']:
        print(f"  • {insight}")
