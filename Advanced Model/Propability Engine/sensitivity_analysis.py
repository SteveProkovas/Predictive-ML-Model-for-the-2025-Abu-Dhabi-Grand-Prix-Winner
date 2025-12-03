#!/usr/bin/env python3
"""
Analyze sensitivity of probabilities to different factors.
"""

import numpy as np
import matplotlib.pyplot as plt
from probability_engine import (
    ProbabilityEngine, DriverPerformanceMetrics,
    create_default_driver_metrics, create_yas_marina_track
)

def analyze_grid_position_impact():
    """Analyze how grid position affects win probability."""
    
    driver_metrics = create_default_driver_metrics()
    track = create_yas_marina_track()
    engine = ProbabilityEngine(model_type="hybrid")
    
    grid_positions_range = range(1, 11)
    results = {driver: [] for driver in driver_metrics.keys()}
    
    print("Analyzing grid position impact...")
    
    for grid_pos in grid_positions_range:
        grid_positions = {driver: grid_pos for driver in driver_metrics.keys()}
        
        probabilities = engine.generate_probabilities(
            driver_metrics=driver_metrics,
            track=track,
            grid_positions=grid_positions
        )
        
        for driver, probs in probabilities.items():
            results[driver].append(probs[0])  # Win probability
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for driver, win_probs in results.items():
        plt.plot(grid_positions_range, win_probs, 'o-', label=driver, linewidth=2, markersize=8)
    
    plt.xlabel('Grid Position', fontsize=12)
    plt.ylabel('Win Probability', fontsize=12)
    plt.title('Impact of Grid Position on Win Probability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('sensitivity_grid_position.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot: sensitivity_grid_position.png")

def analyze_consistency_impact():
    """Analyze how driver consistency affects probability distribution."""
    
    base_metrics = create_default_driver_metrics()
    track = create_yas_marina_track()
    engine = ProbabilityEngine(model_type="hybrid")
    
    consistency_values = np.linspace(0.3, 0.95, 10)
    driver_name = 'NORRIS'  # Analyze Norris
    
    win_probs = []
    podium_probs = []
    dnf_probs = []
    
    print(f"Analyzing consistency impact for {driver_name}...")
    
    for consistency in consistency_values:
        # Create modified metrics
        metrics = base_metrics.copy()
        modified_metrics = metrics[driver_name]
        
        # Create new metrics with modified consistency
        modified_dict = modified_metrics.to_dict()
        modified_dict['consistency_score'] = consistency
        modified_metrics = DriverPerformanceMetrics.from_dict(modified_dict)
        
        metrics[driver_name] = modified_metrics
        
        # Generate probabilities
        probabilities = engine.generate_probabilities(
            driver_metrics=metrics,
            track=track
        )
        
        probs = probabilities[driver_name]
        win_probs.append(probs[0])
        podium_probs.append(sum(probs[0:3]))
        dnf_probs.append(probs[20])
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(consistency_values, win_probs, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Consistency Score', fontsize=11)
    axes[0].set_ylabel('Win Probability', fontsize=11)
    axes[0].set_title('Win Probability vs Consistency', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(consistency_values, podium_probs, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Consistency Score', fontsize=11)
    axes[1].set_ylabel('Podium Probability', fontsize=11)
    axes[1].set_title('Podium Probability vs Consistency', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(consistency_values, dnf_probs, 'r-o', linewidth=2, markersize=8)
    axes[2].set_xlabel('Consistency Score', fontsize=11)
    axes[2].set_ylabel('DNF Probability', fontsize=11)
    axes[2].set_title('DNF Probability vs Consistency', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Impact of Consistency on Performance - {driver_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('sensitivity_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot: sensitivity_consistency.png")

def analyze_momentum_impact():
    """Analyze how recent momentum affects probabilities."""
    
    base_metrics = create_default_driver_metrics()
    track = create_yas_marina_track()
    engine = ProbabilityEngine(model_type="hybrid")
    
    momentum_values = np.linspace(-0.5, 0.5, 11)
    driver_name = 'VERSTAPPEN'
    
    win_probs = []
    expected_positions = []
    
    print(f"Analyzing momentum impact for {driver_name}...")
    
    for momentum in momentum_values:
        metrics = base_metrics.copy()
        modified_metrics = metrics[driver_name]
        
        modified_dict = modified_metrics.to_dict()
        modified_dict['momentum'] = momentum
        
        # Adjust recent finishes based on momentum
        base_finishes = modified_dict['recent_finishes']
        adjusted_finishes = [max(1, min(20, f - int(momentum * 5))) for f in base_finishes]
        modified_dict['recent_finishes'] = adjusted_finishes
        
        modified_metrics = DriverPerformanceMetrics.from_dict(modified_dict)
        metrics[driver_name] = modified_metrics
        
        probabilities = engine.generate_probabilities(
            driver_metrics=metrics,
            track=track
        )
        
        probs = probabilities[driver_name]
        win_probs.append(probs[0])
        expected_pos = sum((i+1) * prob for i, prob in enumerate(probs[:20]))
        expected_positions.append(expected_pos)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(momentum_values, win_probs, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Momentum (-0.5 to +0.5)', fontsize=11)
    ax1.set_ylabel('Win Probability', fontsize=11)
    ax1.set_title('Win Probability vs Momentum', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(momentum_values, expected_positions, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Momentum (-0.5 to +0.5)', fontsize=11)
    ax2.set_ylabel('Expected Finishing Position', fontsize=11)
    ax2.set_title('Expected Position vs Momentum', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Lower position number = better
    
    plt.suptitle(f'Impact of Recent Momentum - {driver_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('sensitivity_momentum.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved plot: sensitivity_momentum.png")

def main():
    print("PROBABILITY ENGINE SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    analyze_grid_position_impact()
    analyze_consistency_impact()
    analyze_momentum_impact()
    
    print("\n✅ Sensitivity analysis complete!")
    print("   Check generated plots for insights.")

if __name__ == "__main__":
    main()
