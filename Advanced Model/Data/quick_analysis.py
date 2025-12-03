#!/usr/bin/env python3
"""
Quick analysis for specific scenarios.
"""

import json
from historical_analyzer import HistoricalPatternAnalyzer
import pandas as pd


def analyze_specific_gaps():
    """Analyze specific points gap scenarios."""

    # Load historical data
    df = pd.read_csv("data/historical_deciders_raw.csv")
    analyzer = HistoricalPatternAnalyzer(df)
    patterns = analyzer.analyze_all_patterns()

    gaps_to_analyze = [0, 5, 10, 12, 20, 50]

    print("Points Gap Analysis")
    print("=" * 50)

    for gap in gaps_to_analyze:
        rec = analyzer.generate_recommendations(gap)
        gap_info = rec['based_on_gap']

        print(f"\n{gap}-point lead:")
        print(f"  Category: {gap_info['category']}")
        print(f"  Historical success: {gap_info['historical_success_rate']:.1%}")
        print(f"  Similar cases: {gap_info['similar_cases_count']}")
        print(f"  Recommendation: {gap_info['recommendation']}")


def compare_decades():
    """Compare patterns across decades."""

    df = pd.read_csv("data/historical_deciders_raw.csv")
    analyzer = HistoricalPatternAnalyzer(df)
    patterns = analyzer.analyze_all_patterns()

    print("\nDecade Comparison")
    print("=" * 50)

    for decade, data in patterns.decade_trends.items():
        print(f"\n{decade}s:")
        print(f"  Deciders: {data['count']}")
        print(f"  Leader win rate: {data['leader_win_rate']:.1%}")
        print(f"  Drama rate: {data['drama_rate']:.1%}")
        print(f"  Common event: {data['most_common_event']}")


def event_probability_breakdown():
    """Show detailed event probability breakdown."""

    with open("data/historical_patterns.json", 'r') as f:
        patterns = json.load(f)

    print("\nEvent Probability Breakdown")
    print("=" * 50)

    for event, prob in patterns['event_probabilities'].items():
        if 'combined' not in event:
            print(f"{event:25} {prob:.1%}")


if __name__ == "__main__":
    print("F1 Championship Historical Quick Analysis")
    print("=" * 50)

    analyze_specific_gaps()
    compare_decades()
    event_probability_breakdown()
