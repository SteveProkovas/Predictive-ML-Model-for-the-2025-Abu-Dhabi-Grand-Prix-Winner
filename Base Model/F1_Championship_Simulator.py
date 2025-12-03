"""
F1 2025 ABU DHABI GRAND PRIX CHAMPIONSHIP PREDICTOR
Minimal Monte Carlo simulation for the 2025 championship decider.
"""

import numpy as np
import random
from datetime import datetime

# ==================== HISTORICAL DATA ====================
historical_data = [
    {"season": 1990, "leader": "Mansell", "points_lead": 1, "champion": "Senna", "leader_won": False, "mechanical_failure": True, "collision": False},
    {"season": 1994, "leader": "Schumacher", "points_lead": 1, "champion": "Schumacher", "leader_won": True, "mechanical_failure": False, "collision": True},
    {"season": 1995, "leader": "Hill", "points_lead": 3, "champion": "Hill", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 1996, "leader": "Hill", "points_lead": 3, "champion": "Hill", "leader_won": True, "mechanical_failure": True, "collision": False},
    {"season": 1997, "leader": "Villeneuve", "points_lead": 3, "champion": "Villeneuve", "leader_won": True, "mechanical_failure": False, "collision": True},
    {"season": 1998, "leader": "Häkkinen", "points_lead": 10, "champion": "Häkkinen", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 1999, "leader": "Häkkinen", "points_lead": 13, "champion": "Häkkinen", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2000, "leader": "Schumacher", "points_lead": 2, "champion": "Schumacher", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2001, "leader": "Schumacher", "points_lead": 2, "champion": "Schumacher", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2003, "leader": "Montoya", "points_lead": 2, "champion": "Räikkönen", "leader_won": False, "mechanical_failure": False, "collision": False},
    {"season": 2005, "leader": "Alonso", "points_lead": 35, "champion": "Alonso", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2006, "leader": "Alonso", "points_lead": 0, "champion": "Alonso", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2007, "leader": "Hamilton", "points_lead": 4, "champion": "Räikkönen", "leader_won": False, "mechanical_failure": False, "collision": False},
    {"season": 2008, "leader": "Hamilton", "points_lead": 7, "champion": "Hamilton", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2009, "leader": "Button", "points_lead": 5, "champion": "Button", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2010, "leader": "Hamilton", "points_lead": 12, "champion": "Vettel", "leader_won": False, "mechanical_failure": False, "collision": False},
    {"season": 2012, "leader": "Vettel", "points_lead": 7, "champion": "Vettel", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2014, "leader": "Hamilton", "points_lead": 0, "champion": "Rosberg", "leader_won": False, "mechanical_failure": False, "collision": False},
    {"season": 2016, "leader": "Hamilton", "points_lead": 9, "champion": "Rosberg", "leader_won": False, "mechanical_failure": False, "collision": False},
    {"season": 2017, "leader": "Hamilton", "points_lead": 12, "champion": "Hamilton", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2020, "leader": "Hamilton", "points_lead": 124, "champion": "Hamilton", "leader_won": True, "mechanical_failure": False, "collision": False},
    {"season": 2021, "leader": "Hamilton", "points_lead": 4, "champion": "Verstappen", "leader_won": False, "mechanical_failure": False, "collision": False},
    {"season": 2022, "leader": "Verstappen", "points_lead": 67, "champion": "Verstappen", "leader_won": True, "mechanical_failure": False, "collision": False},
]

# ==================== HISTORICAL ANALYSIS ====================
def analyze_historical_data(current_lead):
    """Analyze historical championship deciders for context."""
    
    print("\n" + "="*70)
    print("📜 HISTORICAL CHAMPIONSHIP DECIDER ANALYSIS (1990-2022)")
    print("="*70)
    
    # Basic statistics
    total_deciders = len(historical_data)
    leader_wins = sum(1 for d in historical_data if d["leader_won"])
    leader_losses = total_deciders - leader_wins
    mechanical_failures = sum(1 for d in historical_data if d["mechanical_failure"])
    collisions = sum(1 for d in historical_data if d["collision"])
    
    print(f"\nOVERALL STATISTICS ({total_deciders} last-race deciders):")
    print("-"*40)
    print(f"• Leader entering final race won: {leader_wins}/{total_deciders} ({leader_wins/total_deciders*100:.1f}%)")
    print(f"• Leader entering final race lost: {leader_losses}/{total_deciders} ({leader_losses/total_deciders*100:.1f}%)")
    print(f"• Mechanical failures in deciders: {mechanical_failures}")
    print(f"• Collisions in deciders: {collisions}")
    
    # Analyze by lead size
    print(f"\nANALYSIS BY LEAD SIZE (Norris leads by {current_lead} points):")
    print("-"*40)
    
    # Group leads by size
    small_leads = [d for d in historical_data if d["points_lead"] <= 5]
    medium_leads = [d for d in historical_data if 5 < d["points_lead"] <= 15]
    large_leads = [d for d in historical_data if d["points_lead"] > 15]
    
    print(f"Small leads (≤5 points): {len(small_leads)} seasons")
    if small_leads:
        small_leader_wins = sum(1 for d in small_leads if d["leader_won"])
        print(f"  • Leader won: {small_leader_wins}/{len(small_leads)} ({small_leader_wins/len(small_leads)*100:.1f}%)")
    
    print(f"Medium leads (6-15 points): {len(medium_leads)} seasons")
    if medium_leads:
        medium_leader_wins = sum(1 for d in medium_leads if d["leader_won"])
        print(f"  • Leader won: {medium_leader_wins}/{len(medium_leads)} ({medium_leader_wins/len(medium_leads)*100:.1f}%)")
    
    print(f"Large leads (>15 points): {len(large_leads)} seasons")
    if large_leads:
        large_leader_wins = sum(1 for d in large_leads if d["leader_won"])
        print(f"  • Leader won: {large_leader_wins}/{len(large_leads)} ({large_leader_wins/len(large_leads)*100:.1f}%)")
    
    # Find similar historical situations
    similar_leads = [d for d in historical_data if abs(d["points_lead"] - current_lead) <= 3]
    
    if similar_leads:
        print(f"\nSIMILAR HISTORICAL SITUATIONS (±3 points of {current_lead}-point lead):")
        print("-"*40)
        for d in similar_leads:
            outcome = "WON" if d["leader_won"] else "LOST"
            drama = []
            if d["mechanical_failure"]:
                drama.append("mechanical failure")
            if d["collision"]:
                drama.append("collision")
            drama_text = f" ({', '.join(drama)})" if drama else ""
            print(f"• {d['season']}: {d['leader']} led by {d['points_lead']} points, {outcome} to {d['champion']}{drama_text}")
    
    # Most dramatic turnarounds
    print(f"\nBIGGEST LAST-RACE TURNAROUNDS:")
    print("-"*40)
    turnarounds = [d for d in historical_data if not d["leader_won"]]
    turnarounds.sort(key=lambda x: x["points_lead"], reverse=True)
    
    for i, d in enumerate(turnarounds[:3]):  # Top 3
        print(f"{i+1}. {d['season']}: {d['leader']} led by {d['points_lead']} points, lost to {d['champion']}")
    
    return {
        "total_deciders": total_deciders,
        "leader_win_rate": leader_wins / total_deciders,
        "similar_situations": similar_leads
    }

# ==================== CHAMPIONSHIP RULES ====================
def determine_champion(pia_pos, nor_pos, ver_pos, current_points):
    """Determines 2025 Champion based on Abu Dhabi finishing positions."""
    # Oscar Piastri's Scenarios
    if pia_pos == 1 and nor_pos >= 6 and ver_pos >= 2:
        return 'PIASTRI'
    if pia_pos == 2 and nor_pos >= 10 and ver_pos >= 4:
        return 'PIASTRI'
    
    # Lando Norris's Scenarios
    if nor_pos in [1, 2, 3]:
        return 'NORRIS'
    if nor_pos == 4 and ver_pos >= 2:
        return 'NORRIS'
    if nor_pos == 5 and ver_pos >= 2:
        return 'NORRIS'
    if nor_pos == 6 and ver_pos >= 2 and pia_pos >= 2:
        return 'NORRIS'
    if nor_pos == 7 and ver_pos >= 2 and pia_pos >= 2:
        return 'NORRIS'
    if nor_pos == 8 and ver_pos >= 3 and pia_pos >= 2:
        return 'NORRIS'
    if nor_pos == 9 and ver_pos >= 4 and pia_pos >= 2:
        return 'NORRIS'
    if nor_pos == 10 and ver_pos >= 4 and pia_pos >= 3:
        return 'NORRIS'
    if nor_pos == 11 and ver_pos >= 4 and pia_pos >= 3:
        return 'NORRIS'
    
    # Max Verstappen's Scenarios
    if ver_pos == 1 and nor_pos >= 4:
        return 'VERSTAPPEN'
    if ver_pos == 2 and nor_pos >= 8 and pia_pos >= 3:
        return 'VERSTAPPEN'
    if ver_pos == 3 and nor_pos >= 9 and pia_pos >= 2:
        return 'VERSTAPPEN'
    
    # If no scenario met, calculate by points
    points_system = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    
    pia_points = current_points['PIASTRI'] + points_system.get(pia_pos, 0)
    nor_points = current_points['NORRIS'] + points_system.get(nor_pos, 0)
    ver_points = current_points['VERSTAPPEN'] + points_system.get(ver_pos, 0)
    
    if ver_points > nor_points and ver_points > pia_points:
        return 'VERSTAPPEN'
    elif nor_points > ver_points and nor_points > pia_points:
        return 'NORRIS'
    elif pia_points > ver_points and pia_points > nor_points:
        return 'PIASTRI'
    else:
        # Tie-break: more wins (assume Norris has most wins)
        return 'NORRIS'

# ==================== PERFORMANCE PROFILES ====================
def create_simple_probability_profile(driver):
    """Simple probability profiles based on 2025 performance."""
    # Position probabilities for P1-P20 + DNF (21 total)
    profiles = {
        'NORRIS': [
            0.25, 0.20, 0.15, 0.10, 0.08,  # P1-P5: 78%
            0.06, 0.05, 0.03, 0.02, 0.01,  # P6-P10: 17%
            0.005, 0.005, 0.005, 0.005, 0.005,  # P11-P15: 2.5%
            0.002, 0.002, 0.002, 0.002, 0.002,  # P16-P20: 1%
            0.02  # DNF: 2%
        ],
        'VERSTAPPEN': [
            0.22, 0.18, 0.16, 0.12, 0.10,  # P1-P5: 78%
            0.07, 0.05, 0.03, 0.02, 0.01,  # P6-P10: 18%
            0.005, 0.005, 0.005, 0.005, 0.005,  # P11-P15: 2.5%
            0.002, 0.002, 0.002, 0.002, 0.002,  # P16-P20: 1%
            0.02  # DNF: 2%
        ],
        'PIASTRI': [
            0.20, 0.17, 0.15, 0.13, 0.10,  # P1-P5: 75%
            0.08, 0.06, 0.04, 0.02, 0.01,  # P6-P10: 21%
            0.005, 0.005, 0.005, 0.005, 0.005,  # P11-P15: 2.5%
            0.002, 0.002, 0.002, 0.002, 0.002,  # P16-P20: 1%
            0.02  # DNF: 2%
        ]
    }
    return profiles[driver]

# ==================== MONTE CARLO SIMULATION ====================
def simulate_championship(current_points, n_simulations=50000):
    """Run Monte Carlo simulation to predict championship probabilities."""
    
    # Get probability profiles
    prob_nor = create_simple_probability_profile('NORRIS')
    prob_ver = create_simple_probability_profile('VERSTAPPEN')
    prob_pia = create_simple_probability_profile('PIASTRI')
    
    # Pre-calculate cumulative probabilities for speed
    cum_nor = np.cumsum(prob_nor)
    cum_ver = np.cumsum(prob_ver)
    cum_pia = np.cumsum(prob_pia)
    
    # Initialize counters
    wins = {'NORRIS': 0, 'VERSTAPPEN': 0, 'PIASTRI': 0}
    
    print(f"Running {n_simulations:,} Monte Carlo simulations...")
    
    # Progress tracking
    checkpoint = max(1, n_simulations // 10)
    
    for i in range(n_simulations):
        # Sample finishing positions
        pos_nor = np.searchsorted(cum_nor, random.random())
        pos_ver = np.searchsorted(cum_ver, random.random())
        pos_pia = np.searchsorted(cum_pia, random.random())
        
        # Convert to race positions (DNF = 99)
        nor_finish = 99 if pos_nor == 20 else pos_nor + 1
        ver_finish = 99 if pos_ver == 20 else pos_ver + 1
        pia_finish = 99 if pos_pia == 20 else pos_pia + 1
        
        # Determine champion
        champ = determine_champion(pia_finish, nor_finish, ver_finish, current_points)
        wins[champ] += 1
        
        # Progress update
        if checkpoint > 0 and (i + 1) % checkpoint == 0:
            percentage = (i + 1) / n_simulations * 100
            print(f"  Progress: {percentage:.0f}% complete")
    
    # Calculate probabilities
    total = n_simulations
    prob_nor = wins['NORRIS'] / total * 100
    prob_ver = wins['VERSTAPPEN'] / total * 100
    prob_pia = wins['PIASTRI'] / total * 100
    
    return prob_nor, prob_ver, prob_pia, wins

# ==================== VISUALIZATION ====================
def display_results(prob_nor, prob_ver, prob_pia, wins, n_simulations, current_points, historical_stats=None):
    """Display simulation results in terminal."""
    
    print("\n" + "="*70)
    print("2025 FORMULA 1 CHAMPIONSHIP DECIDER - ABU DHABI GP")
    print("="*70)
    
    # Current standings
    print(f"\nCURRENT STANDINGS:")
    print("-"*40)
    for driver, points in current_points.items():
        print(f"  {driver:12} {points:4d} points")
    
    # Championship probabilities
    print(f"\nCHAMPIONSHIP PROBABILITIES ({n_simulations:,} simulations):")
    print("-"*40)
    
    # Create bar chart
    max_bar = 40
    bar_nor = '█' * int(prob_nor/100 * max_bar)
    bar_ver = '█' * int(prob_ver/100 * max_bar)
    bar_pia = '█' * int(prob_pia/100 * max_bar)
    
    print(f"Lando Norris:    {prob_nor:6.2f}%  {bar_nor:<{max_bar}}")
    print(f"Max Verstappen:  {prob_ver:6.2f}%  {bar_ver:<{max_bar}}")
    print(f"Oscar Piastri:   {prob_pia:6.2f}%  {bar_pia:<{max_bar}}")
    
    # Historical context
    if historical_stats:
        historical_win_rate = historical_stats["leader_win_rate"] * 100
        comparison = "HIGHER" if prob_nor > historical_win_rate else "LOWER"
        print(f"\n📊 HISTORICAL CONTEXT: Leaders have won {historical_win_rate:.1f}% of last-race deciders")
        print(f"   Norris's probability is {abs(prob_nor - historical_win_rate):.1f}% {comparison} than historical average")
    
    # Scenario analysis
    print("\nKEY SCENARIOS:")
    print("-"*40)
    points_gap = current_points['NORRIS'] - current_points['VERSTAPPEN']
    print(f"• Norris's {points_gap}-point lead is decisive")
    print(f"• Verstappen MUST WIN + Norris P5 or lower")
    print(f"• Piastri MUST WIN + Norris P6+ + Verstappen P2+")
    
    # Final prediction
    print("\n" + "="*70)
    probs = {'NORRIS': prob_nor, 'VERSTAPPEN': prob_ver, 'PIASTRI': prob_pia}
    champion = max(probs, key=probs.get)
    
    if probs[champion] > 70:
        print(f"🎯 PREDICTION: {champion} is the STRONG FAVORITE ({probs[champion]:.1f}%)")
    elif probs[champion] > 55:
        print(f"🎯 PREDICTION: {champion} is the MODERATE FAVORITE ({probs[champion]:.1f}%)")
    elif probs[champion] > 45:
        print(f"🎯 PREDICTION: {champion} is SLIGHTLY FAVORED ({probs[champion]:.1f}%)")
    else:
        print(f"🎯 PREDICTION: TOO CLOSE TO CALL - {champion} marginally ahead ({probs[champion]:.1f}%)")
    print("="*70)

# ==================== MAIN FUNCTION ====================
def main():
    """Main execution function."""
    
    # 2025 Championship standings (after Las Vegas GP)
    current_points = {
        'NORRIS': 408,    # 1st in championship
        'VERSTAPPEN': 396, # 2nd in championship
        'PIASTRI': 392     # 3rd in championship
    }
    
    print("\n" + "="*70)
    print("🏎️ F1 2025 ABU DHABI GRAND PRIX - CHAMPIONSHIP PREDICTOR")
    print("="*70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Historical analysis
    current_lead = current_points['NORRIS'] - current_points['VERSTAPPEN']
    historical_stats = analyze_historical_data(current_lead)
    
    # Number of simulations (adjust for your hardware)
    # Ryzen 5 4600H: 500,000 simulations ≈ 3-5 seconds
    n_simulations = 500000
    
    # Run simulation
    prob_nor, prob_ver, prob_pia, wins = simulate_championship(
        current_points, n_simulations
    )
    
    # Display results
    display_results(prob_nor, prob_ver, prob_pia, wins, n_simulations, current_points, historical_stats)
    
    # Save results option
    save_option = input("\n💾 Save results to file? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = f"championship_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("F1 2025 ABU DHABI GRAND PRIX - CHAMPIONSHIP PREDICTION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Simulations: {n_simulations:,}\n\n")
            f.write("CURRENT STANDINGS:\n")
            for driver, points in current_points.items():
                f.write(f"  {driver}: {points} points\n")
            f.write("\nCHAMPIONSHIP PROBABILITIES:\n")
            f.write(f"  Lando Norris:    {prob_nor:.2f}%\n")
            f.write(f"  Max Verstappen:  {prob_ver:.2f}%\n")
            f.write(f"  Oscar Piastri:   {prob_pia:.2f}%\n\n")
            f.write(f"PREDICTION: {max({'NORRIS': prob_nor, 'VERSTAPPEN': prob_ver, 'PIASTRI': prob_pia}, key={'NORRIS': prob_nor, 'VERSTAPPEN': prob_ver, 'PIASTRI': prob_pia}.get)} wins championship\n")
        print(f"✅ Results saved to '{filename}'")

# ==================== EXECUTION ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Simulation cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
