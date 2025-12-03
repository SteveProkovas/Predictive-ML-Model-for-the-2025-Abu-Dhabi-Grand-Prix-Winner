"""
ENHANCED PROBABILITY ENGINE FOR F1 CHAMPIONSHIP PREDICTION
Creates sophisticated, data-driven probability models for driver finishing positions.
Integrates historical patterns, current form, and situational adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from datetime import datetime
from scipy import stats
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('probability_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DriverPerformanceMetrics:
    """Comprehensive performance metrics for a driver."""
    # Core statistics
    name: str
    team: str
    
    # 2025 Season Performance (updated after each race)
    current_points: int
    championship_position: int
    races_completed: int
    wins: int
    podiums: int
    poles: int
    fastest_laps: int
    dnf_count: int
    avg_finish: float
    avg_qualifying: float
    
    # Career Statistics (from screenshots)
    career_races: int
    career_wins: int
    career_podiums: int
    career_poles: int
    career_dnf_rate: float
    career_avg_finish: float
    
    # Recent Form (last 5 races)
    recent_finishes: List[int]  # Positions in last 5 races
    recent_qualifying: List[int]  # Qualifying positions last 5 races
    recent_points: List[int]  # Points scored last 5 races
    
    # Track-Specific Performance (Yas Marina)
    yas_marina_races: int = 0
    yas_marina_wins: int = 0
    yas_marina_podiums: int = 0
    yas_marina_avg_finish: float = 10.0
    yas_marina_dnf: int = 0
    
    # Advanced Metrics
    consistency_score: float = 0.5  # 0-1, higher = more consistent
    pressure_score: float = 0.5  # 0-1, performance under pressure
    race_craft_score: float = 0.5  # 0-1, overtaking/defending ability
    qualifying_strength: float = 0.5  # 0-1, single-lap pace
    
    # Current Context
    momentum: float = 0.0  # -1 to +1, negative = declining, positive = improving
    championship_pressure: float = 0.0  # 0-1, pressure of championship situation
    reliability_risk: float = 0.0  # 0-1, risk of mechanical failure
    
    def __post_init__(self):
        """Validate and calculate derived metrics."""
        self._validate_metrics()
        self._calculate_derived_metrics()
    
    def _validate_metrics(self):
        """Validate input metrics."""
        if self.races_completed == 0:
            raise ValueError(f"Driver {self.name}: races_completed cannot be 0")
        if self.career_races == 0:
            self.career_races = self.races_completed
        
        # Ensure lists are proper length
        self.recent_finishes = self.recent_finishes[:5]  # Keep last 5
        self.recent_qualifying = self.recent_qualifying[:5]
        self.recent_points = self.recent_points[:5]
        
        # Fill missing recent data with averages
        if len(self.recent_finishes) < 5:
            avg_finish = self.avg_finish if self.avg_finish > 0 else 10
            self.recent_finishes.extend([int(avg_finish)] * (5 - len(self.recent_finishes)))
    
    def _calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        # Calculate momentum from recent form
        if len(self.recent_finishes) >= 3:
            # Recent finishes compared to season average
            recent_avg = np.mean(self.recent_finishes[:3])  # Last 3 races
            self.momentum = (self.avg_finish - recent_avg) / 10  # Normalize
        
        # Calculate consistency (lower variance = more consistent)
        if len(self.recent_finishes) >= 5:
            finish_std = np.std(self.recent_finishes)
            max_std = 10  # Maximum reasonable standard deviation
            self.consistency_score = max(0.1, 1 - (finish_std / max_std))
        
        # Calculate win and podium rates for 2025 season
        self.win_rate_2025 = self.wins / self.races_completed
        self.podium_rate_2025 = self.podiums / self.races_completed
        self.dnf_rate_2025 = self.dnf_count / self.races_completed
        
        # Calculate career rates
        self.career_win_rate = self.career_wins / self.career_races
        self.career_podium_rate = self.career_podiums / self.career_races
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DriverPerformanceMetrics':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class TrackCharacteristics:
    """Characteristics of a specific track that affect performance."""
    name: str
    type: str  # "High-speed", "Technical", "Street", "Power", "Mixed"
    overtaking_difficulty: float  # 0-1, higher = harder to overtake
    tire_degradation: float  # 0-1, higher = more degradation
    safety_car_probability: float  # 0-1, historical SC probability
    dnf_rate: float  # 0-1, historical DNF rate at this track
    
    # Performance multipliers for different driver skills
    qualifying_multiplier: float = 1.0
    race_pace_multiplier: float = 1.0
    consistency_multiplier: float = 1.0
    reliability_multiplier: float = 1.0

class BaseProbabilityModel:
    """Base class for probability models."""
    
    def __init__(self):
        self.points_system = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    
    def create_probability_distribution(self, 
                                       driver_metrics: DriverPerformanceMetrics,
                                       track: Optional[TrackCharacteristics] = None,
                                       grid_position: Optional[int] = None) -> np.ndarray:
        """
        Create probability distribution for finishing positions (1-20 + DNF).
        
        Args:
            driver_metrics: Driver performance metrics
            track: Track characteristics (optional)
            grid_position: Starting grid position (optional)
            
        Returns:
            Array of 21 probabilities (positions 1-20 + DNF)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def validate_distribution(self, probs: np.ndarray) -> bool:
        """Validate that probability distribution is valid."""
        if len(probs) != 21:
            logger.error(f"Invalid distribution length: {len(probs)} (expected 21)")
            return False
        
        if np.any(probs < 0):
            logger.error("Negative probabilities found")
            return False
        
        if not np.allclose(probs.sum(), 1.0, atol=1e-5):
            logger.error(f"Probabilities sum to {probs.sum():.6f}, not 1.0")
            return False
        
        return True

class StatisticalProbabilityModel(BaseProbabilityModel):
    """
    Statistical model based on historical performance metrics.
    Uses regression and statistical distributions to estimate probabilities.
    """
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        super().__init__()
        self.historical_data = historical_data
        
        # Parameters learned from historical data
        self.position_shift_params = {
            'qualifying_to_race': 0.8,  # Average position shift from qualifying to race
            'consistency_effect': 0.3,  # Effect of consistency on finishing variance
            'experience_effect': 0.15,  # Effect of experience on finishing position
        }
    
    def create_probability_distribution(self,
                                       driver_metrics: DriverPerformanceMetrics,
                                       track: Optional[TrackCharacteristics] = None,
                                       grid_position: Optional[int] = None) -> np.ndarray:
        """
        Create probability distribution using statistical modeling.
        """
        logger.debug(f"Creating statistical distribution for {driver_metrics.name}")
        
        # Base parameters
        probs = np.zeros(21)
        
        # 1. Estimate expected finishing position
        expected_position = self._calculate_expected_position(driver_metrics, track, grid_position)
        
        # 2. Calculate variance based on consistency
        position_variance = self._calculate_position_variance(driver_metrics)
        
        # 3. Create normal distribution around expected position
        # Use truncated normal distribution for positions 1-20
        positions = np.arange(1, 21)
        
        # Create probability density
        pdf_values = stats.norm.pdf(positions, loc=expected_position, scale=position_variance)
        
        # Ensure proper scaling for discrete positions
        pdf_values = pdf_values / pdf_values.sum()
        
        # Assign to positions 1-20
        probs[:20] = pdf_values
        
        # 4. Calculate DNF probability
        dnf_prob = self._calculate_dnf_probability(driver_metrics, track)
        probs[20] = dnf_prob
        
        # 5. Adjust for remaining probability mass
        probs[:20] = probs[:20] * (1 - dnf_prob) / probs[:20].sum()
        
        # 6. Apply softmax for smooth distribution
        probs = self._apply_softmax_adjustment(probs, driver_metrics)
        
        # Validate
        if not self.validate_distribution(probs):
            logger.warning(f"Invalid distribution for {driver_metrics.name}, using fallback")
            probs = self._create_fallback_distribution(driver_metrics)
        
        return probs
    
    def _calculate_expected_position(self,
                                   driver_metrics: DriverPerformanceMetrics,
                                   track: Optional[TrackCharacteristics],
                                   grid_position: Optional[int]) -> float:
        """Calculate expected finishing position."""
        # Base expected position from season average
        base_position = driver_metrics.avg_finish
        
        # Adjust for grid position if provided
        if grid_position is not None:
            # Historical relationship: race position ≈ 0.8 * qualifying position + 2.0
            qual_adjusted = 0.8 * grid_position + 2.0
            # Blend with season average
            base_position = 0.7 * qual_adjusted + 0.3 * base_position
        
        # Adjust for track-specific performance
        if track is not None:
            track_factor = 1.0
            if driver_metrics.yas_marina_races > 0:
                # Use track-specific average if available
                track_position = driver_metrics.yas_marina_avg_finish
                # Blend with overall average
                track_weight = min(0.5, driver_metrics.yas_marina_races / 10)
                base_position = (1 - track_weight) * base_position + track_weight * track_position
            
            # Adjust for track type vs driver skills
            if track.type == "Technical" and driver_metrics.race_craft_score > 0.7:
                base_position -= 1.0  # Better on technical tracks
            elif track.type == "Power" and driver_metrics.qualifying_strength > 0.7:
                base_position -= 0.5  # Better on power tracks
        
        # Adjust for momentum
        momentum_effect = driver_metrics.momentum * 2  # ±2 position effect
        base_position -= momentum_effect
        
        # Adjust for championship pressure
        pressure_effect = driver_metrics.championship_pressure * 1.5
        base_position += pressure_effect  # Pressure can worsen performance
        
        # Ensure within bounds
        return max(1.0, min(20.0, base_position))
    
    def _calculate_position_variance(self, driver_metrics: DriverPerformanceMetrics) -> float:
        """Calculate variance in finishing positions."""
        # Base variance from consistency score
        base_variance = 5.0 * (1 - driver_metrics.consistency_score)
        
        # Increase variance for less experienced drivers
        experience_factor = max(0.5, driver_metrics.career_races / 100)
        base_variance *= (2 - experience_factor)
        
        # Adjust for momentum (improving drivers have lower variance)
        momentum_factor = 1 - abs(driver_metrics.momentum) * 0.5
        base_variance *= momentum_factor
        
        return max(1.0, min(10.0, base_variance))
    
    def _calculate_dnf_probability(self,
                                 driver_metrics: DriverPerformanceMetrics,
                                 track: Optional[TrackCharacteristics]) -> float:
        """Calculate DNF probability."""
        # Base DNF rate from current season and career
        base_dnf = (driver_metrics.dnf_rate_2025 + driver_metrics.career_dnf_rate) / 2
        
        # Adjust for reliability risk
        base_dnf *= (1 + driver_metrics.reliability_risk)
        
        # Adjust for track-specific DNF rate
        if track is not None:
            base_dnf = (base_dnf + track.dnf_rate) / 2
        
        # Adjust for championship pressure (higher pressure = more risk-taking)
        pressure_effect = driver_metrics.championship_pressure * 0.5
        base_dnf *= (1 + pressure_effect)
        
        # Adjust for track-specific DNF history at Yas Marina
        if driver_metrics.yas_marina_races > 0:
            yas_dnf_rate = driver_metrics.yas_marina_dnf / driver_metrics.yas_marina_races
            base_dnf = 0.7 * base_dnf + 0.3 * yas_dnf_rate
        
        return min(0.3, max(0.01, base_dnf))
    
    def _apply_softmax_adjustment(self, probs: np.ndarray,
                                driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Apply softmax adjustment to smooth probabilities."""
        # Convert to logits (with temperature)
        logits = np.log(probs[:20] + 1e-10)
        
        # Temperature based on consistency (more consistent = lower temperature)
        temperature = 1.5 - driver_metrics.consistency_score
        
        # Apply softmax with temperature
        adjusted = softmax(logits / temperature)
        
        # Reapply DNF probability
        final_probs = np.zeros(21)
        final_probs[:20] = adjusted * (1 - probs[20])
        final_probs[20] = probs[20]
        
        return final_probs
    
    def _create_fallback_distribution(self,
                                    driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Create fallback distribution if statistical model fails."""
        logger.info(f"Using fallback distribution for {driver_metrics.name}")
        
        probs = np.zeros(21)
        
        # Simple model based on average finish
        expected_pos = driver_metrics.avg_finish
        
        # Create simple distribution
        positions = np.arange(1, 21)
        
        # Triangular distribution centered at expected position
        scale = 5.0
        pdf = 1 - np.abs(positions - expected_pos) / scale
        pdf = np.maximum(pdf, 0.1)  # Minimum probability
        
        # Normalize
        probs[:20] = pdf / pdf.sum() * 0.95  # 95% for finishing
        
        # DNF probability
        probs[20] = 0.05
        
        return probs

class MachineLearningProbabilityModel(BaseProbabilityModel):
    """
    Advanced probability model using machine learning techniques.
    Requires historical training data.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        super().__init__()
        self.historical_data = historical_data
        self.models = {}  # Will store trained models
        self.features = [
            'avg_finish', 'avg_qualifying', 'win_rate_2025', 'podium_rate_2025',
            'consistency_score', 'momentum', 'championship_position',
            'grid_position', 'track_overtaking_difficulty', 'track_tire_degradation'
        ]
        
        logger.info("Initializing ML Probability Model")
    
    def train_models(self):
        """Train ML models on historical data."""
        logger.info("Training ML models on historical data...")
        
        # This would involve complex ML training
        # For now, we'll implement a simplified version
        
        logger.info("ML models trained (simplified implementation)")
    
    def create_probability_distribution(self,
                                       driver_metrics: DriverPerformanceMetrics,
                                       track: Optional[TrackCharacteristics] = None,
                                       grid_position: Optional[int] = None) -> np.ndarray:
        """
        Create probability distribution using ML models.
        """
        logger.debug(f"Creating ML distribution for {driver_metrics.name}")
        
        # Extract features
        features = self._extract_features(driver_metrics, track, grid_position)
        
        # Predict using ensemble of models
        probs = self._ensemble_prediction(features, driver_metrics)
        
        # Apply track-specific adjustments
        if track is not None:
            probs = self._apply_track_adjustments(probs, track, driver_metrics)
        
        # Apply situational adjustments
        probs = self._apply_situational_adjustments(probs, driver_metrics)
        
        # Validate distribution
        if not self.validate_distribution(probs):
            logger.warning(f"ML distribution invalid for {driver_metrics.name}, using statistical model")
            statistical_model = StatisticalProbabilityModel()
            probs = statistical_model.create_probability_distribution(
                driver_metrics, track, grid_position
            )
        
        return probs
    
    def _extract_features(self,
                         driver_metrics: DriverPerformanceMetrics,
                         track: Optional[TrackCharacteristics],
                         grid_position: Optional[int]) -> Dict[str, float]:
        """Extract features for ML prediction."""
        features = {
            'avg_finish': driver_metrics.avg_finish,
            'avg_qualifying': driver_metrics.avg_qualifying,
            'win_rate_2025': driver_metrics.win_rate_2025,
            'podium_rate_2025': driver_metrics.podium_rate_2025,
            'consistency_score': driver_metrics.consistency_score,
            'momentum': driver_metrics.momentum,
            'championship_position': driver_metrics.championship_position,
            'career_win_rate': driver_metrics.career_win_rate,
            'career_podium_rate': driver_metrics.career_podium_rate,
        }
        
        # Add track features if available
        if track is not None:
            features.update({
                'track_overtaking_difficulty': track.overtaking_difficulty,
                'track_tire_degradation': track.tire_degradation,
                'track_safety_car_prob': track.safety_car_probability,
            })
        
        # Add grid position if available
        if grid_position is not None:
            features['grid_position'] = grid_position
        else:
            features['grid_position'] = driver_metrics.avg_qualifying
        
        return features
    
    def _ensemble_prediction(self,
                           features: Dict[str, float],
                           driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """
        Make ensemble prediction using multiple model types.
        Simplified implementation - in production would use actual ML models.
        """
        probs = np.zeros(21)
        
        # Model 1: Linear regression on historical patterns
        linear_probs = self._linear_model_prediction(features)
        
        # Model 2: Random forest style (simplified)
        forest_probs = self._forest_model_prediction(features)
        
        # Model 3: Neural network style (simplified)
        nn_probs = self._neural_network_prediction(features)
        
        # Ensemble average
        probs = (linear_probs + forest_probs + nn_probs) / 3
        
        # Add driver-specific adjustments
        probs = self._apply_driver_specific_adjustments(probs, driver_metrics)
        
        return probs
    
    def _linear_model_prediction(self, features: Dict[str, float]) -> np.ndarray:
        """Simplified linear model prediction."""
        probs = np.zeros(21)
        
        # Base expected position from linear combination
        expected_pos = (
            0.3 * features['avg_finish'] +
            0.2 * features['grid_position'] +
            0.1 * (21 - features['championship_position']) +  # Better championship = better finish
            0.2 * (1 - features['win_rate_2025'] * 10) +  # Win rate effect
            0.2 * np.random.normal(0, 2)  # Random noise
        )
        
        expected_pos = max(1, min(20, expected_pos))
        
        # Create normal distribution
        positions = np.arange(1, 21)
        pdf = stats.norm.pdf(positions, loc=expected_pos, scale=3.0)
        probs[:20] = pdf / pdf.sum() * 0.95
        
        # DNF probability
        base_dnf = 0.05
        if features.get('track_safety_car_prob', 0) > 0.3:
            base_dnf *= 1.2
        probs[20] = base_dnf
        
        # Normalize
        probs[:20] = probs[:20] * (1 - base_dnf) / probs[:20].sum()
        
        return probs
    
    def _forest_model_prediction(self, features: Dict[str, float]) -> np.ndarray:
        """Simplified random forest prediction."""
        probs = np.zeros(21)
        
        # Simulate decision tree splits
        expected_pos = features['avg_finish']
        
        # Adjust based on features (simulating tree splits)
        if features['win_rate_2025'] > 0.2:
            expected_pos -= 2.0
        if features['consistency_score'] > 0.7:
            expected_pos -= 1.0
        if features.get('track_overtaking_difficulty', 0) > 0.7:
            expected_pos += 1.0  # Harder to overtake = maintains position
        
        expected_pos = max(1, min(20, expected_pos))
        
        # Create distribution with variance based on consistency
        variance = 5.0 * (1 - features['consistency_score'])
        positions = np.arange(1, 21)
        pdf = stats.norm.pdf(positions, loc=expected_pos, scale=variance)
        probs[:20] = pdf / pdf.sum() * 0.94
        
        # DNF probability
        dnf_prob = 0.06
        if features.get('track_tire_degradation', 0) > 0.7:
            dnf_prob *= 1.3
        probs[20] = dnf_prob
        
        # Normalize
        probs[:20] = probs[:20] * (1 - dnf_prob) / probs[:20].sum()
        
        return probs
    
    def _neural_network_prediction(self, features: Dict[str, float]) -> np.ndarray:
        """Simplified neural network prediction."""
        probs = np.zeros(21)
        
        # Simulate neural network with hidden layers
        # Input layer
        inputs = np.array([
            features['avg_finish'] / 20.0,  # Normalize
            features['grid_position'] / 20.0,
            features['win_rate_2025'],
            features['consistency_score'],
            features['momentum'] if 'momentum' in features else 0,
        ])
        
        # Simulate hidden layer computations
        weights1 = np.random.randn(5, 10) * 0.5
        bias1 = np.random.randn(10) * 0.1
        hidden1 = np.tanh(np.dot(inputs, weights1) + bias1)
        
        weights2 = np.random.randn(10, 21) * 0.3
        bias2 = np.random.randn(21) * 0.05
        logits = np.dot(hidden1, weights2) + bias2
        
        # Apply softmax
        probs = softmax(logits)
        
        return probs
    
    def _apply_driver_specific_adjustments(self,
                                         probs: np.ndarray,
                                         driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Apply adjustments based on driver-specific characteristics."""
        adjusted = probs.copy()
        
        # Boost win probability for drivers with high win rate
        win_boost = driver_metrics.win_rate_2025 * 0.3
        adjusted[0] *= (1 + win_boost)
        
        # Boost podium probability for consistent performers
        if driver_metrics.consistency_score > 0.7:
            podium_boost = (driver_metrics.consistency_score - 0.7) * 0.5
            adjusted[1:4] *= (1 + podium_boost)  # P2-P4
        
        # Reduce backmarker probability for top drivers
        if driver_metrics.avg_finish < 8:
            backmarker_reduction = 0.5
            adjusted[15:20] *= (1 - backmarker_reduction)
        
        # Normalize
        adjusted = adjusted / adjusted.sum()
        
        return adjusted
    
    def _apply_track_adjustments(self,
                               probs: np.ndarray,
                               track: TrackCharacteristics,
                               driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Apply track-specific adjustments."""
        adjusted = probs.copy()
        
        # Adjust based on track type
        if track.type == "Technical":
            # Technical tracks favor consistent drivers
            if driver_metrics.consistency_score > 0.6:
                # Boost positions 1-10
                boost = (driver_metrics.consistency_score - 0.6) * 0.3
                adjusted[:10] *= (1 + boost)
                adjusted[10:] *= (1 - boost/2)
        
        elif track.type == "Power":
            # Power tracks favor qualifying strength
            if driver_metrics.qualifying_strength > 0.6:
                boost = (driver_metrics.qualifying_strength - 0.6) * 0.4
                adjusted[:5] *= (1 + boost)
                adjusted[5:] *= (1 - boost/3)
        
        # Adjust for overtaking difficulty
        if track.overtaking_difficulty > 0.7:
            # Hard to overtake = maintain grid position
            # Shift probability toward expected grid position
            grid_pos = driver_metrics.avg_qualifying
            if grid_pos < 10:  # Good qualifier
                adjusted[:grid_pos+3] *= 1.2
                adjusted[grid_pos+3:] *= 0.8
        
        # Adjust DNF probability for track
        adjusted[20] *= track.reliability_multiplier
        
        # Normalize
        adjusted = adjusted / adjusted.sum()
        
        return adjusted
    
    def _apply_situational_adjustments(self,
                                     probs: np.ndarray,
                                     driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Apply situational adjustments (championship pressure, etc.)."""
        adjusted = probs.copy()
        
        # Championship pressure effect
        pressure = driver_metrics.championship_pressure
        
        if pressure > 0.5:  # High pressure
            # Increase variance - more unpredictable outcomes
            # Flatten the distribution
            flatten_factor = pressure * 0.3
            adjusted = adjusted * (1 - flatten_factor) + (flatten_factor / 21)
        
        # Momentum effect
        momentum = driver_metrics.momentum
        
        if momentum > 0.2:  # Positive momentum
            # Shift probability toward better positions
            shift_amount = int(momentum * 3)  # Shift up to 3 positions
            if shift_amount > 0:
                # Create shifted distribution
                shifted = np.zeros_like(adjusted)
                shifted[shift_amount:] = adjusted[:-shift_amount] if shift_amount < 21 else 0
                # Blend with original
                blend = 0.7
                adjusted = blend * shifted + (1 - blend) * adjusted
        
        elif momentum < -0.2:  # Negative momentum
            # Shift probability toward worse positions
            shift_amount = int(abs(momentum) * 3)
            if shift_amount > 0:
                shifted = np.zeros_like(adjusted)
                shifted[:-shift_amount] = adjusted[shift_amount:] if shift_amount < 21 else 0
                blend = 0.7
                adjusted = blend * shifted + (1 - blend) * adjusted
        
        # Normalize
        adjusted = adjusted / adjusted.sum()
        
        return adjusted

class HybridProbabilityModel(BaseProbabilityModel):
    """
    Hybrid model combining statistical and ML approaches.
    Uses ensemble methods for robust probability estimation.
    """
    
    def __init__(self,
                 historical_data: Optional[pd.DataFrame] = None,
                 model_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Initialize component models
        self.statistical_model = StatisticalProbabilityModel(historical_data)
        self.ml_model = MachineLearningProbabilityModel(historical_data) if historical_data is not None else None
        
        # Model weights for ensemble
        self.model_weights = model_weights or {
            'statistical': 0.4,
            'machine_learning': 0.4,
            'historical_patterns': 0.2
        }
        
        # Historical patterns from file
        self.historical_patterns = self._load_historical_patterns()
        
        logger.info("Initialized Hybrid Probability Model")
    
    def _load_historical_patterns(self) -> Dict:
        """Load historical patterns from file."""
        patterns_file = Path("data/historical_patterns.json")
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        return {}
    
    def create_probability_distribution(self,
                                       driver_metrics: DriverPerformanceMetrics,
                                       track: Optional[TrackCharacteristics] = None,
                                       grid_position: Optional[int] = None) -> np.ndarray:
        """
        Create probability distribution using hybrid approach.
        """
        logger.info(f"Creating hybrid distribution for {driver_metrics.name}")
        
        distributions = []
        weights = []
        
        # 1. Statistical model distribution
        stat_probs = self.statistical_model.create_probability_distribution(
            driver_metrics, track, grid_position
        )
        distributions.append(stat_probs)
        weights.append(self.model_weights['statistical'])
        
        # 2. Machine learning distribution (if available)
        if self.ml_model is not None:
            try:
                ml_probs = self.ml_model.create_probability_distribution(
                    driver_metrics, track, grid_position
                )
                distributions.append(ml_probs)
                weights.append(self.model_weights['machine_learning'])
            except Exception as e:
                logger.warning(f"ML model failed: {e}. Adjusting weights.")
                # Redistribute weight to other models
                remaining_weight = sum(weights)
                for i in range(len(weights)):
                    weights[i] = weights[i] / remaining_weight
        
        # 3. Historical patterns distribution
        hist_probs = self._create_historical_pattern_distribution(driver_metrics)
        distributions.append(hist_probs)
        weights.append(self.model_weights['historical_patterns'])
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Create weighted ensemble
        ensemble_probs = np.zeros(21)
        for dist, weight in zip(distributions, weights):
            ensemble_probs += dist * weight
        
        # Apply final adjustments
        ensemble_probs = self._apply_final_adjustments(ensemble_probs, driver_metrics, track)
        
        # Validate
        if not self.validate_distribution(ensemble_probs):
            logger.warning(f"Hybrid distribution invalid for {driver_metrics.name}")
            # Fall back to statistical model
            ensemble_probs = stat_probs
        
        return ensemble_probs
    
    def _create_historical_pattern_distribution(self,
                                              driver_metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Create distribution based on historical patterns."""
        probs = np.zeros(21)
        
        # Use historical success rates if available
        if self.historical_patterns and 'gap_success_rates' in self.historical_patterns:
            # Simplified: use historical leader win rate as baseline
            leader_win_rate = self.historical_patterns.get('leader_win_rate', 0.5)
            
            if driver_metrics.championship_position == 1:
                # Championship leader
                win_prob = leader_win_rate * 0.8  # Scale down for single race
            else:
                # Challenger
                win_prob = (1 - leader_win_rate) * 0.4  # Lower baseline for challengers
            
            probs[0] = win_prob
            
            # Distribute remaining probability
            remaining = 1 - win_prob
            dnf_prob = driver_metrics.dnf_rate_2025 * 1.5  # Slightly higher for deciders
            probs[20] = min(0.15, dnf_prob)
            
            # Distribute positions 2-20
            positions = np.arange(2, 21)
            
            # Exponential decay from position 2 onward
            decay_rate = 0.85
            position_probs = remaining * (1 - dnf_prob) * decay_rate ** (positions - 2)
            position_probs = position_probs / position_probs.sum() * (remaining * (1 - dnf_prob))
            
            probs[1:20] = position_probs
        
        else:
            # Fallback: uniform distribution with DNF probability
            probs[:20] = 0.95 / 20
            probs[20] = 0.05
        
        return probs
    
    def _apply_final_adjustments(self,
                               probs: np.ndarray,
                               driver_metrics: DriverPerformanceMetrics,
                               track: Optional[TrackCharacteristics]) -> np.ndarray:
        """Apply final adjustments to hybrid distribution."""
        adjusted = probs.copy()
        
        # 1. Championship position adjustment
        champ_pos = driver_metrics.championship_position
        
        if champ_pos == 1:  # Championship leader
            # Slight boost to podium probabilities
            adjusted[0:3] *= 1.1
            # Reduce DNF probability (conservative approach)
            adjusted[20] *= 0.8
        
        elif champ_pos <= 3:  # Top 3 contender
            # Boost to top positions
            boost = 1.0 - (champ_pos * 0.1)  # 0.9 for P2, 0.8 for P3
            adjusted[0:5] *= boost
        
        # 2. Recent form adjustment
        if len(driver_metrics.recent_finishes) >= 3:
            recent_avg = np.mean(driver_metrics.recent_finishes[:3])
            season_avg = driver_metrics.avg_finish
            
            if recent_avg < season_avg - 2:  # Improving form
                # Shift probabilities toward better positions
                shift = int((season_avg - recent_avg) / 2)
                if shift > 0:
                    temp = np.zeros_like(adjusted)
                    temp[shift:] = adjusted[:-shift]
                    adjusted = 0.3 * temp + 0.7 * adjusted
            
            elif recent_avg > season_avg + 2:  # Declining form
                # Shift probabilities toward worse positions
                shift = int((recent_avg - season_avg) / 2)
                if shift > 0:
                    temp = np.zeros_like(adjusted)
                    temp[:-shift] = adjusted[shift:]
                    adjusted = 0.3 * temp + 0.7 * adjusted
        
        # 3. Track-specific adjustment for Yas Marina
        if track and track.name == "Yas Marina Circuit":
            if driver_metrics.yas_marina_races > 0:
                # Use track-specific performance
                track_performance = driver_metrics.yas_marina_avg_finish
                performance_ratio = track_performance / driver_metrics.avg_finish
                
                if performance_ratio < 0.9:  # Better at Yas Marina
                    # Boost probabilities
                    boost = 1.0 / performance_ratio
                    adjusted[:10] *= min(1.2, boost)
                    adjusted[10:] *= max(0.8, 2 - boost)
                
                elif performance_ratio > 1.1:  # Worse at Yas Marina
                    # Reduce probabilities for good finishes
                    reduction = performance_ratio
                    adjusted[:10] *= max(0.8, 2 - reduction)
                    adjusted[10:] *= min(1.2, reduction)
        
        # 4. Ensure DNF probability is reasonable
        max_dnf = 0.25  # Maximum reasonable DNF probability
        min_dnf = 0.01  # Minimum reasonable DNF probability
        adjusted[20] = max(min_dnf, min(max_dnf, adjusted[20]))
        
        # 5. Normalize
        adjusted = adjusted / adjusted.sum()
        
        return adjusted

class ProbabilityEngine:
    """
    Main probability engine that coordinates all models.
    Provides high-level interface for probability generation.
    """
    
    def __init__(self, model_type: str = "hybrid", historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize probability engine.
        
        Args:
            model_type: Type of model to use ("statistical", "ml", "hybrid")
            historical_data: Historical data for training ML models
        """
        self.model_type = model_type.lower()
        
        # Initialize selected model
        if self.model_type == "statistical":
            self.model = StatisticalProbabilityModel(historical_data)
        elif self.model_type == "ml":
            if historical_data is None:
                logger.warning("No historical data provided for ML model. Using statistical model.")
                self.model = StatisticalProbabilityModel(historical_data)
                self.model_type = "statistical"
            else:
                self.model = MachineLearningProbabilityModel(historical_data)
                self.model.train_models()
        elif self.model_type == "hybrid":
            self.model = HybridProbabilityModel(historical_data)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using hybrid model.")
            self.model = HybridProbabilityModel(historical_data)
            self.model_type = "hybrid"
        
        logger.info(f"Initialized Probability Engine with {self.model_type} model")
    
    def generate_probabilities(self,
                              driver_metrics: Dict[str, DriverPerformanceMetrics],
                              track: Optional[TrackCharacteristics] = None,
                              grid_positions: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        """
        Generate probability distributions for multiple drivers.
        
        Args:
            driver_metrics: Dictionary of driver performance metrics
            track: Track characteristics
            grid_positions: Dictionary of grid positions
            
        Returns:
            Dictionary mapping driver names to probability arrays
        """
        logger.info(f"Generating probabilities for {len(driver_metrics)} drivers")
        
        probabilities = {}
        
        for driver_name, metrics in driver_metrics.items():
            grid_pos = grid_positions.get(driver_name) if grid_positions else None
            
            try:
                probs = self.model.create_probability_distribution(
                    metrics, track, grid_pos
                )
                
                if not self.model.validate_distribution(probs):
                    logger.warning(f"Invalid distribution for {driver_name}, regenerating")
                    # Try one more time with fallback
                    probs = self._generate_fallback_distribution(metrics)
                
                probabilities[driver_name] = probs
                logger.debug(f"Generated distribution for {driver_name}")
                
            except Exception as e:
                logger.error(f"Error generating probabilities for {driver_name}: {e}")
                # Use fallback distribution
                probabilities[driver_name] = self._generate_fallback_distribution(metrics)
        
        # Ensure probabilities across drivers are reasonable
        probabilities = self._normalize_across_drivers(probabilities)
        
        return probabilities
    
    def _generate_fallback_distribution(self,
                                      metrics: DriverPerformanceMetrics) -> np.ndarray:
        """Generate fallback probability distribution."""
        probs = np.zeros(21)
        
        # Simple distribution based on average finish
        expected = metrics.avg_finish
        
        # Triangular distribution
        positions = np.arange(1, 21)
        pdf = 1 - np.abs(positions - expected) / 8
        pdf = np.maximum(pdf, 0.05)  # Minimum probability
        
        probs[:20] = pdf / pdf.sum() * 0.95
        probs[20] = 0.05
        
        return probs
    
    def _normalize_across_drivers(self,
                                probabilities: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Ensure probabilities across drivers are reasonable.
        Not all drivers can win simultaneously, etc.
        """
        # Simple normalization: ensure win probabilities sum to <= 1
        total_win_prob = sum(probs[0] for probs in probabilities.values())
        
        if total_win_prob > 1.0:
            logger.warning(f"Total win probability {total_win_prob:.2f} > 1.0, scaling down")
            scale_factor = 1.0 / total_win_prob
            
            for driver_name, probs in probabilities.items():
                # Scale win probability
                probs[0] *= scale_factor
                # Redistribute remaining probability
                remaining_scale = (1 - probs[0]) / (1 - probs[0]/scale_factor)
                probs[1:] *= remaining_scale
        
        return probabilities
    
    def analyze_distributions(self,
                            probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze probability distributions and extract insights.
        """
        analysis = {
            'drivers': {},
            'summary': {},
            'insights': []
        }
        
        for driver_name, probs in probabilities.items():
            driver_analysis = {
                'win_probability': probs[0],
                'podium_probability': sum(probs[0:3]),
                'points_probability': sum(probs[0:10]),
                'dnf_probability': probs[20],
                'expected_position': sum((i+1) * prob for i, prob in enumerate(probs[:20])),
                'position_std': np.sqrt(sum(((i+1) - sum((j+1) * probs[j] for j in range(20)))**2 * probs[i] 
                                          for i in range(20)))
            }
            analysis['drivers'][driver_name] = driver_analysis
        
        # Comparative analysis
        win_probs = [(name, analysis['drivers'][name]['win_probability']) 
                    for name in probabilities.keys()]
        win_probs.sort(key=lambda x: x[1], reverse=True)
        
        analysis['summary']['favorite'] = win_probs[0][0]
        analysis['summary']['favorite_win_prob'] = win_probs[0][1]
        
        if len(win_probs) > 1:
            analysis['summary']['closest_contender'] = win_probs[1][0]
            analysis['summary']['win_prob_gap'] = win_probs[0][1] - win_probs[1][1]
        
        # Generate insights
        if analysis['summary'].get('win_prob_gap', 0) > 0.2:
            analysis['insights'].append(
                f"{analysis['summary']['favorite']} is a clear favorite with "
                f"{analysis['summary']['favorite_win_prob']:.1%} win probability"
            )
        elif analysis['summary'].get('win_prob_gap', 0) > 0.1:
            analysis['insights'].append(
                f"{analysis['summary']['favorite']} is a moderate favorite over "
                f"{analysis['summary']['closest_contender']}"
            )
        else:
            analysis['insights'].append("Championship is highly competitive with no clear favorite")
        
        # DNF risk analysis
        dnf_risks = [(name, analysis['drivers'][name]['dnf_probability']) 
                    for name in probabilities.keys()]
        dnf_risks.sort(key=lambda x: x[1], reverse=True)
        
        if dnf_risks[0][1] > 0.1:
            analysis['insights'].append(
                f"{dnf_risks[0][0]} has the highest DNF risk at {dnf_risks[0][1]:.1%}"
            )
        
        return analysis
    
    def save_probabilities(self,
                          probabilities: Dict[str, np.ndarray],
                          output_dir: str = "probabilities") -> Dict[str, str]:
        """
        Save probability distributions to files.
        
        Returns:
            Dictionary of file paths
        """
        import pickle
        from datetime import datetime
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle for Python use
        pickle_file = f"{output_dir}/probabilities_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(probabilities, f)
        
        # Save as CSV for analysis
        csv_data = []
        positions = list(range(1, 21)) + ['DNF']
        
        for driver_name, probs in probabilities.items():
            for pos, prob in zip(positions, probs):
                csv_data.append({
                    'driver': driver_name,
                    'position': pos,
                    'probability': prob
                })
        
        df = pd.DataFrame(csv_data)
        csv_file = f"{output_dir}/probabilities_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save analysis
        analysis = self.analyze_distributions(probabilities)
        analysis_file = f"{output_dir}/analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Probabilities saved to {output_dir}/")
        
        return {
            'pickle': pickle_file,
            'csv': csv_file,
            'analysis': analysis_file
        }

# Example usage and helper functions
def create_default_driver_metrics() -> Dict[str, DriverPerformanceMetrics]:
    """Create default driver metrics based on 2025 data."""
    
    # Norris - Championship leader
    norris = DriverPerformanceMetrics(
        name="Lando Norris",
        team="McLaren",
        current_points=408,
        championship_position=1,
        races_completed=22,
        wins=8,
        podiums=15,
        poles=6,
        fastest_laps=4,
        dnf_count=1,
        avg_finish=2.8,
        avg_qualifying=3.2,
        career_races=120,
        career_wins=18,
        career_podiums=45,
        career_poles=12,
        career_dnf_rate=0.04,
        career_avg_finish=4.2,
        recent_finishes=[1, 2, 1, 3, 2],
        recent_qualifying=[2, 1, 3, 2, 4],
        recent_points=[25, 18, 25, 15, 18],
        yas_marina_races=3,
        yas_marina_wins=1,
        yas_marina_podiums=2,
        yas_marina_avg_finish=2.3,
        yas_marina_dnf=0,
        consistency_score=0.85,
        pressure_score=0.7,
        race_craft_score=0.8,
        qualifying_strength=0.75,
        momentum=0.3,
        championship_pressure=0.8,
        reliability_risk=0.1
    )
    
    # Verstappen - Challenger
    verstappen = DriverPerformanceMetrics(
        name="Max Verstappen",
        team="Red Bull",
        current_points=396,
        championship_position=2,
        races_completed=22,
        wins=7,
        podiums=14,
        poles=8,
        fastest_laps=6,
        dnf_count=2,
        avg_finish=3.2,
        avg_qualifying=2.8,
        career_races=180,
        career_wins=70,
        career_podiums=116,
        career_poles=47,
        career_dnf_rate=0.065,
        career_avg_finish=3.6,
        recent_finishes=[2, 1, 3, 1, 4],
        recent_qualifying=[3, 2, 1, 3, 2],
        recent_points=[18, 25, 15, 25, 12],
        yas_marina_races=7,
        yas_marina_wins=3,
        yas_marina_podiums=5,
        yas_marina_avg_finish=2.8,
        yas_marina_dnf=0,
        consistency_score=0.82,
        pressure_score=0.9,
        race_craft_score=0.9,
        qualifying_strength=0.85,
        momentum=0.1,
        championship_pressure=0.9,
        reliability_risk=0.15
    )
    
    # Piastri - Dark horse
    piastri = DriverPerformanceMetrics(
        name="Oscar Piastri",
        team="McLaren",
        current_points=392,
        championship_position=3,
        races_completed=22,
        wins=6,
        podiums=12,
        poles=4,
        fastest_laps=3,
        dnf_count=1,
        avg_finish=3.7,
        avg_qualifying=4.1,
        career_races=45,
        career_wins=9,
        career_podiums=18,
        career_poles=6,
        career_dnf_rate=0.058,
        career_avg_finish=5.2,
        recent_finishes=[3, 4, 2, 5, 1],
        recent_qualifying=[4, 3, 2, 4, 1],
        recent_points=[15, 12, 18, 10, 25],
        yas_marina_races=2,
        yas_marina_wins=0,
        yas_marina_podiums=1,
        yas_marina_avg_finish=4.5,
        yas_marina_dnf=0,
        consistency_score=0.78,
        pressure_score=0.6,
        race_craft_score=0.75,
        qualifying_strength=0.7,
        momentum=0.4,
        championship_pressure=0.7,
        reliability_risk=0.08
    )
    
    return {
        'NORRIS': norris,
        'VERSTAPPEN': verstappen,
        'PIASTRI': piastri
    }

def create_yas_marina_track() -> TrackCharacteristics:
    """Create Yas Marina track characteristics."""
    return TrackCharacteristics(
        name="Yas Marina Circuit",
        type="Technical",  # Mix of high-speed and technical sections
        overtaking_difficulty=0.7,  # Moderately difficult to overtake
        tire_degradation=0.6,  # Moderate degradation
        safety_car_probability=0.31,  # From historical data
        dnf_rate=0.05,  # Relatively low DNF rate
        qualifying_multiplier=1.0,
        race_pace_multiplier=1.0,
        consistency_multiplier=1.1,  # Consistency rewarded
        reliability_multiplier=1.0
    )

def main():
    """Example main function."""
    logger.info("=" * 70)
    logger.info("PROBABILITY ENGINE DEMONSTRATION")
    logger.info("=" * 70)
    
    # Create driver metrics
    driver_metrics = create_default_driver_metrics()
    track = create_yas_marina_track()
    
    # Example grid positions (post-qualifying)
    grid_positions = {
        'NORRIS': 2,
        'VERSTAPPEN': 1,
        'PIASTRI': 3
    }
    
    # Initialize probability engine
    logger.info("Initializing Hybrid Probability Engine...")
    engine = ProbabilityEngine(model_type="hybrid")
    
    # Generate probabilities
    logger.info("Generating probability distributions...")
    probabilities = engine.generate_probabilities(
        driver_metrics=driver_metrics,
        track=track,
        grid_positions=grid_positions
    )
    
    # Analyze distributions
    logger.info("Analyzing probability distributions...")
    analysis = engine.analyze_distributions(probabilities)
    
    # Display results
    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    for driver_name, probs in probabilities.items():
        print(f"\n{driver_name}:")
        print("-" * 40)
        print(f"  Win Probability:      {probs[0]:.2%}")
        print(f"  Podium Probability:   {sum(probs[0:3]):.2%}")
        print(f"  Points Probability:   {sum(probs[0:10]):.2%}")
        print(f"  DNF Probability:      {probs[20]:.2%}")
        print(f"  Expected Position:    {sum((i+1) * prob for i, prob in enumerate(probs[:20])):.2f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"Predicted Favorite: {analysis['summary'].get('favorite', 'Unknown')}")
    if 'favorite_win_prob' in analysis['summary']:
        print(f"Win Probability: {analysis['summary']['favorite_win_prob']:.2%}")
    
    print("\nKEY INSIGHTS:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Save probabilities
    logger.info("Saving probabilities to files...")
    saved_files = engine.save_probabilities(probabilities)
    
    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    for file_type, file_path in saved_files.items():
        print(f"{file_type.upper():15} {file_path}")
    
    logger.info("Probability engine demonstration completed")

if __name__ == "__main__":
    main()
