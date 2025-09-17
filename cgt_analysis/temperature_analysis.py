"""
Temperature Analysis for Combinatorial Games

This module implements comprehensive temperature calculations and
thermographic analysis for CGT positions, with focus on the
mean-preserving temperature oscillation principle.
"""

from typing import List, Dict, Tuple, Optional, Union
import math
from fractions import Fraction
from .cgt_position import CGTPosition

class TemperatureCalculator:
    """
    Calculates game temperatures and performs thermographic analysis.
    
    Temperature measures how much a game's value changes based on
    who moves first. This is crucial for understanding the 2^n×k
    structural resonance principle.
    """
    
    def __init__(self):
        """Initialize temperature calculator with caching"""
        self.temperature_cache: Dict[str, float] = {}
        self.thermograph_cache: Dict[str, Dict] = {}
        
    def compute_temperature(self, position: CGTPosition, precision: float = 0.001) -> float:
        """
        Compute the temperature of a position.
        
        Temperature is the value t such that the game cools to an integer
        when both players pass on all moves with value < t.
        
        Args:
            position: CGTPosition to analyze
            precision: Numerical precision for temperature calculation
            
        Returns:
            Temperature value (non-negative float)
        """
        cache_key = self._position_hash(position)
        if cache_key in self.temperature_cache:
            return self.temperature_cache[cache_key]
        
        temperature = self._compute_temperature_recursive(position, precision)
        self.temperature_cache[cache_key] = temperature
        return temperature
    
    def _compute_temperature_recursive(self, position: CGTPosition, precision: float) -> float:
        """Recursive temperature computation"""
        
        if position.is_terminal():
            return 0.0
        
        # Get the game value
        game_value = position.compute_game_value()
        
        # For confused positions (NaN), temperature is typically high
        if isinstance(game_value, float) and game_value != game_value:  # NaN check
            return self._estimate_confused_temperature(position)
        
        # Compute left and right incentives
        left_incentive = self._compute_left_incentive(position)
        right_incentive = self._compute_right_incentive(position)
        
        # Temperature is the minimum of left and right incentives
        temperature = min(left_incentive, right_incentive)
        
        return max(0.0, temperature)  # Temperature is non-negative
    
    def _compute_left_incentive(self, position: CGTPosition) -> float:
        """Compute left player's incentive to move"""
        if not position.left_options:
            return 0.0
        
        # Left incentive is the maximum advantage Left can gain by moving
        game_value = float(position.compute_game_value())
        left_values = [float(opt.compute_game_value()) for opt in position.left_options]
        
        if not left_values:
            return 0.0
        
        max_left_value = max(left_values)
        return max(0.0, max_left_value - game_value)
    
    def _compute_right_incentive(self, position: CGTPosition) -> float:
        """Compute right player's incentive to move"""
        if not position.right_options:
            return 0.0
        
        # Right incentive is the maximum advantage Right can gain by moving
        game_value = float(position.compute_game_value())
        right_values = [float(opt.compute_game_value()) for opt in position.right_options]
        
        if not right_values:
            return 0.0
        
        min_right_value = min(right_values)
        return max(0.0, game_value - min_right_value)
    
    def _estimate_confused_temperature(self, position: CGTPosition) -> float:
        """Estimate temperature for confused positions"""
        # For confused positions, use the spread of option values
        all_values = []
        
        for opt in position.left_options:
            val = opt.compute_game_value()
            if not (isinstance(val, float) and val != val):  # Not NaN
                all_values.append(float(val))
        
        for opt in position.right_options:
            val = opt.compute_game_value()
            if not (isinstance(val, float) and val != val):  # Not NaN
                all_values.append(float(val))
        
        if len(all_values) < 2:
            return 1.0  # Default temperature for confused positions
        
        return (max(all_values) - min(all_values)) / 2.0
    
    def compute_mean_value(self, position: CGTPosition) -> float:
        """
        Compute the mean value of a position.
        
        Mean value is the positional advantage independent of temperature.
        This is crucial for analyzing mean-preserving temperature oscillation.
        
        Args:
            position: CGTPosition to analyze
            
        Returns:
            Mean value (can be positive, negative, or zero)
        """
        if position.is_terminal():
            return position.get_game_value() if hasattr(position, 'get_game_value') else 0.0
        
        # Mean value is typically the average of the game value under
        # optimal play for both sides
        game_value = position.compute_game_value()
        
        if isinstance(game_value, float) and game_value != game_value:  # NaN
            return 0.0  # Confused positions have mean value 0
        
        return float(game_value)
    
    def generate_thermograph(self, position: CGTPosition, temperature_range: Tuple[float, float] = (0.0, 5.0), 
                           num_points: int = 100) -> Dict[str, any]:
        """
        Generate a complete thermograph for a position.
        
        A thermograph shows how the game value changes as temperature varies.
        This is essential for understanding temperature dynamics.
        
        Args:
            position: Position to analyze
            temperature_range: (min_temp, max_temp) to analyze
            num_points: Number of temperature points to sample
            
        Returns:
            Dictionary with thermograph data
        """
        cache_key = f"{self._position_hash(position)}_{temperature_range}_{num_points}"
        if cache_key in self.thermograph_cache:
            return self.thermograph_cache[cache_key]
        
        min_temp, max_temp = temperature_range
        temp_step = (max_temp - min_temp) / (num_points - 1)
        
        temperatures = []
        left_values = []
        right_values = []
        
        for i in range(num_points):
            temp = min_temp + i * temp_step
            left_val, right_val = self._compute_values_at_temperature(position, temp)
            
            temperatures.append(temp)
            left_values.append(left_val)
            right_values.append(right_val)
        
        # Find critical temperatures (where slopes change)
        critical_temps = self._find_critical_temperatures(position, temperature_range)
        
        # Compute thermograph properties
        actual_temperature = self.compute_temperature(position)
        mean_value = self.compute_mean_value(position)
        
        thermograph = {
            'position_name': position.position_name,
            'temperatures': temperatures,
            'left_values': left_values,
            'right_values': right_values,
            'critical_temperatures': critical_temps,
            'actual_temperature': actual_temperature,
            'mean_value': mean_value,
            'temperature_range': temperature_range,
            'is_number': actual_temperature == 0.0,
            'thermograph_type': self._classify_thermograph(left_values, right_values, actual_temperature)
        }
        
        self.thermograph_cache[cache_key] = thermograph
        return thermograph
    
    def _compute_values_at_temperature(self, position: CGTPosition, temperature: float) -> Tuple[float, float]:
        """
        Compute left and right values at a specific temperature.
        
        At temperature t, players pass on all moves with incentive < t.
        """
        if position.is_terminal():
            value = position.get_game_value() if hasattr(position, 'get_game_value') else 0.0
            return value, value
        
        # Compute available moves at this temperature
        available_left_options = []
        available_right_options = []
        
        game_value = float(position.compute_game_value())
        
        # Left options available if incentive >= temperature
        for opt in position.left_options:
            opt_value = float(opt.compute_game_value())
            incentive = opt_value - game_value
            if incentive >= temperature:
                available_left_options.append(opt)
        
        # Right options available if incentive >= temperature
        for opt in position.right_options:
            opt_value = float(opt.compute_game_value())
            incentive = game_value - opt_value
            if incentive >= temperature:
                available_right_options.append(opt)
        
        # Compute values with available options
        if available_left_options:
            left_values = [self._compute_values_at_temperature(opt, temperature)[0] for opt in available_left_options]
            left_value = max(left_values)
        else:
            left_value = game_value
        
        if available_right_options:
            right_values = [self._compute_values_at_temperature(opt, temperature)[1] for opt in available_right_options]
            right_value = min(right_values)
        else:
            right_value = game_value
        
        return left_value, right_value
    
    def _find_critical_temperatures(self, position: CGTPosition, temp_range: Tuple[float, float]) -> List[float]:
        """Find temperatures where the thermograph changes slope"""
        critical_temps = []
        
        # Add the actual temperature of the position
        actual_temp = self.compute_temperature(position)
        if temp_range[0] <= actual_temp <= temp_range[1]:
            critical_temps.append(actual_temp)
        
        # Add temperatures of all options
        for opt in position.left_options + position.right_options:
            opt_temp = self.compute_temperature(opt)
            if temp_range[0] <= opt_temp <= temp_range[1]:
                critical_temps.append(opt_temp)
        
        # Remove duplicates and sort
        critical_temps = sorted(list(set(critical_temps)))
        
        return critical_temps
    
    def _classify_thermograph(self, left_values: List[float], right_values: List[float], 
                             temperature: float) -> str:
        """Classify the type of thermograph"""
        if temperature == 0.0:
            return "number"
        
        # Check if left and right values are parallel
        left_diffs = [left_values[i+1] - left_values[i] for i in range(len(left_values)-1)]
        right_diffs = [right_values[i+1] - right_values[i] for i in range(len(right_values)-1)]
        
        if all(abs(d) < 0.001 for d in left_diffs + right_diffs):
            return "constant"
        
        # Check if they converge
        final_diff = abs(left_values[-1] - right_values[-1])
        initial_diff = abs(left_values[0] - right_values[0])
        
        if final_diff < initial_diff / 2:
            return "converging"
        elif final_diff > initial_diff * 2:
            return "diverging"
        else:
            return "parallel"
    
    def analyze_temperature_sequence(self, positions: List[CGTPosition]) -> Dict[str, any]:
        """
        Analyze temperature patterns across a sequence of positions.
        
        This is crucial for detecting the temperature oscillation patterns
        predicted by the 2^n×k principle.
        
        Args:
            positions: Ordered sequence of positions to analyze
            
        Returns:
            Dictionary with temperature sequence analysis
        """
        temperatures = [self.compute_temperature(pos) for pos in positions]
        mean_values = [self.compute_mean_value(pos) for pos in positions]
        
        # Statistical analysis
        temp_stats = self._compute_sequence_statistics(temperatures)
        mean_stats = self._compute_sequence_statistics(mean_values)
        
        # Oscillation analysis
        oscillation_analysis = self._analyze_oscillations(temperatures)
        
        # Periodicity analysis
        periodicity_analysis = self._analyze_temperature_periodicity(temperatures)
        
        # Mean preservation analysis
        mean_preservation = self._analyze_mean_preservation(mean_values)
        
        return {
            'sequence_length': len(positions),
            'temperatures': temperatures,
            'mean_values': mean_values,
            'temperature_statistics': temp_stats,
            'mean_value_statistics': mean_stats,
            'oscillation_analysis': oscillation_analysis,
            'periodicity_analysis': periodicity_analysis,
            'mean_preservation_analysis': mean_preservation,
            'temperature_range': (min(temperatures), max(temperatures)),
            'mean_value_range': (min(mean_values), max(mean_values))
        }
    
    def _compute_sequence_statistics(self, sequence: List[float]) -> Dict[str, float]:
        """Compute statistical properties of a sequence"""
        if not sequence:
            return {}
        
        n = len(sequence)
        mean = sum(sequence) / n
        variance = sum((x - mean) ** 2 for x in sequence) / n
        std_dev = math.sqrt(variance)
        
        return {
            'mean': mean,
            'variance': variance,
            'standard_deviation': std_dev,
            'min': min(sequence),
            'max': max(sequence),
            'range': max(sequence) - min(sequence)
        }
    
    def _analyze_oscillations(self, temperatures: List[float]) -> Dict[str, any]:
        """Analyze oscillation patterns in temperature sequence"""
        if len(temperatures) < 3:
            return {'insufficient_data': True}
        
        # Find peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(temperatures) - 1):
            if temperatures[i] > temperatures[i-1] and temperatures[i] > temperatures[i+1]:
                peaks.append((i, temperatures[i]))
            elif temperatures[i] < temperatures[i-1] and temperatures[i] < temperatures[i+1]:
                valleys.append((i, temperatures[i]))
        
        # Compute oscillation frequency
        total_oscillations = len(peaks) + len(valleys)
        oscillation_frequency = total_oscillations / len(temperatures) if temperatures else 0
        
        # Compute average amplitude
        if peaks and valleys:
            peak_values = [p[1] for p in peaks]
            valley_values = [v[1] for v in valleys]
            avg_amplitude = (sum(peak_values) / len(peak_values) - sum(valley_values) / len(valley_values)) / 2
        else:
            avg_amplitude = 0.0
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'oscillation_frequency': oscillation_frequency,
            'average_amplitude': avg_amplitude,
            'is_oscillating': total_oscillations > 0
        }
    
    def _analyze_temperature_periodicity(self, temperatures: List[float], tolerance: float = 0.1) -> Dict[str, any]:
        """Analyze periodic patterns in temperature sequence"""
        periods = []
        
        # Check for periods up to half the sequence length
        for period in range(1, len(temperatures) // 2 + 1):
            if self._is_approximately_periodic(temperatures, period, tolerance):
                periods.append(period)
        
        # Special check for 16-card periodicity
        has_16_period = 16 in periods if periods else False
        
        return {
            'detected_periods': periods,
            'fundamental_period': periods[0] if periods else None,
            'has_16_card_period': has_16_period,
            'is_periodic': len(periods) > 0,
            'period_strength': self._compute_period_strength(temperatures, periods[0]) if periods else 0.0
        }
    
    def _is_approximately_periodic(self, sequence: List[float], period: int, tolerance: float) -> bool:
        """Check if sequence is approximately periodic with given period"""
        if period >= len(sequence):
            return False
        
        for i in range(len(sequence) - period):
            if abs(sequence[i] - sequence[i + period]) > tolerance:
                return False
        
        return True
    
    def _compute_period_strength(self, sequence: List[float], period: int) -> float:
        """Compute how strongly periodic a sequence is"""
        if not period or period >= len(sequence):
            return 0.0
        
        deviations = []
        for i in range(len(sequence) - period):
            deviations.append(abs(sequence[i] - sequence[i + period]))
        
        if not deviations:
            return 0.0
        
        avg_deviation = sum(deviations) / len(deviations)
        sequence_range = max(sequence) - min(sequence)
        
        if sequence_range == 0:
            return 1.0
        
        # Strength is inversely related to average deviation
        return max(0.0, 1.0 - (avg_deviation / sequence_range))
    
    def _analyze_mean_preservation(self, mean_values: List[float], tolerance: float = 0.05) -> Dict[str, any]:
        """Analyze how well mean values are preserved (stay near zero)"""
        if not mean_values:
            return {}
        
        # Check if mean values oscillate around zero
        overall_mean = sum(mean_values) / len(mean_values)
        deviations_from_zero = [abs(val) for val in mean_values]
        avg_deviation = sum(deviations_from_zero) / len(deviations_from_zero)
        
        # Check preservation quality
        is_well_preserved = abs(overall_mean) < tolerance and avg_deviation < tolerance * 2
        
        return {
            'overall_mean': overall_mean,
            'average_deviation_from_zero': avg_deviation,
            'is_mean_preserving': is_well_preserved,
            'preservation_quality': max(0.0, 1.0 - avg_deviation / 1.0),  # Normalize to [0,1]
            'max_deviation': max(deviations_from_zero),
            'values_near_zero': sum(1 for val in mean_values if abs(val) < tolerance)
        }
    
    def _position_hash(self, position: CGTPosition) -> str:
        """Generate hash key for position caching"""
        if position.war_position:
            return position.war_position.hash_key()
        else:
            return position.position_name