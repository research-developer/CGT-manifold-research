"""
Mathematical Verification Module for CGT Analysis

This module provides rigorous verification functions to ensure 
mathematical consistency and identify potential errors in 
CGT calculations and theoretical claims.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
from fractions import Fraction
from .cgt_position import CGTPosition
from .grundy_numbers import GrundyCalculator
from .temperature_analysis import TemperatureCalculator


class MathematicalVerifier:
    """
    Comprehensive verification of mathematical properties in CGT analysis.
    
    This class implements rigorous checks for:
    - Grundy number calculation consistency
    - Temperature computation accuracy
    - Game value correctness
    - Theoretical claim validation
    """
    
    def __init__(self):
        """Initialize verifier with calculators"""
        self.grundy_calc = GrundyCalculator()
        self.temp_calc = TemperatureCalculator()
        self.verification_results = {}
        
    def verify_position_consistency(self, position: CGTPosition) -> Dict[str, Any]:
        """
        Verify mathematical consistency of a single position.
        
        Args:
            position: CGTPosition to verify
            
        Returns:
            Dictionary with verification results and any inconsistencies found
        """
        results = {
            'position_name': position.position_name,
            'inconsistencies_found': [],
            'warnings': [],
            'verification_passed': True
        }
        
        # Verify game value calculation
        game_value_check = self._verify_game_value(position)
        results['game_value_verification'] = game_value_check
        if not game_value_check['consistent']:
            results['inconsistencies_found'].append('Game value calculation inconsistent')
            results['verification_passed'] = False
        
        # Verify Grundy number if applicable
        grundy_check = self._verify_grundy_number(position)
        results['grundy_verification'] = grundy_check
        if not grundy_check['consistent']:
            results['inconsistencies_found'].append('Grundy number calculation inconsistent')
            results['verification_passed'] = False
        
        # Verify temperature calculation
        temp_check = self._verify_temperature(position)
        results['temperature_verification'] = temp_check
        if not temp_check['consistent']:
            results['inconsistencies_found'].append('Temperature calculation inconsistent')
            results['verification_passed'] = False
        
        # Verify CGT notation consistency
        notation_check = self._verify_cgt_notation(position)
        results['notation_verification'] = notation_check
        if not notation_check['consistent']:
            results['warnings'].append('CGT notation may be incomplete')
        
        return results
    
    def _verify_game_value(self, position: CGTPosition) -> Dict[str, Any]:
        """Verify game value calculation follows CGT principles"""
        try:
            value = position.compute_game_value()
            
            if position.is_terminal():
                # Terminal positions should have value 0
                expected = 0
                consistent = (value == expected or abs(float(value) - expected) < 1e-10)
                return {
                    'consistent': consistent,
                    'computed_value': value,
                    'expected_value': expected,
                    'method': 'terminal_position_check'
                }
            
            # For non-terminal positions, verify recursive definition
            left_values = []
            right_values = []
            
            for left_opt in position.left_options:
                left_values.append(left_opt.compute_game_value())
            
            for right_opt in position.right_options:
                right_values.append(right_opt.compute_game_value())
            
            # Check Conway's recursive definition
            max_left = max([float(v) for v in left_values]) if left_values else float('-inf')
            min_right = min([float(v) for v in right_values]) if right_values else float('inf')
            
            if max_left >= min_right:
                # Should be confused (NaN or similar)
                is_confused = isinstance(value, float) and (value != value or not math.isfinite(value))
                return {
                    'consistent': is_confused,
                    'computed_value': value,
                    'explanation': f'Position should be confused: max_left={max_left} >= min_right={min_right}',
                    'method': 'confused_position_check'
                }
            else:
                # Should be a number in interval (max_left, min_right)
                if isinstance(value, float) and (value != value or not math.isfinite(value)):
                    return {
                        'consistent': False,
                        'computed_value': value,
                        'explanation': f'Position should be number in ({max_left}, {min_right}) but got confused',
                        'method': 'number_position_check'
                    }
                
                value_float = float(value)
                in_interval = max_left < value_float < min_right
                return {
                    'consistent': in_interval,
                    'computed_value': value,
                    'interval': (max_left, min_right),
                    'in_interval': in_interval,
                    'method': 'interval_check'
                }
                
        except Exception as e:
            return {
                'consistent': False,
                'error': str(e),
                'method': 'exception_caught'
            }
    
    def _verify_grundy_number(self, position: CGTPosition) -> Dict[str, Any]:
        """Verify Grundy number calculation"""
        try:
            # Compute using main algorithm
            grundy1 = self.grundy_calc.compute_grundy_number(position)
            
            # Compute using verification algorithm
            verification_result = self.grundy_calc._independent_verification(position)
            grundy2 = verification_result['grundy_number']
            
            consistent = (grundy1 == grundy2)
            
            return {
                'consistent': consistent,
                'main_algorithm': grundy1,
                'verification_algorithm': grundy2,
                'verification_details': verification_result,
                'method': 'dual_algorithm_comparison'
            }
            
        except Exception as e:
            return {
                'consistent': False,
                'error': str(e),
                'method': 'exception_caught'
            }
    
    def _verify_temperature(self, position: CGTPosition) -> Dict[str, Any]:
        """Verify temperature calculation"""
        try:
            temp = self.temp_calc.compute_temperature(position)
            
            # Temperature should be non-negative
            non_negative = temp >= 0
            
            # For terminal positions, temperature should be 0
            if position.is_terminal():
                correct_terminal = (temp == 0)
                return {
                    'consistent': non_negative and correct_terminal,
                    'computed_temperature': temp,
                    'non_negative': non_negative,
                    'correct_terminal': correct_terminal,
                    'method': 'terminal_temperature_check'
                }
            
            # For non-terminal positions, verify it's reasonable
            finite = math.isfinite(temp)
            
            return {
                'consistent': non_negative and finite,
                'computed_temperature': temp,
                'non_negative': non_negative,
                'finite': finite,
                'method': 'general_temperature_check'
            }
            
        except Exception as e:
            return {
                'consistent': False,
                'error': str(e),
                'method': 'exception_caught'
            }
    
    def _verify_cgt_notation(self, position: CGTPosition) -> Dict[str, Any]:
        """Verify CGT notation is sensible"""
        try:
            notation = position.get_cgt_notation()
            
            # Should contain | symbol unless terminal
            has_pipe = '|' in notation
            
            if position.is_terminal():
                expected_notation = notation == "0"
                return {
                    'consistent': expected_notation,
                    'notation': notation,
                    'expected': "0",
                    'method': 'terminal_notation_check'
                }
            else:
                return {
                    'consistent': has_pipe,
                    'notation': notation,
                    'has_pipe_symbol': has_pipe,
                    'method': 'general_notation_check'
                }
                
        except Exception as e:
            return {
                'consistent': False,
                'error': str(e),
                'method': 'exception_caught'
            }
    
    def verify_theoretical_claims(self, positions_by_deck_size: Dict[int, List[CGTPosition]]) -> Dict[str, Any]:
        """
        Verify theoretical claims about the 2^n×k principle and related properties.
        
        Args:
            positions_by_deck_size: Dictionary mapping deck sizes to position lists
            
        Returns:
            Verification results for theoretical claims
        """
        results = {
            'claims_verified': {},
            'inconsistencies': [],
            'recommendations': []
        }
        
        # Verify claim: "Grundy number analysis: G(44) ≠ 0, G(48) = 0, G(52) ≠ 0"
        grundy_claim = self._verify_grundy_claim(positions_by_deck_size)
        results['claims_verified']['grundy_pattern'] = grundy_claim
        
        # Verify claim: "Temperature maximizes at 48 cards while maintaining mean value = 0"
        temp_claim = self._verify_temperature_claim(positions_by_deck_size)
        results['claims_verified']['temperature_maximum'] = temp_claim
        
        # Verify claim: "Mean-preserving temperature oscillation"
        mean_preservation_claim = self._verify_mean_preservation(positions_by_deck_size)
        results['claims_verified']['mean_preservation'] = mean_preservation_claim
        
        # Collect inconsistencies
        for claim_name, claim_result in results['claims_verified'].items():
            if not claim_result.get('verified', False):
                results['inconsistencies'].append(f"{claim_name}: {claim_result.get('issue', 'verification failed')}")
        
        return results
    
    def _verify_grundy_claim(self, positions_by_deck_size: Dict[int, List[CGTPosition]]) -> Dict[str, Any]:
        """Verify the specific Grundy number pattern claim"""
        # This is a simplified verification - would need actual position data
        return {
            'verified': False,
            'issue': 'Insufficient data - need actual Grundy calculations for verification',
            'method': 'grundy_pattern_analysis'
        }
    
    def _verify_temperature_claim(self, positions_by_deck_size: Dict[int, List[CGTPosition]]) -> Dict[str, Any]:
        """Verify the temperature maximization claim"""
        return {
            'verified': False,
            'issue': 'Requires complete temperature analysis across all deck sizes',
            'method': 'temperature_comparison_analysis'
        }
    
    def _verify_mean_preservation(self, positions_by_deck_size: Dict[int, List[CGTPosition]]) -> Dict[str, Any]:
        """Verify mean-preserving temperature oscillation"""
        return {
            'verified': False,
            'issue': 'Requires statistical analysis of mean values across positions',
            'method': 'mean_preservation_analysis'
        }
    
    def generate_verification_report(self, positions: List[CGTPosition]) -> str:
        """Generate a comprehensive verification report"""
        report_lines = [
            "MATHEMATICAL VERIFICATION REPORT",
            "=" * 50,
            ""
        ]
        
        total_positions = len(positions)
        passed = 0
        failed = 0
        
        for position in positions:
            result = self.verify_position_consistency(position)
            
            if result['verification_passed']:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"
            
            report_lines.append(f"{status} {position.position_name}")
            
            if result['inconsistencies_found']:
                for issue in result['inconsistencies_found']:
                    report_lines.append(f"  - Issue: {issue}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    report_lines.append(f"  - Warning: {warning}")
        
        report_lines.extend([
            "",
            "SUMMARY",
            "-" * 20,
            f"Total positions verified: {total_positions}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Success rate: {passed/total_positions*100:.1f}%" if total_positions > 0 else "No positions",
            ""
        ])
        
        if failed > 0:
            report_lines.extend([
                "RECOMMENDATIONS",
                "-" * 20,
                "1. Review failed position calculations for mathematical errors",
                "2. Ensure Grundy number calculations follow Sprague-Grundy theorem correctly",
                "3. Verify temperature calculations follow Conway's thermographic theory",
                "4. Check that game value computations use proper recursive definitions",
                ""
            ])
        
        return "\n".join(report_lines)