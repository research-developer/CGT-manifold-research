"""
Test suite for statistical analysis framework.

Tests hypothesis testing, power analysis, bootstrap confidence intervals,
and integration with Monte Carlo simulations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cgt_analysis.statistical_analysis import StatisticalAnalyzer, HypothesisTest
from cgt_analysis.base import CGTAnalyzer, DataManager
from cgt_analysis.war_engine import WarGameEngine


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = StatisticalAnalyzer(significance_level=0.05)
        
        # Create mock results for different deck sizes
        self.results_44 = {
            'num_simulations': 1000,
            'deck_size': 44,
            'game_name': 'War',
            'win_rate_p1': 0.48,
            'win_rate_p2': 0.52,
            'game_lengths': [100 + np.random.normal(0, 10) for _ in range(1000)],
            'avg_game_length': 100.0,
            'std_game_length': 10.0
        }
        
        self.results_48 = {
            'num_simulations': 1000,
            'deck_size': 48,
            'game_name': 'War',
            'win_rate_p1': 0.50,
            'win_rate_p2': 0.50,
            'game_lengths': [110 + np.random.normal(0, 12) for _ in range(1000)],
            'avg_game_length': 110.0,
            'std_game_length': 12.0
        }
        
        self.results_52 = {
            'num_simulations': 1000,
            'deck_size': 52,
            'game_name': 'War',
            'win_rate_p1': 0.49,
            'win_rate_p2': 0.51,
            'game_lengths': [120 + np.random.normal(0, 15) for _ in range(1000)],
            'avg_game_length': 120.0,
            'std_game_length': 15.0
        }
    
    def test_initialization(self):
        """Test StatisticalAnalyzer initialization"""
        analyzer = StatisticalAnalyzer()
        assert analyzer.alpha == 0.05
        assert isinstance(analyzer.data_manager, DataManager)
        
        analyzer_custom = StatisticalAnalyzer(significance_level=0.01)
        assert analyzer_custom.alpha == 0.01
    
    def test_comprehensive_hypothesis_tests(self):
        """Test comprehensive hypothesis testing suite"""
        tests = self.analyzer.run_comprehensive_hypothesis_tests(
            self.results_44, self.results_48, self.results_52
        )
        
        # Check all expected tests are present
        expected_tests = [
            'game_length_anova',
            'win_rate_balance',
            'temperature_variance',
            'periodicity_test',
            'grundy_consistency'
        ]
        
        for test_name in expected_tests:
            assert test_name in tests
            assert isinstance(tests[test_name], HypothesisTest)
            assert hasattr(tests[test_name], 'p_value')
            assert hasattr(tests[test_name], 'statistic')
            assert hasattr(tests[test_name], 'effect_size')
    
    def test_game_length_anova(self):
        """Test game length ANOVA specifically"""
        test_result = self.analyzer._test_game_length_differences(
            self.results_44, self.results_48, self.results_52
        )
        
        assert test_result.test_name == "Game Length ANOVA"
        assert isinstance(test_result.statistic, float)
        assert 0 <= test_result.p_value <= 1
        assert test_result.effect_size >= 0
        assert len(test_result.confidence_interval) == 2
        assert test_result.null_hypothesis is not None
        assert test_result.alternative_hypothesis is not None
    
    def test_win_rate_balance_test(self):
        """Test win rate balance test"""
        test_result = self.analyzer._test_win_rate_balance(
            self.results_44, self.results_48, self.results_52
        )
        
        assert test_result.test_name == "Win Rate Balance Test"
        assert isinstance(test_result.statistic, float)
        assert 0 <= test_result.p_value <= 1
        assert test_result.effect_size >= 0
        assert "48-card deviation" in test_result.interpretation
    
    def test_temperature_variance_test(self):
        """Test temperature variance analysis"""
        test_result = self.analyzer._test_temperature_variance(
            self.results_44, self.results_48, self.results_52
        )
        
        assert test_result.test_name == "Temperature Variance Test"
        assert isinstance(test_result.statistic, float)
        assert 0 <= test_result.p_value <= 1
        assert test_result.effect_size >= 0
    
    def test_periodicity_test(self):
        """Test periodicity evidence test"""
        test_result = self.analyzer._test_periodicity_evidence(
            self.results_44, self.results_48, self.results_52
        )
        
        assert test_result.test_name == "Periodicity Evidence Test"
        assert isinstance(test_result.statistic, float)
        assert 0 <= test_result.p_value <= 1
        assert "autocorr" in test_result.interpretation
    
    def test_grundy_consistency_test(self):
        """Test Grundy number consistency"""
        test_result = self.analyzer._test_grundy_consistency(
            self.results_44, self.results_48, self.results_52
        )
        
        assert test_result.test_name == "Grundy Consistency Test"
        assert isinstance(test_result.statistic, float)
        assert 0 <= test_result.p_value <= 1
        assert "Coefficient of variation" in test_result.interpretation
    
    def test_power_analysis(self):
        """Test power analysis calculations"""
        power_result = self.analyzer.calculate_power_analysis(
            effect_size=0.8,
            sample_size=1000,
            test_type='two_sample'
        )
        
        assert 'power' in power_result
        assert 'effect_size' in power_result
        assert 'sample_size' in power_result
        assert 'min_sample_for_80_power' in power_result
        assert 'interpretation' in power_result
        
        assert power_result['effect_size'] == 0.8
        assert power_result['sample_size'] == 1000
        assert 0 <= power_result['power'] <= 1
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation"""
        data = [100 + np.random.normal(0, 10) for _ in range(100)]
        
        ci_lower, ci_upper = self.analyzer.bootstrap_confidence_intervals(
            data, statistic_func=np.mean, n_bootstrap=1000
        )
        
        assert ci_lower < ci_upper
        assert ci_lower <= np.mean(data) <= ci_upper
        
        # Test with different statistics
        ci_std = self.analyzer.bootstrap_confidence_intervals(
            data, statistic_func=np.std, n_bootstrap=500
        )
        
        assert ci_std[0] < ci_std[1]
        assert ci_std[0] <= np.std(data) <= ci_std[1]
    
    def test_bootstrap_empty_data(self):
        """Test bootstrap with empty data"""
        ci_lower, ci_upper = self.analyzer.bootstrap_confidence_intervals([])
        assert ci_lower == 0.0
        assert ci_upper == 0.0
    
    def test_enhanced_monte_carlo_integration(self):
        """Test integration with enhanced Monte Carlo"""
        # Create mock engine and analyzer
        engine = Mock()
        engine.deck_size = 48
        engine.game_name = "War"
        engine.seed = 42
        
        # Mock simulate_game to return consistent results
        engine.simulate_game.return_value = {
            'winner': Mock(name='PLAYER1'),
            'num_moves': 100,
            'temperature_trajectory': [1.0, 0.8, 0.6]
        }
        
        cgt_analyzer = CGTAnalyzer(engine)
        
        # Run enhanced Monte Carlo
        results = self.analyzer.run_enhanced_monte_carlo(
            cgt_analyzer,
            num_simulations=100,
            use_antithetic_variates=True,
            track_convergence=True
        )
        
        assert results['num_simulations'] == 100
        assert results['use_antithetic_variates'] is True
        assert 'convergence_diagnostics' in results
        assert len(results['game_lengths']) == 100
    
    def test_statistical_report_generation(self):
        """Test statistical report generation"""
        # Create mock test results
        hypothesis_tests = {
            'test1': HypothesisTest(
                test_name="Test 1",
                statistic=2.5,
                p_value=0.02,
                effect_size=0.3,
                confidence_interval=(0.1, 0.5),
                null_hypothesis="No effect",
                alternative_hypothesis="There is an effect",
                interpretation="Significant result"
            )
        }
        
        power_analyses = {
            'analysis1': {
                'power': 0.85,
                'effect_size': 0.5,
                'sample_size': 1000,
                'min_sample_for_80_power': 800,
                'interpretation': 'Adequate power'
            }
        }
        
        bootstrap_results = {
            'metric1': (0.45, 0.55)
        }
        
        report = self.analyzer.generate_statistical_report(
            hypothesis_tests, power_analyses, bootstrap_results
        )
        
        assert "STATISTICAL ANALYSIS REPORT" in report
        assert "HYPOTHESIS TESTING RESULTS" in report
        assert "POWER ANALYSIS RESULTS" in report
        assert "BOOTSTRAP CONFIDENCE INTERVALS" in report
        assert "Test 1" in report
        assert "0.02" in report  # p-value
        assert "analysis1" in report
    
    def test_autocorrelation_calculation(self):
        """Test autocorrelation calculation"""
        # Create data with known autocorrelation
        data = [np.sin(2 * np.pi * i / 16) for i in range(100)]
        
        autocorr_16 = self.analyzer._calculate_autocorrelation(data, lag=16)
        autocorr_8 = self.analyzer._calculate_autocorrelation(data, lag=8)
        
        # For sinusoidal data with period 16, lag 16 should have high autocorrelation
        assert abs(autocorr_16) > abs(autocorr_8)
        
        # Test edge cases
        assert self.analyzer._calculate_autocorrelation([], lag=1) == 0.0
        assert self.analyzer._calculate_autocorrelation([1, 2], lag=5) == 0.0
    
    def test_effect_size_calculations(self):
        """Test effect size calculations"""
        # Test eta-squared calculation
        group1 = [10, 12, 14, 16, 18]
        group2 = [20, 22, 24, 26, 28]
        group3 = [30, 32, 34, 36, 38]
        
        eta_squared = self.analyzer._compute_eta_squared([group1, group2, group3])
        
        assert 0 <= eta_squared <= 1
        assert eta_squared > 0  # Should be positive for different means
        
        # Test Kruskal effect size
        kruskal_effect = self.analyzer._compute_kruskal_effect_size(group1, group2, group3)
        
        assert kruskal_effect >= 0
    
    def test_bootstrap_effect_size_ci(self):
        """Test bootstrap confidence intervals for effect sizes"""
        groups = [
            [10 + np.random.normal(0, 1) for _ in range(50)],
            [12 + np.random.normal(0, 1) for _ in range(50)],
            [14 + np.random.normal(0, 1) for _ in range(50)]
        ]
        
        ci_lower, ci_upper = self.analyzer._bootstrap_effect_size_ci(
            groups, self.analyzer._compute_eta_squared, n_bootstrap=100
        )
        
        assert ci_lower <= ci_upper
        assert ci_lower >= 0  # Eta-squared is non-negative
    
    def test_bootstrap_proportion_ci(self):
        """Test bootstrap confidence intervals for proportions"""
        # Test normal case
        ci_lower, ci_upper = self.analyzer._bootstrap_proportion_ci(50, 100)
        
        assert 0 <= ci_lower <= ci_upper <= 1
        assert ci_lower < 0.5 < ci_upper  # Should contain true proportion
        
        # Test edge cases
        ci_0 = self.analyzer._bootstrap_proportion_ci(0, 100)
        assert ci_0[0] == 0
        
        ci_all = self.analyzer._bootstrap_proportion_ci(100, 100)
        assert ci_all[1] == 1
        
        ci_empty = self.analyzer._bootstrap_proportion_ci(0, 0)
        assert ci_empty == (0.0, 0.0)
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction in hypothesis tests"""
        tests = self.analyzer.run_comprehensive_hypothesis_tests(
            self.results_44, self.results_48, self.results_52
        )
        
        # All test interpretations should mention Bonferroni correction
        for test_name, test_result in tests.items():
            assert "Bonferroni corrected" in test_result.interpretation
    
    def test_temperature_data_extraction(self):
        """Test temperature data extraction from results"""
        # Test with temperature trajectories
        results_with_temp = {
            'temperature_trajectories': [
                [1.0, 0.8, 0.6],
                [1.2, 1.0, 0.8],
                [0.9, 0.7, 0.5]
            ]
        }
        
        temp_data = self.analyzer._extract_temperature_data(results_with_temp)
        expected_length = 3 * 3  # 3 trajectories × 3 points each
        assert len(temp_data) == expected_length
        
        # Test without temperature data
        results_no_temp = {'game_lengths': [100, 110, 120]}
        temp_data_empty = self.analyzer._extract_temperature_data(results_no_temp)
        assert temp_data_empty == []


class TestIntegrationWithCGTFramework:
    """Integration tests with the broader CGT framework"""
    
    def test_integration_with_war_engine(self):
        """Test integration with actual War game engine"""
        # This test might take longer, so we use fewer simulations
        engine = WarGameEngine(deck_size=48, seed=42)
        analyzer = CGTAnalyzer(engine)
        statistical_analyzer = StatisticalAnalyzer()
        
        # Run enhanced Monte Carlo
        results = statistical_analyzer.run_enhanced_monte_carlo(
            analyzer,
            num_simulations=50,  # Small number for testing
            use_antithetic_variates=True,
            track_convergence=True
        )
        
        # Verify structure
        assert results['deck_size'] == 48
        assert results['game_name'] == 'War'
        assert len(results['game_lengths']) == 50
        assert 'convergence_diagnostics' in results
        assert results['use_antithetic_variates'] is True
    
    def test_data_manager_integration(self):
        """Test integration with DataManager"""
        statistical_analyzer = StatisticalAnalyzer()
        
        # Create mock results
        test_results = {
            'test_statistic': 2.5,
            'p_value': 0.02,
            'effect_size': 0.3
        }
        
        # Save results
        filepath = statistical_analyzer.data_manager.save_analysis_result(
            game_name="War",
            deck_size=48,
            analysis_type="statistical_test",
            data=test_results
        )
        
        assert filepath is not None
        assert "War_48_statistical_test" in filepath
        
        # Load results
        loaded_results = statistical_analyzer.data_manager.load_analysis_results(
            game_name="War",
            deck_size=48,
            analysis_type="statistical_test"
        )
        
        assert len(loaded_results) > 0
        assert loaded_results[-1]['test_statistic'] == 2.5


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        self.analyzer = StatisticalAnalyzer()
    
    def test_empty_results(self):
        """Test handling of empty results"""
        empty_results = {
            'num_simulations': 0,
            'game_lengths': [],
            'win_rate_p1': 0.0
        }
        
        # Should not crash
        tests = self.analyzer.run_comprehensive_hypothesis_tests(
            empty_results, empty_results, empty_results
        )
        
        assert len(tests) == 5  # All tests should still be present
    
    def test_invalid_power_analysis_inputs(self):
        """Test power analysis with invalid inputs"""
        # Very small effect size
        power_result = self.analyzer.calculate_power_analysis(
            effect_size=0.01,
            sample_size=10
        )
        
        assert power_result['power'] < 0.5  # Should have low power
        
        # Very large sample size
        power_result_large = self.analyzer.calculate_power_analysis(
            effect_size=0.8,
            sample_size=10000
        )
        
        assert power_result_large['power'] > 0.99  # Should have very high power
    
    def test_single_value_data(self):
        """Test with data that has no variance"""
        constant_data = [100] * 1000
        
        ci_lower, ci_upper = self.analyzer.bootstrap_confidence_intervals(constant_data)
        
        # Should handle constant data gracefully
        assert ci_lower == ci_upper == 100


if __name__ == "__main__":
    pytest.main([__file__])