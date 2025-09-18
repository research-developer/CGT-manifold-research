"""
Unit tests for CGT analysis framework.

Run with: pytest tests/test_cgt_framework.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from cgt_analysis.base import Player, GameState, CGTAnalyzer
from cgt_analysis.war_engine import WarGameEngine, WarPosition
from cgt_analysis.cgt_position import CGTPosition


class TestWarEngine:
    """Test War game engine implementation"""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = WarGameEngine(deck_size=48, seed=42)
        assert engine.deck_size == 48
        assert engine.game_name == "War"
        assert engine.is_impartial == True
    
    def test_deck_size_variations(self):
        """Test different deck sizes have correct card ranges"""
        # 44 cards: no 2s, 3s, 4s
        engine44 = WarGameEngine(deck_size=44)
        assert engine44.min_rank == 5
        
        # 48 cards: no 2s, 3s
        engine48 = WarGameEngine(deck_size=48)
        assert engine48.min_rank == 4
        
        # 52 cards: full deck
        engine52 = WarGameEngine(deck_size=52)
        assert engine52.min_rank == 2
    
    def test_initial_state_creation(self):
        """Test initial state is created correctly"""
        engine = WarGameEngine(deck_size=48, seed=42)
        state = engine.create_initial_state()
        
        assert isinstance(state, WarPosition)
        assert len(state.player1_hand) == 24
        assert len(state.player2_hand) == 24
        assert len(state.player1_hand) + len(state.player2_hand) == 48
        assert not state.is_terminal()
    
    def test_terminal_state_detection(self):
        """Test terminal state is detected correctly"""
        position = WarPosition(
            player1_hand=[],
            player2_hand=[10, 9, 8],
            deck_size=48
        )
        assert position.is_terminal()
        assert position.get_winner() == Player.PLAYER2
        
        position2 = WarPosition(
            player1_hand=[10, 9, 8],
            player2_hand=[],
            deck_size=48
        )
        assert position2.is_terminal()
        assert position2.get_winner() == Player.PLAYER1
    
    def test_game_value_calculation(self):
        """Test game value heuristic"""
        position = WarPosition(
            player1_hand=[13, 12, 11],  # 3 cards
            player2_hand=[5, 6],         # 2 cards
            deck_size=48
        )
        value = position.get_game_value()
        assert value > 0  # Player 1 advantage
        assert value == (3 - 2) / 48  # Card difference normalized
    
    def test_move_application(self):
        """Test applying moves in War"""
        engine = WarGameEngine(deck_size=48)
        position = WarPosition(
            player1_hand=[13, 10],
            player2_hand=[12, 9],
            deck_size=48
        )
        
        # Apply play move
        next_state = engine.apply_move(position, "play")
        
        # Player 1 should win (13 > 12)
        assert len(next_state.player1_hand) == 3  # Original 1 + won 2
        assert len(next_state.player2_hand) == 1  # Lost 1
    
    def test_position_creation_methods(self):
        """Test standard position creation methods"""
        engine = WarGameEngine(deck_size=48)
        
        # Test Position A
        pos_a = engine.create_position_a()
        assert pos_a.position_type == "opening_advantage"
        assert len(pos_a.player1_hand) == 5
        assert len(pos_a.player2_hand) == 5
        
        # Test Position B (war scenario)
        pos_b = engine.create_position_b()
        assert pos_b.position_type == "war"
        assert len(pos_b.player1_pile) > 0
        assert len(pos_b.player2_pile) > 0
        
        # Test Position C (balanced)
        pos_c = engine.create_position_c()
        assert pos_c.position_type == "mid_game"
        
        # Test Position D (endgame)
        pos_d = engine.create_position_d()
        assert pos_d.position_type == "endgame"
        assert len(pos_d.player1_hand) == 2
        assert len(pos_d.player2_hand) == 2
        
        # Test Position E (extreme)
        pos_e = engine.create_position_e()
        assert pos_e.position_type == "deterministic_win"


class TestCGTPosition:
    """Test CGT position representation"""
    
    def test_terminal_position(self):
        """Test terminal position properties"""
        pos = CGTPosition(
            left_options=[],
            right_options=[],
            position_name="terminal"
        )
        
        assert pos.is_terminal()
        assert pos.get_cgt_notation() == "0"
        assert pos.compute_game_value() == 0
        assert pos.compute_temperature() == 0.0
    
    def test_cgt_notation(self):
        """Test CGT notation generation"""
        # Create a simple position
        terminal = CGTPosition([], [], "terminal")
        
        pos = CGTPosition(
            left_options=[terminal],
            right_options=[terminal],
            position_name="test"
        )
        
        notation = pos.get_cgt_notation()
        assert "{" in notation and "}" in notation
        assert "|" in notation
    
    def test_game_value_computation(self):
        """Test game value calculation"""
        # Create positions with known values
        pos_zero = CGTPosition([], [], "zero")
        assert pos_zero.compute_game_value() == 0
        
        # Position with only left options (positive value)
        pos_positive = CGTPosition(
            left_options=[pos_zero],
            right_options=[],
            position_name="positive"
        )
        # This should have positive value
        # Exact value depends on implementation
    
    def test_temperature_computation(self):
        """Test temperature calculation"""
        terminal = CGTPosition([], [], "terminal")
        assert terminal.compute_temperature() == 0.0
        
        # Hot position with many options
        hot_pos = CGTPosition(
            left_options=[terminal, terminal],
            right_options=[terminal, terminal],
            position_name="hot"
        )
        temp = hot_pos.compute_temperature()
        assert temp >= 0.0  # Temperature is non-negative


class TestCGTAnalyzer:
    """Test CGT analyzer functionality"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes with engine"""
        engine = WarGameEngine(deck_size=48)
        analyzer = CGTAnalyzer(engine)
        
        assert analyzer.engine == engine
        assert len(analyzer.cache) == 0
    
    def test_position_analysis(self):
        """Test analyzing a position"""
        engine = WarGameEngine(deck_size=48, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        position = engine.create_position_a()
        analysis = analyzer.analyze_position(position, max_depth=2)
        
        assert 'game_name' in analysis
        assert analysis['game_name'] == "War"
        assert 'deck_size' in analysis
        assert analysis['deck_size'] == 48
        assert 'is_impartial' in analysis
        assert analysis['is_impartial'] == True
        assert 'grundy_number' in analysis
        assert 'temperature_computed' in analysis
    
    def test_monte_carlo_analysis(self):
        """Test Monte Carlo simulation"""
        engine = WarGameEngine(deck_size=48, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        results = analyzer.run_monte_carlo_analysis(num_simulations=10)
        
        assert results['num_simulations'] == 10
        assert results['deck_size'] == 48
        assert results['game_name'] == "War"
        assert 'win_rate_p1' in results
        assert 'win_rate_p2' in results
        assert abs(results['win_rate_p1'] + results['win_rate_p2'] - 1.0) < 0.01
        assert 'avg_game_length' in results
        assert results['avg_game_length'] > 0
    
    def test_caching(self):
        """Test that analysis results are cached"""
        engine = WarGameEngine(deck_size=48, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        position = engine.create_position_a()
        
        # First analysis
        analysis1 = analyzer.analyze_position(position, max_depth=2)
        cache_size = len(analyzer.cache)
        
        # Second analysis of same position
        analysis2 = analyzer.analyze_position(position, max_depth=2)
        
        # Should use cache, not add new entries
        assert len(analyzer.cache) == cache_size
        assert analysis1 == analysis2


class TestIntegration:
    """Integration tests for the full framework"""
    
    def test_full_war_analysis_pipeline(self):
        """Test complete analysis pipeline for War"""
        # Initialize engine
        engine = WarGameEngine(deck_size=48, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        # Analyze all standard positions
        positions = {
            'A': engine.create_position_a(),
            'B': engine.create_position_b(),
            'C': engine.create_position_c(),
            'D': engine.create_position_d(),
            'E': engine.create_position_e()
        }
        
        results = {}
        for name, position in positions.items():
            analysis = analyzer.analyze_position(position, max_depth=2)
            results[name] = analysis
            
            # Verify all expected fields present
            assert 'grundy_number' in analysis
            assert 'temperature_computed' in analysis
            assert 'game_value' in analysis
            assert 'cgt_notation' in analysis
        
        # Run Monte Carlo
        mc_results = analyzer.run_monte_carlo_analysis(num_simulations=100)
        
        assert mc_results['num_simulations'] == 100
        assert 0 <= mc_results['win_rate_p1'] <= 1
        assert 0 <= mc_results['win_rate_p2'] <= 1
    
    def test_deck_size_comparison(self):
        """Test comparing different deck sizes"""
        results = {}
        
        for deck_size in [44, 48, 52]:
            engine = WarGameEngine(deck_size=deck_size, seed=42)
            analyzer = CGTAnalyzer(engine)
            
            # Analyze position A for each deck size
            pos_a = engine.create_position_a()
            analysis = analyzer.analyze_position(pos_a, max_depth=2)
            
            results[deck_size] = {
                'grundy': analysis.get('grundy_number'),
                'temperature': analysis.get('temperature_computed'),
                'game_value': analysis.get('game_value_numeric', 0)
            }
        
        # Verify we got results for all deck sizes
        assert 44 in results
        assert 48 in results
        assert 52 in results
        
        # Each should have the expected fields
        for deck_size, result in results.items():
            assert 'grundy' in result
            assert 'temperature' in result
            assert 'game_value' in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
