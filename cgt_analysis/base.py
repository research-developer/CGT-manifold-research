"""
Abstract base classes for game engines and CGT analysis.

This module defines the interfaces that all game implementations must follow,
ensuring consistency across War, Crazy Eights, and future games.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Player(Enum):
    """Standard player enumeration for two-player games"""
    PLAYER1 = 1  # Left player in CGT notation
    PLAYER2 = 2  # Right player in CGT notation


@dataclass
class GameState(ABC):
    """
    Abstract base class for game states.
    
    All game-specific states (WarPosition, CrazyEightsState, etc.) 
    should inherit from this class.
    """
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (game over)"""
        pass
    
    @abstractmethod
    def get_winner(self) -> Optional[Player]:
        """Get the winner if the game is terminal, None otherwise"""
        pass
    
    @abstractmethod
    def get_player_to_move(self) -> Player:
        """Get which player moves next from this state"""
        pass
    
    @abstractmethod
    def get_state_hash(self) -> str:
        """Get a unique hash for this state (for caching)"""
        pass
    
    @abstractmethod
    def get_game_value(self) -> float:
        """
        Get a heuristic value for this state.
        Positive = advantage for Player1, Negative = advantage for Player2
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        pass


class GameEngine(ABC):
    """
    Abstract base class for game engines.
    
    Defines the interface that all game implementations must provide
    for CGT analysis and simulation.
    """
    
    def __init__(self, deck_size: int = 48, seed: Optional[int] = None):
        """
        Initialize game engine with deck size and random seed.
        
        Args:
            deck_size: Number of cards in deck (44, 48, or 52)
            seed: Random seed for reproducibility
        """
        self.deck_size = deck_size
        self.seed = seed
    
    @abstractmethod
    def create_initial_state(self) -> GameState:
        """Create the initial game state for a new game"""
        pass
    
    @abstractmethod
    def get_possible_moves(self, state: GameState) -> List[Any]:
        """
        Get all possible moves from the current state.
        
        Returns:
            List of move objects (game-specific type)
        """
        pass
    
    @abstractmethod
    def apply_move(self, state: GameState, move: Any) -> GameState:
        """
        Apply a move to get the next state.
        
        Args:
            state: Current game state
            move: Move to apply (game-specific type)
            
        Returns:
            New game state after the move
        """
        pass
    
    @abstractmethod
    def get_next_states(self, state: GameState) -> List[Tuple[Any, GameState]]:
        """
        Get all possible next states from current state.
        
        Returns:
            List of (move, resulting_state) tuples
        """
        pass
    
    @abstractmethod
    def simulate_game(self, initial_state: Optional[GameState] = None) -> Dict[str, Any]:
        """
        Simulate a complete game from initial state.
        
        Args:
            initial_state: Starting state (or create new if None)
            
        Returns:
            Dictionary with game results including:
            - winner: Player who won
            - num_moves: Total moves played
            - final_state: Terminal state
            - trajectory: Optional list of states
        """
        pass
    
    @abstractmethod
    def evaluate_position(self, state: GameState) -> Dict[str, float]:
        """
        Evaluate a position for CGT analysis.
        
        Returns:
            Dictionary with:
            - game_value: Numerical evaluation
            - confidence: Confidence in evaluation (0-1)
            - temperature: Position temperature
        """
        pass
    
    @property
    @abstractmethod
    def game_name(self) -> str:
        """Get the name of this game"""
        pass
    
    @property  
    @abstractmethod
    def is_impartial(self) -> bool:
        """
        Check if this is an impartial game.
        
        Impartial games have the same moves available to both players.
        War is impartial, Crazy Eights is partisan.
        """
        pass


class CGTAnalyzer:
    """
    Universal CGT analyzer that works with any GameEngine implementation.
    
    This class provides the bridge between game-specific engines and
    the formal CGT analysis framework.
    """
    
    def __init__(self, game_engine: GameEngine):
        """
        Initialize analyzer with a game engine.
        
        Args:
            game_engine: Any GameEngine implementation
        """
        self.engine = game_engine
        self.cache = {}
    
    def analyze_position(self, state: GameState, max_depth: int = 3) -> Dict[str, Any]:
        """
        Perform complete CGT analysis on a game state.
        
        Args:
            state: Game state to analyze
            max_depth: Maximum depth for game tree expansion
            
        Returns:
            Dictionary with CGT analysis results
        """
        # Check cache
        state_hash = state.get_state_hash()
        if state_hash in self.cache:
            return self.cache[state_hash]
        
        # Import CGT framework
        from cgt_analysis.cgt_position import CGTPosition, GameTree
        from cgt_analysis.grundy_numbers import GrundyCalculator
        from cgt_analysis.temperature_analysis import TemperatureCalculator
        
        # Convert to CGT position
        cgt_pos = self._convert_to_cgt_position(state, 0, max_depth)
        
        # Create game tree
        tree = GameTree(cgt_pos, max_depth)
        tree.generate_full_tree()
        
        # Calculate Grundy number
        grundy_calc = GrundyCalculator()
        grundy = grundy_calc.compute_grundy_number(cgt_pos)
        
        # Calculate temperature
        temp_calc = TemperatureCalculator()
        temperature = temp_calc.compute_temperature(cgt_pos)
        
        # Get complete analysis
        analysis = cgt_pos.get_analysis_summary()
        analysis['grundy_number'] = grundy
        analysis['temperature_computed'] = temperature
        analysis['game_name'] = self.engine.game_name
        analysis['deck_size'] = self.engine.deck_size
        analysis['is_impartial'] = self.engine.is_impartial
        
        # Cache result
        self.cache[state_hash] = analysis
        
        return analysis
    
    def _convert_to_cgt_position(self, state: GameState, depth: int, max_depth: int) -> 'CGTPosition':
        """Convert a game state to CGT position format"""
        from cgt_analysis.cgt_position import CGTPosition
        
        if state.is_terminal() or depth >= max_depth:
            cgt_pos = CGTPosition(
                left_options=[],
                right_options=[],
                position_name=f"{self.engine.game_name}_{state.get_state_hash()[:8]}_d{depth}",
                war_position=state if hasattr(state, 'player1_hand') else None
            )
            return cgt_pos
        
        # Get next states
        next_states = self.engine.get_next_states(state)
        
        # Separate into left and right options based on who benefits
        left_options = []
        right_options = []
        
        current_value = state.get_game_value()
        
        for move, next_state in next_states:
            next_cgt = self._convert_to_cgt_position(next_state, depth + 1, max_depth)
            next_value = next_state.get_game_value()
            
            # Assign based on value change
            if next_value > current_value:
                left_options.append(next_cgt)  # Good for Player1
            elif next_value < current_value:
                right_options.append(next_cgt)  # Good for Player2
            else:
                # Neutral, assign based on whose turn it is
                if state.get_player_to_move() == Player.PLAYER1:
                    left_options.append(next_cgt)
                else:
                    right_options.append(next_cgt)
        
        return CGTPosition(
            left_options=left_options,
            right_options=right_options,
            position_name=f"{self.engine.game_name}_{state.get_state_hash()[:8]}_d{depth}",
            war_position=state if hasattr(state, 'player1_hand') else None
        )
    
    def run_monte_carlo_analysis(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for statistical analysis.
        
        Args:
            num_simulations: Number of games to simulate
            
        Returns:
            Dictionary with statistical results
        """
        results = {
            'num_simulations': num_simulations,
            'deck_size': self.engine.deck_size,
            'game_name': self.engine.game_name,
            'wins': {Player.PLAYER1: 0, Player.PLAYER2: 0},
            'game_lengths': [],
            'temperature_trajectories': []
        }
        
        for i in range(num_simulations):
            game_result = self.engine.simulate_game()
            
            # Record winner
            if game_result['winner']:
                results['wins'][game_result['winner']] += 1
            
            # Record game length
            results['game_lengths'].append(game_result['num_moves'])
            
            # Record temperature if available
            if 'temperature_trajectory' in game_result:
                results['temperature_trajectories'].append(game_result['temperature_trajectory'])
        
        # Calculate statistics
        import numpy as np
        results['win_rate_p1'] = results['wins'][Player.PLAYER1] / num_simulations
        results['win_rate_p2'] = results['wins'][Player.PLAYER2] / num_simulations
        results['avg_game_length'] = np.mean(results['game_lengths'])
        results['std_game_length'] = np.std(results['game_lengths'])
        
        return results


class DataManager:
    """
    Centralized data management for all CGT analysis results.
    
    Handles saving, loading, and organizing results across different
    games and deck sizes for easy comparison and analysis.
    """
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize data manager.
        
        Args:
            base_path: Base directory for all data storage
        """
        self.base_path = base_path
        self._ensure_directory_structure()
    
    def _ensure_directory_structure(self):
        """Create necessary directory structure"""
        import os
        
        # Create main directories
        directories = [
            self.base_path,
            f"{self.base_path}/raw",
            f"{self.base_path}/processed", 
            f"{self.base_path}/visualizations",
            f"{self.base_path}/reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_analysis_result(self, game_name: str, deck_size: int, 
                            analysis_type: str, data: Dict[str, Any]) -> str:
        """
        Save analysis result with standardized naming.
        
        Args:
            game_name: Name of the game analyzed
            deck_size: Deck size used
            analysis_type: Type of analysis (cgt, monte_carlo, etc.)
            data: Analysis results
            
        Returns:
            Path to saved file
        """
        import json
        from datetime import datetime
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_name}_{deck_size}_{analysis_type}_{timestamp}.json"
        filepath = f"{self.base_path}/raw/{filename}"
        
        # Add metadata (without mutating caller's original reference deeply)
        data = dict(data)  # shallow copy
        data['metadata'] = {
            'game_name': game_name,
            'deck_size': deck_size,
            'analysis_type': analysis_type,
            'timestamp': timestamp,
            'version': '1.0.0'
        }

        # Sanitize data to ensure JSON serializable (convert Enum keys/values, numpy scalars, sets, etc.)
        sanitized = self._sanitize_for_json(data)
        
        # Save data
        with open(filepath, 'w') as f:
            json.dump(sanitized, f, indent=2)
        
        return filepath
    
    def load_analysis_results(self, game_name: Optional[str] = None,
                             deck_size: Optional[int] = None,
                             analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load analysis results matching criteria.
        
        Args:
            game_name: Filter by game name
            deck_size: Filter by deck size
            analysis_type: Filter by analysis type
            
        Returns:
            List of matching analysis results
        """
        import json
        import os
        
        results = []
        
        # Scan raw directory
        for filename in os.listdir(f"{self.base_path}/raw"):
            if not filename.endswith('.json'):
                continue
            
            # Check filters
            if game_name and not filename.startswith(game_name):
                continue
            if deck_size and f"_{deck_size}_" not in filename:
                continue
            if analysis_type and f"_{analysis_type}_" not in filename:
                continue
            
            # Load file
            filepath = f"{self.base_path}/raw/{filename}"
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
        
        return results
    
    def create_comparison_report(self, games: List[str], deck_sizes: List[int]) -> str:
        """
        Create a comparison report across games and deck sizes.
        
        Args:
            games: List of game names to compare
            deck_sizes: List of deck sizes to compare
            
        Returns:
            Path to generated report
        """
        import pandas as pd
        from datetime import datetime
        
        # Collect all relevant data
        comparison_data = []
        
        for game in games:
            for deck_size in deck_sizes:
                results = self.load_analysis_results(game, deck_size)
                
                if results:
                    # Take most recent result
                    latest = max(results, key=lambda x: x.get('metadata', {}).get('timestamp', ''))
                    
                    comparison_data.append({
                        'game': game,
                        'deck_size': deck_size,
                        'win_rate_p1': latest.get('win_rate_p1', None),
                        'avg_game_length': latest.get('avg_game_length', None),
                        'grundy_number': latest.get('grundy_number', None),
                        'temperature': latest.get('temperature', None)
                    })
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.base_path}/reports/comparison_{timestamp}.csv"
        df.to_csv(report_path, index=False)
        
        return report_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert an object into a JSON-serializable form.

        Handles:
        - Enum keys & values (stored as their .name)
        - numpy scalar types (converted to native Python)
        - numpy arrays (converted to lists)
        - sets and tuples (converted to lists)
        - objects providing to_dict()
        - Fallback: str() for unknown non-serializable keys
        """
        from enum import Enum as _Enum
        try:
            import numpy as _np  # type: ignore
        except ImportError:  # pragma: no cover
            _np = None  # type: ignore

        # Enum instance itself
        if isinstance(obj, _Enum):
            return obj.name

        # Basic types already serializable
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # numpy scalar
        if _np is not None:
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.ndarray,)):
                return [self._sanitize_for_json(x) for x in obj.tolist()]

        # Dict: sanitize keys & values
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # Sanitize key
                if isinstance(k, _Enum):
                    key = k.name
                elif isinstance(k, (str, int, float, bool)) or k is None:
                    key = k
                else:
                    key = str(k)
                new_dict[key] = self._sanitize_for_json(v)
            return new_dict

        # Iterable containers
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(x) for x in obj]

        # Dataclass / custom object with to_dict
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            try:
                return self._sanitize_for_json(obj.to_dict())
            except Exception:  # pragma: no cover
                return str(obj)

        # Fallback to string representation
        return str(obj)
