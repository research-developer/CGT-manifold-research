"""
Combinatorial Game Theory Position Representation

This module implements formal CGT position classes with {L | R} notation,
game tree generation, and all necessary structures for rigorous CGT analysis.
"""

from typing import List, Dict, Set, Optional, Tuple, Union
from dataclasses import dataclass
from fractions import Fraction
import copy
import math
from enum import Enum

class GameOutcome(Enum):
    """Standard CGT game outcomes"""
    LEFT_WIN = "L"      # Left (Player 1) wins
    RIGHT_WIN = "R"     # Right (Player 2) wins  
    FIRST_PLAYER_WIN = "*"   # First player to move wins
    SECOND_PLAYER_WIN = "0"  # Second player to move wins (previous player wins)

@dataclass
class CGTPosition:
    """
    Formal CGT position with {L | R} notation.
    
    This class represents a combinatorial game position in the standard
    Conway notation, with left and right option sets and computed values.
    """
    left_options: List['CGTPosition']   # Moves available to Left player
    right_options: List['CGTPosition']  # Moves available to Right player
    position_name: str                  # Human-readable identifier
    war_position: Optional[object] = None  # Reference to underlying WarPosition
    
    # Computed values (calculated lazily)
    _game_value: Optional[Union[Fraction, float]] = None
    _grundy_number: Optional[int] = None
    _temperature: Optional[float] = None
    _mean_value: Optional[float] = None
    _outcome_class: Optional[GameOutcome] = None
    
    def __post_init__(self):
        """Initialize computed values as None"""
        if self._game_value is None:
            self._game_value = None
        if self._grundy_number is None:
            self._grundy_number = None
        if self._temperature is None:
            self._temperature = None
        if self._mean_value is None:
            self._mean_value = None
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal position (no moves available)"""
        return len(self.left_options) == 0 and len(self.right_options) == 0
    
    def get_cgt_notation(self) -> str:
        """
        Return the standard CGT {L | R} notation for this position.
        
        Examples:
            {1, 0 | -1} for a position with left options 1,0 and right option -1
            {2* | 1} for a position with confused left option and right option 1
        """
        if self.is_terminal():
            return "0"  # Terminal position with no moves
        
        # Get notation for left options
        left_strs = []
        for left_opt in self.left_options:
            if left_opt.is_terminal():
                left_strs.append("0")
            else:
                # For non-terminal, use computed value if available
                if left_opt._game_value is not None:
                    left_strs.append(str(left_opt._game_value))
                else:
                    left_strs.append("?")  # Value not yet computed
        
        # Get notation for right options  
        right_strs = []
        for right_opt in self.right_options:
            if right_opt.is_terminal():
                right_strs.append("0")
            else:
                if right_opt._game_value is not None:
                    right_strs.append(str(right_opt._game_value))
                else:
                    right_strs.append("?")
        
        left_part = ", ".join(left_strs) if left_strs else ""
        right_part = ", ".join(right_strs) if right_strs else ""
        
        return f"{{{left_part} | {right_part}}}"
    
    def compute_game_value(self) -> Union[Fraction, float]:
        """
        Compute the game value using the standard CGT recursive definition.
        
        The value of {L | R} is the simplest number in the interval
        (max(L), min(R)) if this interval is non-empty, or the simplest
        surreal number otherwise.
        """
        if self._game_value is not None:
            return self._game_value
        
        if self.is_terminal():
            self._game_value = 0
            return self._game_value
        
        # Compute values of all options
        left_values = []
        for left_opt in self.left_options:
            left_values.append(left_opt.compute_game_value())
        
        right_values = []
        for right_opt in self.right_options:
            right_values.append(right_opt.compute_game_value())
        
        # Find the simplest number between max(left) and min(right)
        max_left = max(left_values) if left_values else float('-inf')
        min_right = min(right_values) if right_values else float('inf')
        
        if max_left >= min_right:
            # No number exists between max_left and min_right
            # This is a confused position, use special handling
            self._game_value = float('nan')  # Represents confusion
        else:
            # Find simplest number in interval (max_left, min_right)
            self._game_value = self._find_simplest_number(max_left, min_right)
        
        return self._game_value
    
    def _find_simplest_number(self, lower: float, upper: float) -> Fraction:
        """
        Find the simplest number (in Conway's sense) in the interval (lower, upper).
        
        The simplest numbers are integers, then halves, then quarters, etc.
        """
        # Handle infinity cases
        if not (math.isfinite(lower) and math.isfinite(upper)):
            if math.isinf(lower) and math.isinf(upper):
                return Fraction(0)  # Default for double infinity
            elif math.isinf(upper):
                return Fraction(int(lower) + 1) if math.isfinite(lower) else Fraction(1)
            elif math.isinf(lower):
                return Fraction(int(upper) - 1) if math.isfinite(upper) else Fraction(-1)
        
        # Check if 0 is in the interval
        if lower < 0 < upper:
            return Fraction(0)
        
        # Check integers
        for i in range(-10, 11):  # Reasonable range for card games
            if lower < i < upper:
                return Fraction(i)
        
        # Check halves
        for i in range(-20, 21):
            half = Fraction(i, 2)
            if lower < half < upper:
                return half
        
        # Check quarters
        for i in range(-40, 41):
            quarter = Fraction(i, 4)
            if lower < quarter < upper:
                return quarter
        
        # Fallback: use midpoint (with safety checks)
        if math.isfinite(lower) and math.isfinite(upper):
            return Fraction(lower + upper) / 2
        else:
            return Fraction(0)  # Safe fallback
    
    def compute_outcome_class(self) -> GameOutcome:
        """
        Determine the outcome class of this position.
        
        Returns:
            GameOutcome indicating who wins with optimal play
        """
        if self._outcome_class is not None:
            return self._outcome_class
        
        value = self.compute_game_value()
        
        if isinstance(value, float) and value != value:  # NaN check for confusion
            # Confused position - first player wins
            self._outcome_class = GameOutcome.FIRST_PLAYER_WIN
        elif value > 0:
            self._outcome_class = GameOutcome.LEFT_WIN
        elif value < 0:
            self._outcome_class = GameOutcome.RIGHT_WIN
        else:  # value == 0
            self._outcome_class = GameOutcome.SECOND_PLAYER_WIN
        
        return self._outcome_class
    
    def compute_temperature(self) -> float:
        """
        Compute the temperature of this position.
        
        Temperature measures how much the value can change based on
        who moves first. High temperature = hot game with many options.
        """
        if self._temperature is not None:
            return self._temperature
        
        if self.is_terminal():
            self._temperature = 0.0
            return self._temperature
        
        # Temperature is related to the difference between left and right values
        left_values = [opt.compute_game_value() for opt in self.left_options]
        right_values = [opt.compute_game_value() for opt in self.right_options]
        
        if not left_values or not right_values:
            self._temperature = 0.0
            return self._temperature
        
        # Simplified temperature calculation
        max_left = max(left_values) if left_values else 0
        min_right = min(right_values) if right_values else 0
        
        # Temperature is roughly half the gap between best moves
        self._temperature = abs(float(max_left) - float(min_right)) / 2.0
        
        return self._temperature
    
    def compute_mean_value(self) -> float:
        """
        Compute the mean value (average of left and right values).
        
        This is used in thermographic analysis to separate temperature
        effects from positional advantage.
        """
        if self._mean_value is not None:
            return self._mean_value
        
        if self.is_terminal():
            self._mean_value = 0.0
            return self._mean_value
        
        left_values = [float(opt.compute_game_value()) for opt in self.left_options]
        right_values = [float(opt.compute_game_value()) for opt in self.right_options]
        
        if not left_values and not right_values:
            self._mean_value = 0.0
        elif not left_values:
            self._mean_value = sum(right_values) / len(right_values)
        elif not right_values:
            self._mean_value = sum(left_values) / len(left_values)
        else:
            all_values = left_values + right_values
            self._mean_value = sum(all_values) / len(all_values)
        
        return self._mean_value
    
    def get_game_value(self) -> float:
        """
        Get a simple game value, using war_position if available.
        This is a fallback for positions that can't be fully computed.
        """
        if self.war_position:
            return self.war_position.get_game_value()
        else:
            try:
                value = self.compute_game_value()
                if isinstance(value, float) and value != value:  # NaN check
                    return 0.0
                return float(value)
            except:
                return 0.0
    
    def get_analysis_summary(self) -> Dict[str, Union[str, float, int]]:
        """
        Get a complete analysis summary of this position.
        
        Returns:
            Dictionary with all computed CGT values and analysis
        """
        try:
            game_value = self.compute_game_value()
            if isinstance(game_value, float) and game_value != game_value:  # NaN check
                game_value_str = "confused"
                game_value_float = 0.0
            else:
                game_value_str = str(float(game_value))
                game_value_float = float(game_value)
        except:
            game_value_str = "unknown"
            game_value_float = self.get_game_value()
        
        return {
            'position_name': self.position_name,
            'cgt_notation': self.get_cgt_notation(),
            'game_value': game_value_str,
            'game_value_numeric': game_value_float,
            'outcome_class': self.compute_outcome_class().value,
            'temperature': self.compute_temperature(),
            'mean_value': self.compute_mean_value(),
            'is_terminal': self.is_terminal(),
            'num_left_options': len(self.left_options),
            'num_right_options': len(self.right_options)
        }

class GameTree:
    """
    Represents a complete game tree for CGT analysis.
    
    This class builds and manages the full game tree needed for
    formal CGT calculations, including position generation,
    tree traversal, and analysis caching.
    """
    
    def __init__(self, root_position: CGTPosition, max_depth: int = 10):
        """
        Initialize game tree with root position.
        
        Args:
            root_position: Starting position for tree generation
            max_depth: Maximum depth to generate (prevents infinite trees)
        """
        self.root = root_position
        self.max_depth = max_depth
        self.position_cache: Dict[str, CGTPosition] = {}
        self.analysis_cache: Dict[str, Dict] = {}
        
    def generate_full_tree(self) -> None:
        """
        Generate the complete game tree up to max_depth.
        
        This performs a breadth-first expansion of all positions,
        caching results to avoid recomputation.
        """
        self._generate_tree_recursive(self.root, 0)
    
    def _generate_tree_recursive(self, position: CGTPosition, depth: int) -> None:
        """Recursively generate game tree"""
        if depth >= self.max_depth or position.is_terminal():
            return
        
        # Generate left options
        for left_opt in position.left_options:
            if left_opt.position_name not in self.position_cache:
                self.position_cache[left_opt.position_name] = left_opt
                self._generate_tree_recursive(left_opt, depth + 1)
        
        # Generate right options
        for right_opt in position.right_options:
            if right_opt.position_name not in self.position_cache:
                self.position_cache[right_opt.position_name] = right_opt
                self._generate_tree_recursive(right_opt, depth + 1)
    
    def get_all_positions(self) -> List[CGTPosition]:
        """Get all positions in the tree"""
        positions = [self.root]
        positions.extend(self.position_cache.values())
        return positions
    
    def analyze_tree(self) -> Dict[str, Dict]:
        """
        Perform complete CGT analysis on all positions in the tree.
        
        Returns:
            Dictionary mapping position names to analysis results
        """
        results = {}
        
        # Analyze root
        results[self.root.position_name] = self.root.get_analysis_summary()
        
        # Analyze all cached positions
        for pos_name, position in self.position_cache.items():
            results[pos_name] = position.get_analysis_summary()
        
        self.analysis_cache = results
        return results
    
    def get_positions_by_depth(self, depth: int) -> List[CGTPosition]:
        """Get all positions at a specific depth"""
        positions = []
        self._collect_positions_at_depth(self.root, 0, depth, positions)
        return positions
    
    def _collect_positions_at_depth(self, position: CGTPosition, current_depth: int, 
                                   target_depth: int, positions: List[CGTPosition]) -> None:
        """Helper method to collect positions at specific depth"""
        if current_depth == target_depth:
            positions.append(position)
            return
        
        if current_depth < target_depth:
            for opt in position.left_options + position.right_options:
                self._collect_positions_at_depth(opt, current_depth + 1, target_depth, positions)
    
    def export_tree_structure(self) -> Dict:
        """
        Export the complete tree structure for visualization or further analysis.
        
        Returns:
            Dictionary representing the tree structure with all positions and connections
        """
        def export_position(pos: CGTPosition) -> Dict:
            return {
                'name': pos.position_name,
                'cgt_notation': pos.get_cgt_notation(),
                'value': float(pos.compute_game_value()) if not isinstance(pos.compute_game_value(), float) or pos.compute_game_value() == pos.compute_game_value() else "confused",
                'temperature': pos.compute_temperature(),
                'left_options': [export_position(opt) for opt in pos.left_options],
                'right_options': [export_position(opt) for opt in pos.right_options],
                'is_terminal': pos.is_terminal()
            }
        
        return {
            'root': export_position(self.root),
            'max_depth': self.max_depth,
            'total_positions': len(self.position_cache) + 1,
            'analysis_complete': len(self.analysis_cache) > 0
        }

def create_position_from_war(war_position, war_engine, depth: int = 0, max_depth: int = 3) -> CGTPosition:
    """
    Convert a WarPosition to a CGTPosition with proper {L | R} structure.
    
    This function bridges the War game engine with formal CGT analysis
    by creating the appropriate position representation with limited depth.
    
    Args:
        war_position: WarPosition instance from the game engine
        war_engine: WarGameEngine instance for move generation
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        CGTPosition with computed left and right options
    """
    if war_position.is_terminal() or depth >= max_depth:
        # For non-terminal positions at max depth, treat as terminal for CGT analysis
        return CGTPosition(
            left_options=[],
            right_options=[],
            position_name=f"War_{war_position.position_type}_{len(war_position.player1_hand)}_{len(war_position.player2_hand)}_d{depth}",
            war_position=war_position
        )
    
    # Get all possible next positions
    next_positions = war_engine.get_possible_moves(war_position)
    
    # In War, the game is deterministic, so we have at most one next position
    # We'll create a simplified CGT structure for analysis
    left_options = []
    right_options = []
    
    if next_positions:
        # Take only the first next position to avoid explosion
        next_pos = next_positions[0]
        next_cgt = create_position_from_war(next_pos, war_engine, depth + 1, max_depth)
        
        # Assign to left or right based on who benefits from the move
        # This is a simplified heuristic for War
        if len(next_pos.player1_hand) > len(war_position.player1_hand):
            left_options.append(next_cgt)  # Good for player 1 (Left)
        elif len(next_pos.player2_hand) > len(war_position.player2_hand):
            right_options.append(next_cgt)  # Good for player 2 (Right)
        else:
            # Neutral move, assign to left by default
            left_options.append(next_cgt)
    
    return CGTPosition(
        left_options=left_options,
        right_options=right_options,
        position_name=f"War_{war_position.position_type}_{len(war_position.player1_hand)}_{len(war_position.player2_hand)}_d{depth}",
        war_position=war_position
    )