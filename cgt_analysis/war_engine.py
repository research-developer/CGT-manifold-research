"""
War game engine implementation following the CGT base interface.

This module provides a complete War game implementation that integrates
with the CGT analysis framework.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random
import hashlib

from .base import GameEngine, GameState, Player


@dataclass
class WarPosition(GameState):
    """
    Represents a position in the War card game.
    
    Attributes:
        player1_hand: List of cards in Player 1's hand
        player2_hand: List of cards in Player 2's hand
        player1_pile: Cards Player 1 has played (during war)
        player2_pile: Cards Player 2 has played (during war)
        position_type: Type of position (opening, battle, etc.)
        deck_size: Total deck size being used
    """
    player1_hand: List[int]
    player2_hand: List[int]
    player1_pile: List[int] = None
    player2_pile: List[int] = None
    position_type: str = "standard"
    deck_size: int = 48
    
    def __post_init__(self):
        if self.player1_pile is None:
            self.player1_pile = []
        if self.player2_pile is None:
            self.player2_pile = []
    
    def is_terminal(self) -> bool:
        """Check if game is over (one player has all cards)"""
        return len(self.player1_hand) == 0 or len(self.player2_hand) == 0
    
    def get_winner(self) -> Optional[Player]:
        """Get the winner if terminal"""
        if not self.is_terminal():
            return None
        return Player.PLAYER1 if len(self.player1_hand) > 0 else Player.PLAYER2
    
    def get_player_to_move(self) -> Player:
        """In War, both players move simultaneously (simplified to P1)"""
        return Player.PLAYER1
    
    def get_state_hash(self) -> str:
        """Get unique hash for this state"""
        state_str = f"{sorted(self.player1_hand)}_{sorted(self.player2_hand)}_{self.player1_pile}_{self.player2_pile}"
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def hash_key(self) -> str:
        """Get hash key for caching (alias for get_state_hash)"""
        return self.get_state_hash()
    
    def get_game_value(self) -> float:
        """
        Heuristic value: card difference normalized by deck size.
        Positive = P1 advantage, Negative = P2 advantage
        """
        if self.is_terminal():
            return 1.0 if self.get_winner() == Player.PLAYER1 else -1.0
        
        # Card count difference
        card_diff = len(self.player1_hand) - len(self.player2_hand)
        
        # Normalize by deck size
        return card_diff / self.deck_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'player1_hand': self.player1_hand,
            'player2_hand': self.player2_hand,
            'player1_pile': self.player1_pile,
            'player2_pile': self.player2_pile,
            'position_type': self.position_type,
            'deck_size': self.deck_size,
            'is_terminal': self.is_terminal(),
            'winner': self.get_winner().value if self.get_winner() else None,
            'game_value': self.get_game_value()
        }


class WarGameEngine(GameEngine):
    """
    War game engine implementation.
    
    War is a simple impartial game where both players reveal cards
    simultaneously and the higher card wins both cards.
    """
    
    def __init__(self, deck_size: int = 48, seed: Optional[int] = None):
        """
        Initialize War game engine.
        
        Args:
            deck_size: Number of cards (44, 48, or 52)
            seed: Random seed for reproducibility
        """
        super().__init__(deck_size, seed)
        if seed is not None:
            random.seed(seed)
        
        # Define card ranks based on deck size
        if deck_size == 44:
            # Remove 2s, 3s, 4s
            self.min_rank = 5
        elif deck_size == 48:
            # Remove 2s, 3s
            self.min_rank = 4
        else:  # 52
            self.min_rank = 2
        
        self.max_rank = 14  # Ace high
    
    def create_initial_state(self) -> WarPosition:
        """Create initial War game state with shuffled deck"""
        # Create deck
        deck = []
        for rank in range(self.min_rank, self.max_rank + 1):
            for _ in range(4):  # 4 suits
                deck.append(rank)
        
        # Shuffle
        random.shuffle(deck)
        
        # Deal cards
        mid = len(deck) // 2
        return WarPosition(
            player1_hand=deck[:mid],
            player2_hand=deck[mid:],
            position_type="opening",
            deck_size=self.deck_size
        )
    
    def get_possible_moves(self, state: WarPosition) -> List[str]:
        """
        Get possible moves (in War, only one move: play top card).
        
        Returns:
            List with single move "play" or empty if terminal
        """
        if state.is_terminal():
            return []
        return ["play"]
    
    def apply_move(self, state: WarPosition, move: str) -> WarPosition:
        """
        Apply move to get next state.
        
        In War, both players play their top cards simultaneously.
        """
        if state.is_terminal() or move != "play":
            return state
        
        # Copy state
        new_state = WarPosition(
            player1_hand=state.player1_hand.copy(),
            player2_hand=state.player2_hand.copy(),
            player1_pile=state.player1_pile.copy(),
            player2_pile=state.player2_pile.copy(),
            position_type="battle",
            deck_size=state.deck_size
        )
        
        # Play cards
        if new_state.player1_hand and new_state.player2_hand:
            card1 = new_state.player1_hand.pop(0)
            card2 = new_state.player2_hand.pop(0)
            
            new_state.player1_pile.append(card1)
            new_state.player2_pile.append(card2)
            
            # Determine winner
            if card1 > card2:
                # Player 1 wins
                new_state.player1_hand.extend(new_state.player1_pile)
                new_state.player1_hand.extend(new_state.player2_pile)
                new_state.player1_pile = []
                new_state.player2_pile = []
            elif card2 > card1:
                # Player 2 wins
                new_state.player2_hand.extend(new_state.player1_pile)
                new_state.player2_hand.extend(new_state.player2_pile)
                new_state.player1_pile = []
                new_state.player2_pile = []
            else:
                # War! (tie)
                new_state.position_type = "war"
                # In simplified War, just play another round
                # In full War, would add 3 face-down cards first
        
        return new_state
    
    def get_next_states(self, state: WarPosition) -> List[Tuple[str, WarPosition]]:
        """Get all possible next states"""
        moves = self.get_possible_moves(state)
        return [(move, self.apply_move(state, move)) for move in moves]
    
    def simulate_game(self, initial_state: Optional[WarPosition] = None) -> Dict[str, Any]:
        """Simulate a complete War game"""
        if initial_state is None:
            state = self.create_initial_state()
        else:
            state = initial_state
        
        num_moves = 0
        max_moves = 10000  # Prevent infinite loops
        states = [state]
        
        while not state.is_terminal() and num_moves < max_moves:
            state = self.apply_move(state, "play")
            states.append(state)
            num_moves += 1
        
        return {
            'winner': state.get_winner(),
            'num_moves': num_moves,
            'final_state': state,
            'trajectory': states
        }
    
    def evaluate_position(self, state: WarPosition) -> Dict[str, float]:
        """Evaluate a War position for CGT analysis"""
        # Basic evaluation
        game_value = state.get_game_value()
        
        # Temperature based on game stage
        total_cards = len(state.player1_hand) + len(state.player2_hand)
        temperature = 2.0 * (total_cards / self.deck_size)  # Higher early game
        
        # Confidence based on card difference
        card_diff = abs(len(state.player1_hand) - len(state.player2_hand))
        confidence = min(1.0, card_diff / 10.0)
        
        return {
            'game_value': game_value,
            'temperature': temperature,
            'confidence': confidence
        }
    
    @property
    def game_name(self) -> str:
        """Get game name"""
        return "War"
    
    @property
    def is_impartial(self) -> bool:
        """War is an impartial game"""
        return True
    
    def create_position_a(self) -> WarPosition:
        """Position A: Opening with high vs low cards"""
        return WarPosition(
            player1_hand=[13, 12, 11, 10, 9],  # High cards
            player2_hand=[5, 6, 7, 8, 4],      # Low cards
            position_type="opening_advantage",
            deck_size=self.deck_size
        )
    
    def create_position_b(self) -> WarPosition:
        """Position B: Tied battle scenario"""
        return WarPosition(
            player1_hand=[10, 8, 6],
            player2_hand=[10, 9, 5],
            player1_pile=[10],
            player2_pile=[10],
            position_type="war",
            deck_size=self.deck_size
        )
    
    def create_position_c(self) -> WarPosition:
        """Position C: Balanced mid-game"""
        cards_per_player = self.deck_size // 4
        return WarPosition(
            player1_hand=list(range(self.min_rank, self.min_rank + cards_per_player)),
            player2_hand=list(range(self.min_rank, self.min_rank + cards_per_player)),
            position_type="mid_game",
            deck_size=self.deck_size
        )
    
    def create_position_d(self) -> WarPosition:
        """Position D: Endgame with few cards"""
        return WarPosition(
            player1_hand=[12, 8],
            player2_hand=[11, 9],
            position_type="endgame",
            deck_size=self.deck_size
        )
    
    def create_position_e(self) -> WarPosition:
        """Position E: Extreme advantage (all face cards)"""
        return WarPosition(
            player1_hand=[14, 13, 12, 11],  # All face cards
            player2_hand=[6, 5, 4, 7],      # All low cards
            position_type="deterministic_win",
            deck_size=self.deck_size
        )
