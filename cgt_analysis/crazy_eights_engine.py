"""
Crazy Eights game engine implementation (skeleton for Task 6).

This module provides a template for implementing Crazy Eights following
the CGT base interface. To be completed by the agent working on Task 6.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random

from .base import GameEngine, GameState, Player


@dataclass
class CrazyEightsState(GameState):
    """
    Represents a state in Crazy Eights.
    
    TODO (Task 6): Complete this implementation for partisan game analysis.
    
    Attributes:
        player1_hand: Cards in Player 1's hand
        player2_hand: Cards in Player 2's hand  
        discard_pile: Cards in discard pile (top card is active)
        draw_pile: Cards remaining in draw pile
        current_suit: Active suit (can be changed by 8s)
        current_rank: Active rank
        player_to_move: Which player's turn
        deck_size: Total deck size
    """
    player1_hand: List[Tuple[int, str]]  # (rank, suit)
    player2_hand: List[Tuple[int, str]]
    discard_pile: List[Tuple[int, str]]
    draw_pile: List[Tuple[int, str]]
    current_suit: str
    current_rank: int
    player_to_move: Player
    deck_size: int = 48
    
    def is_terminal(self) -> bool:
        """Check if game is over (someone has no cards)"""
        # TODO: Implement
        return len(self.player1_hand) == 0 or len(self.player2_hand) == 0
    
    def get_winner(self) -> Optional[Player]:
        """Get winner if terminal"""
        # TODO: Implement
        if not self.is_terminal():
            return None
        return Player.PLAYER1 if len(self.player1_hand) == 0 else Player.PLAYER2
    
    def get_player_to_move(self) -> Player:
        """Get current player to move"""
        return self.player_to_move
    
    def get_state_hash(self) -> str:
        """Get unique hash for this state"""
        # TODO: Implement proper hashing
        import hashlib
        state_str = f"{self.player1_hand}_{self.player2_hand}_{self.current_suit}_{self.current_rank}"
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def get_game_value(self) -> float:
        """
        Heuristic value for this state.
        
        TODO: Implement sophisticated evaluation considering:
        - Number of cards in each hand
        - Number of 8s (wild cards)
        - Suit control
        - Playable cards
        """
        if self.is_terminal():
            # Player with no cards wins
            return 1.0 if len(self.player1_hand) == 0 else -1.0
        
        # Simple heuristic: negative card difference
        # (fewer cards is better in Crazy Eights)
        card_diff = len(self.player2_hand) - len(self.player1_hand)
        return card_diff / self.deck_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'player1_hand': self.player1_hand,
            'player2_hand': self.player2_hand,
            'discard_pile': self.discard_pile,
            'draw_pile_size': len(self.draw_pile),
            'current_suit': self.current_suit,
            'current_rank': self.current_rank,
            'player_to_move': self.player_to_move.value,
            'is_terminal': self.is_terminal(),
            'winner': self.get_winner().value if self.get_winner() else None
        }


class CrazyEightsEngine(GameEngine):
    """
    Crazy Eights game engine implementation.
    
    Crazy Eights is a PARTISAN game where players have different available
    moves based on their hands. This makes it fundamentally different from War.
    
    TODO (Task 6): Complete this implementation following the pattern in war_engine.py
    """
    
    def __init__(self, deck_size: int = 48, seed: Optional[int] = None):
        """Initialize Crazy Eights engine"""
        super().__init__(deck_size, seed)
        if seed is not None:
            random.seed(seed)
        
        # Define suits and ranks based on deck size
        self.suits = ['♠', '♥', '♦', '♣']
        
        if deck_size == 44:
            self.min_rank = 5  # No 2s, 3s, 4s
        elif deck_size == 48:
            self.min_rank = 4  # No 2s, 3s
        else:  # 52
            self.min_rank = 2
        
        self.max_rank = 14  # Ace high
    
    def create_initial_state(self) -> CrazyEightsState:
        """Create initial Crazy Eights state"""
        # TODO: Implement proper deck creation and dealing
        
        # Create deck
        deck = []
        for rank in range(self.min_rank, self.max_rank + 1):
            for suit in self.suits:
                deck.append((rank, suit))
        
        # Shuffle
        random.shuffle(deck)
        
        # Deal 7 cards to each player
        player1_hand = deck[:7]
        player2_hand = deck[7:14]
        draw_pile = deck[14:]
        
        # Start discard pile with one card
        discard_pile = [draw_pile.pop()]
        
        return CrazyEightsState(
            player1_hand=player1_hand,
            player2_hand=player2_hand,
            discard_pile=discard_pile,
            draw_pile=draw_pile,
            current_suit=discard_pile[-1][1],
            current_rank=discard_pile[-1][0],
            player_to_move=Player.PLAYER1,
            deck_size=self.deck_size
        )
    
    def get_possible_moves(self, state: CrazyEightsState) -> List[Dict[str, Any]]:
        """
        Get all possible moves for current player.
        
        Moves in Crazy Eights:
        - Play a card matching suit or rank
        - Play an 8 (wild) and declare suit
        - Draw a card (if no playable cards)
        
        TODO: Implement full move generation
        """
        if state.is_terminal():
            return []
        
        moves = []
        current_player_hand = (state.player1_hand if state.player_to_move == Player.PLAYER1 
                              else state.player2_hand)
        
        # Check for playable cards
        for i, (rank, suit) in enumerate(current_player_hand):
            if rank == 8:  # Wild card
                # Can play 8 and declare any suit
                for new_suit in self.suits:
                    moves.append({
                        'type': 'play_eight',
                        'card_index': i,
                        'card': (rank, suit),
                        'declare_suit': new_suit
                    })
            elif suit == state.current_suit or rank == state.current_rank:
                # Can play matching card
                moves.append({
                    'type': 'play',
                    'card_index': i,
                    'card': (rank, suit)
                })
        
        # Can always draw if draw pile not empty
        if state.draw_pile:
            moves.append({'type': 'draw'})
        
        return moves
    
    def apply_move(self, state: CrazyEightsState, move: Dict[str, Any]) -> CrazyEightsState:
        """
        Apply a move to get next state.
        
        TODO: Implement complete move application logic
        """
        # This is a skeleton implementation
        # Task 6 agent should complete this
        
        # Create new state (deep copy)
        new_state = CrazyEightsState(
            player1_hand=state.player1_hand.copy(),
            player2_hand=state.player2_hand.copy(),
            discard_pile=state.discard_pile.copy(),
            draw_pile=state.draw_pile.copy(),
            current_suit=state.current_suit,
            current_rank=state.current_rank,
            player_to_move=state.player_to_move,
            deck_size=state.deck_size
        )
        
        # Apply move based on type
        if move['type'] == 'play' or move['type'] == 'play_eight':
            # Remove card from hand
            # Add to discard
            # Update current suit/rank
            # Switch player
            pass  # TODO: Implement
        elif move['type'] == 'draw':
            # Draw card from pile
            # Switch player only if can't play drawn card
            pass  # TODO: Implement
        
        # Switch to next player
        new_state.player_to_move = (Player.PLAYER2 if state.player_to_move == Player.PLAYER1 
                                   else Player.PLAYER1)
        
        return new_state
    
    def get_next_states(self, state: CrazyEightsState) -> List[Tuple[Dict, CrazyEightsState]]:
        """Get all possible next states"""
        moves = self.get_possible_moves(state)
        return [(move, self.apply_move(state, move)) for move in moves]
    
    def simulate_game(self, initial_state: Optional[CrazyEightsState] = None) -> Dict[str, Any]:
        """
        Simulate a complete Crazy Eights game.
        
        TODO: Implement complete game simulation with:
        - Proper move selection (random or strategic)
        - Stalemate detection
        - Temperature tracking
        """
        if initial_state is None:
            state = self.create_initial_state()
        else:
            state = initial_state
        
        num_moves = 0
        max_moves = 1000  # Prevent infinite loops
        
        while not state.is_terminal() and num_moves < max_moves:
            moves = self.get_possible_moves(state)
            
            if not moves:
                break  # Stalemate
            
            # Random move selection (TODO: Add strategy)
            move = random.choice(moves)
            state = self.apply_move(state, move)
            num_moves += 1
        
        return {
            'winner': state.get_winner(),
            'num_moves': num_moves,
            'final_state': state,
            'stalemate': num_moves >= max_moves
        }
    
    def evaluate_position(self, state: CrazyEightsState) -> Dict[str, float]:
        """
        Evaluate a Crazy Eights position.
        
        TODO: Implement sophisticated evaluation considering:
        - Hand strength (8s, playable cards)
        - Suit control
        - Draw pile size
        - Opponent's hand size
        """
        # Basic evaluation
        game_value = state.get_game_value()
        
        # Temperature based on number of choices
        moves = self.get_possible_moves(state)
        temperature = len(moves) / 10.0  # Normalize
        
        # Confidence based on card difference
        card_diff = abs(len(state.player1_hand) - len(state.player2_hand))
        confidence = min(1.0, card_diff / 5.0)
        
        return {
            'game_value': game_value,
            'temperature': temperature,
            'confidence': confidence
        }
    
    @property
    def game_name(self) -> str:
        """Get game name"""
        return "CrazyEights"
    
    @property
    def is_impartial(self) -> bool:
        """Crazy Eights is a PARTISAN game"""
        return False
    
    def create_position_opening(self) -> CrazyEightsState:
        """
        Create opening position for analysis.
        
        TODO (Task 6): Create meaningful test positions like:
        - Wild eight decision
        - Suit lock scenario
        - Endgame racing
        - Stalemate approach
        """
        # Simplified opening position
        return CrazyEightsState(
            player1_hand=[(8, '♠'), (13, '♠'), (5, '♦')],
            player2_hand=[(12, '♠'), (5, '♣'), (7, '♦')],
            discard_pile=[(10, '♠')],
            draw_pile=[(i, s) for i in range(self.min_rank, 8) 
                      for s in self.suits][:10],
            current_suit='♠',
            current_rank=10,
            player_to_move=Player.PLAYER1,
            deck_size=self.deck_size
        )


# TODO (Task 6): Add more position creation methods
# TODO (Task 6): Add partisan game analysis specific methods
# TODO (Task 6): Add thermographic analysis for partisan games
# TODO (Task 6): Document differences from impartial game analysis
