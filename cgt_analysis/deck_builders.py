"""Deck builder strategies (Phase 1 scaffolding).

Provides a registry-based interface similar to lenses so future strategies
can be plugged in without modifying core engines.
"""
from __future__ import annotations
from typing import List, Dict, Type, Callable
import random

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class AbstractDeckBuilder:
    strategy_id = 'abstract'
    def build(self, deck_size: int, seed: int | None = None, **policy) -> List[int]:
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_DECK_REGISTRY: Dict[str, Type[AbstractDeckBuilder]] = {}

def register_deck(cls: Type[AbstractDeckBuilder]):
    if cls.strategy_id in _DECK_REGISTRY:
        raise ValueError(f"Deck strategy already registered: {cls.strategy_id}")
    _DECK_REGISTRY[cls.strategy_id] = cls
    return cls


def get_deck_builder(strategy_id: str) -> AbstractDeckBuilder:
    if strategy_id not in _DECK_REGISTRY:
        raise KeyError(f"Deck strategy '{strategy_id}' not found. Registered: {list(_DECK_REGISTRY)}")
    return _DECK_REGISTRY[strategy_id]()


def available_deck_strategies() -> List[str]:
    return sorted(_DECK_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Standard truncation (placeholder consistent with engine defaults)
# ---------------------------------------------------------------------------
@register_deck
class StandardTruncation(AbstractDeckBuilder):
    strategy_id = 'standard'

    def build(self, deck_size: int, seed: int | None = None, **policy) -> List[int]:
        if seed is not None:
            random.seed(seed)
        # Defer to existing WarGameEngine logic: placeholder just returns empty to signal engine default.
        return []  # Engine will construct its own default deck

# ---------------------------------------------------------------------------
# High bias augmentation strategy (explicit duplicate of high ranks)
# ---------------------------------------------------------------------------
@register_deck
class HighBiasAugmentation(AbstractDeckBuilder):
    strategy_id = 'high_bias'

    def build(self, deck_size: int, seed: int | None = None, **policy) -> List[int]:
        if seed is not None:
            random.seed(seed)
        # Simplified: build ascending then pad with max rank copies
        if deck_size not in (44, 48, 52):
            base_min = 2
            base_max = 14
        else:
            # mimic current engine min_rank logic indirectly
            base_min = {44:5, 48:4, 52:2}[deck_size]
            base_max = 14
        ranks = list(range(base_min, base_max + 1))
        # naive fill distributing equally until near deck_size, then pad with Aces
        deck: List[int] = []
        idx = 0
        while len(deck) + len(ranks) <= deck_size:
            deck.extend(ranks)
        # Fill remainder with highest rank
        while len(deck) < deck_size:
            deck.append(base_max)
        random.shuffle(deck)
        return deck

__all__ = [
    'AbstractDeckBuilder', 'StandardTruncation', 'HighBiasAugmentation',
    'get_deck_builder', 'available_deck_strategies'
]
