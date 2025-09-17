"""
Combinatorial Game Theory Analysis Module

This module provides comprehensive CGT analysis tools for card games,
specifically focused on the War card game and the 2^n×k structural resonance principle.
"""

__version__ = "1.0.0"
__author__ = "Preston Temple"

from .war_engine import WarGameEngine, WarPosition
from .cgt_position import CGTPosition, GameTree
from .grundy_numbers import GrundyCalculator
from .temperature_analysis import TemperatureCalculator
from .thermographic_analysis import ThermographAnalyzer

__all__ = [
    'WarGameEngine',
    'WarPosition', 
    'CGTPosition',
    'GameTree',
    'GrundyCalculator',
    'TemperatureCalculator',
    'ThermographAnalyzer'
]