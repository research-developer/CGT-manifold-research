"""Lens framework for modular analytical perspectives.

Phase 1: Provides a minimal pluggable infrastructure plus two lenses:
- TemperatureLens: derives temperature statistics from Monte Carlo trajectories
- GrundyLens: computes Grundy distribution over sampled positions using existing analyzer

Each lens returns a standardized dict structure:
{
  'lens_id': str,
  'schema_version': '1.0.0',
  'parameters': {...},
  'metrics': { scalar_name: value },
  'series': { series_name: [ { 'x': ..., 'y': ...}, ... ] },
  'diagnostics': { 'warnings': [], 'compute_time_ms': int }
}

The orchestrator logic will be expanded in later phases. For now, lenses
can be invoked manually.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable, Callable, Type
import time

# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class AnalysisContext:
    """Lightweight context object passed to lenses.

    Expected optional fields (Phase 1):
      - 'analyzer': CGTAnalyzer instance
      - 'engine': GameEngine instance
      - 'sample_positions': List[GameState]
      - 'trajectories': List[List]

    Lenses should degrade gracefully if required inputs are missing.
    """
    def __init__(self, **kwargs):
        self._data = kwargs

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def require(self, key: str):
        if key not in self._data:
            raise KeyError(f"AnalysisContext missing required key: {key}")
        return self._data[key]


class AbstractLens(ABC):
    lens_id: str = "abstract"
    schema_version: str = "1.0.0"

    @abstractmethod
    def required_inputs(self) -> Iterable[str]:
        """Return iterable of context keys required."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        """Perform computation and return standardized result dict."""
        raise NotImplementedError

    # Helper to build standard skeleton
    def _result_template(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'lens_id': self.lens_id,
            'schema_version': self.schema_version,
            'parameters': parameters,
            'metrics': {},
            'series': {},
            'diagnostics': {'warnings': [], 'compute_time_ms': 0}
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LENS_REGISTRY: Dict[str, Type[AbstractLens]] = {}

def register_lens(cls: Type[AbstractLens]):
    if cls.lens_id in _LENS_REGISTRY:
        raise ValueError(f"Lens id already registered: {cls.lens_id}")
    _LENS_REGISTRY[cls.lens_id] = cls
    return cls


def get_lens(lens_id: str) -> AbstractLens:
    if lens_id not in _LENS_REGISTRY:
        raise KeyError(f"Lens '{lens_id}' not found. Registered: {list(_LENS_REGISTRY)}")
    return _LENS_REGISTRY[lens_id]()


def available_lenses() -> List[str]:
    return sorted(_LENS_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Temperature Lens
# ---------------------------------------------------------------------------
@register_lens
class TemperatureLens(AbstractLens):
    lens_id = 'temperature'

    def required_inputs(self) -> Iterable[str]:
        return ['trajectories']

    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        start = time.time()
        result = self._result_template(parameters={})
        trajectories = ctx.get('trajectories', [])

        temps: List[float] = []
        temp_series: List[Dict[str, Any]] = []

        # We assume each trajectory is a list of states each having get_game_value
        # For Phase 1, approximate temperature as 2.0 * (remaining_cards_ratio)
        for ti, traj in enumerate(trajectories):
            if not traj:
                continue
            local_series = []
            for si, state in enumerate(traj):
                # Best effort: try attributes used by WarPosition
                deck_size = getattr(state, 'deck_size', None)
                if deck_size:
                    total_cards = len(getattr(state, 'player1_hand', [])) + len(getattr(state, 'player2_hand', []))
                    temperature = 2.0 * (total_cards / deck_size)
                    temps.append(temperature)
                    local_series.append({'x': si, 'y': temperature})
            if local_series:
                temp_series.append({'trajectory_index': ti, 'points': local_series})

        if temps:
            import numpy as np
            result['metrics'] = {
                'mean_temperature': float(np.mean(temps)),
                'max_temperature': float(np.max(temps)),
                'min_temperature': float(np.min(temps)),
                'std_temperature': float(np.std(temps)),
                'num_samples': len(temps)
            }
        else:
            result['diagnostics']['warnings'].append('No temperature samples found')

        result['series']['temperature_trajectories'] = temp_series
        result['diagnostics']['compute_time_ms'] = int((time.time() - start) * 1000)
        return result


# ---------------------------------------------------------------------------
# Grundy Lens
# ---------------------------------------------------------------------------
@register_lens
class GrundyLens(AbstractLens):
    lens_id = 'grundy'

    def required_inputs(self) -> Iterable[str]:
        return ['analyzer', 'sample_positions']

    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        start = time.time()
        result = self._result_template(parameters={'max_depth': 2})
        analyzer = ctx.require('analyzer')
        positions = ctx.get('sample_positions', [])

        grundy_values: List[int] = []
        summaries: List[Dict[str, Any]] = []
        for pos in positions:
            analysis = analyzer.analyze_position(pos, max_depth=2)
            g = analysis.get('grundy_number')
            if g is not None:
                grundy_values.append(g)
            summaries.append({'position_hash': pos.get_state_hash(), 'grundy': g})

        if grundy_values:
            import numpy as np
            unique, counts = np.unique(grundy_values, return_counts=True)
            distribution = {int(k): int(v) for k, v in zip(unique, counts)}
            result['metrics'] = {
                'unique_grundy_values': len(unique),
                'grundy_distribution_entropy': float(-sum((c/len(grundy_values))*__import__('math').log2(c/len(grundy_values)) for c in counts)),
                'num_positions': len(grundy_values)
            }
            result['series']['grundy_distribution'] = [
                {'value': int(k), 'count': int(v)} for k, v in distribution.items()
            ]
        else:
            result['diagnostics']['warnings'].append('No Grundy values computed')

        result['series']['sample_summaries'] = summaries
        result['diagnostics']['compute_time_ms'] = int((time.time() - start) * 1000)
        return result


__all__ = [
    'AbstractLens', 'AnalysisContext', 'TemperatureLens', 'GrundyLens', 'StructuralLens',
    'get_lens', 'available_lenses'
]

# ---------------------------------------------------------------------------
# Structural Lens (added in Phase 2)
# ---------------------------------------------------------------------------
@register_lens
class StructuralLens(AbstractLens):
    lens_id = 'structural'

    def required_inputs(self):
        return ['engine', 'analyzer', 'sample_positions']

    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        start = time.time()
        result = self._result_template(parameters={'max_nodes': 200, 'max_depth': 4})
        engine = ctx.require('engine')
        analyzer = ctx.require('analyzer')
        seeds = ctx.get('sample_positions', [])
        seen = {}
        edges = 0

        def add_state(state):
            h = state.get_state_hash()
            if h not in seen:
                seen[h] = {
                    'terminal': state.is_terminal(),
                    'value': state.get_game_value()
                }
            return h

        frontier = []
        for s in seeds[:5]:
            frontier.append((s, 0))
            add_state(s)

        while frontier and len(seen) < 200:
            state, depth = frontier.pop(0)
            if depth >= 4:
                continue
            if state.is_terminal():
                continue
            nexts = engine.get_next_states(state)
            for _move, nxt in nexts:
                edges += 1
                h = add_state(nxt)
                if len(seen) < 200:
                    frontier.append((nxt, depth + 1))

        if seen:
            import numpy as np
            values = [d['value'] for d in seen.values()]
            terminals = sum(1 for d in seen.values() if d['terminal'])
            result['metrics'] = {
                'num_nodes': len(seen),
                'num_edges': edges,
                'terminal_fraction': terminals / len(seen),
                'value_mean': float(np.mean(values)),
                'value_std': float(np.std(values)),
            }
        else:
            result['diagnostics']['warnings'].append('No nodes explored')

        result['diagnostics']['compute_time_ms'] = int((time.time() - start) * 1000)
        return result
