"""Lens orchestrator (Phase 2).

Centralizes sampling and execution of registered lenses with lazy artifact creation.
"""
from __future__ import annotations
from typing import List, Dict, Any, Sequence, Optional
import random
from .lens_base import get_lens, AnalysisContext
from .base import CGTAnalyzer

class LensOrchestrator:
    def __init__(self, engine, lens_ids: Sequence[str], seed: int | None = None,
                 num_samples: int = 6, trajectory_simulations: int = 5):
        self.engine = engine
        self.lens_ids = list(lens_ids)
        self.seed = seed
        self.num_samples = num_samples
        self.trajectory_simulations = trajectory_simulations
        if seed is not None:
            random.seed(seed)
        self.analyzer = CGTAnalyzer(engine)

    def sample_positions(self) -> List[Any]:
        samples = []
        # Always include canonical positions if available
        creator_methods = [
            getattr(self.engine, 'create_position_a', None),
            getattr(self.engine, 'create_position_b', None),
            getattr(self.engine, 'create_position_c', None),
            getattr(self.engine, 'create_position_d', None),
            getattr(self.engine, 'create_position_e', None),
        ]
        for m in creator_methods:
            if callable(m):
                samples.append(m())
        # Supplement with random partial simulations
        while len(samples) < self.num_samples:
            st = self.engine.create_initial_state()
            # advance a few moves
            steps = random.randint(0, 5)
            for _ in range(steps):
                if st.is_terminal():
                    break
                nexts = self.engine.get_next_states(st)
                if not nexts:
                    break
                st = nexts[0][1]
            samples.append(st)
        return samples[: self.num_samples]

    def simulate_trajectories(self) -> List[List[Any]]:
        trajs = []
        for _ in range(self.trajectory_simulations):
            res = self.engine.simulate_game()
            trajs.append(res.get('trajectory', []))
        return trajs

    def run(self) -> Dict[str, Any]:
        sample_positions = self.sample_positions()
        trajectories = self.simulate_trajectories()
        ctx = AnalysisContext(
            analyzer=self.analyzer,
            engine=self.engine,
            sample_positions=sample_positions,
            trajectories=trajectories
        )
        outputs: Dict[str, Any] = {}
        for lens_id in self.lens_ids:
            try:
                lens = get_lens(lens_id)
                outputs[lens_id] = lens.compute(ctx)
            except Exception as e:  # Capture errors per lens
                outputs[lens_id] = {'error': str(e)}
        return {
            'deck_size': self.engine.deck_size,
            'game_name': self.engine.game_name,
            'lens_outputs': outputs
        }

__all__ = ['LensOrchestrator']
