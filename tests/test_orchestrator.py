from cgt_analysis.war_engine import WarGameEngine
from cgt_analysis.lens_orchestrator import LensOrchestrator
from cgt_analysis.lens_base import available_lenses


def test_structural_lens_registered():
    assert 'structural' in available_lenses()


def test_orchestrator_runs_basic_lenses():
    engine = WarGameEngine(deck_size=48, seed=42)
    orchestrator = LensOrchestrator(engine, lens_ids=['temperature','grundy','structural'], seed=42, num_samples=4, trajectory_simulations=3)
    result = orchestrator.run()
    assert result['game_name'] == 'War'
    assert 'lens_outputs' in result
    assert 'temperature' in result['lens_outputs']
    assert 'grundy' in result['lens_outputs']
    assert 'structural' in result['lens_outputs']
    assert 'metrics' in result['lens_outputs']['structural'] or 'error' not in result['lens_outputs']['structural']
