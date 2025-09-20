from cgt_analysis.war_engine import WarGameEngine
from cgt_analysis.base import CGTAnalyzer
from cgt_analysis.lens_base import get_lens, AnalysisContext, available_lenses


def test_available_lenses_phase1():
    lenses = available_lenses()
    assert 'temperature' in lenses
    assert 'grundy' in lenses


def test_temperature_lens_runs():
    engine = WarGameEngine(deck_size=48, seed=42)
    analyzer = CGTAnalyzer(engine)
    trajectories = []
    for _ in range(3):
        res = engine.simulate_game()
        trajectories.append(res['trajectory'])
    ctx = AnalysisContext(analyzer=analyzer, engine=engine, trajectories=trajectories, sample_positions=[])
    lens = get_lens('temperature')
    output = lens.compute(ctx)
    assert output['lens_id'] == 'temperature'
    assert 'metrics' in output


def test_grundy_lens_runs():
    engine = WarGameEngine(deck_size=48, seed=42)
    analyzer = CGTAnalyzer(engine)
    sample_positions = [engine.create_position_a(), engine.create_position_b()]
    ctx = AnalysisContext(analyzer=analyzer, engine=engine, sample_positions=sample_positions, trajectories=[])
    lens = get_lens('grundy')
    output = lens.compute(ctx)
    assert output['lens_id'] == 'grundy'
    assert 'metrics' in output or 'diagnostics' in output
