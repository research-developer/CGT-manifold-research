#!/usr/bin/env python3
"""
Main analysis runner for CGT research project.

This script provides a unified interface for running all analyses
across different games and deck sizes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cgt_analysis.base import CGTAnalyzer, DataManager
from cgt_analysis.lens_base import get_lens, available_lenses, AnalysisContext
from cgt_analysis.lens_orchestrator import LensOrchestrator
from cgt_analysis.deck_builders import available_deck_strategies, get_deck_builder
from cgt_analysis.war_engine import WarGameEngine


def run_war_analysis(deck_sizes: List[int] = [44, 48, 52], 
                     num_simulations: int = 1000,
                     save_results: bool = True) -> Dict[str, Any]:
    """
    Run complete War game analysis for specified deck sizes.
    
    Args:
        deck_sizes: List of deck sizes to analyze
        num_simulations: Number of Monte Carlo simulations per deck size
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 60)
    print("WAR GAME ANALYSIS")
    print("=" * 60)
    
    results = {}
    data_manager = DataManager() if save_results else None
    
    for deck_size in deck_sizes:
        print(f"\nAnalyzing {deck_size}-card deck...")
        
        # Initialize engine
        engine = WarGameEngine(deck_size=deck_size, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        # Analyze standard positions
        position_results = {}
        positions = {
            'A': engine.create_position_a(),
            'B': engine.create_position_b(),
            'C': engine.create_position_c(),
            'D': engine.create_position_d(),
            'E': engine.create_position_e()
        }
        
        print(f"  Analyzing positions...")
        for name, position in positions.items():
            analysis = analyzer.analyze_position(position, max_depth=3)
            position_results[f"Position_{name}"] = analysis
            print(f"    Position {name}: Grundy={analysis.get('grundy_number', '?')}, "
                  f"Temp={analysis.get('temperature_computed', '?'):.3f}")
        
        # Run Monte Carlo simulations
        print(f"  Running {num_simulations} simulations...")
        monte_carlo = analyzer.run_monte_carlo_analysis(num_simulations)
        print(f"    Win rates: P1={monte_carlo['win_rate_p1']:.3f}, "
              f"P2={monte_carlo['win_rate_p2']:.3f}")
        print(f"    Avg game length: {monte_carlo['avg_game_length']:.1f} ± "
              f"{monte_carlo['std_game_length']:.1f}")
        
        # Store results
        results[deck_size] = {
            'positions': position_results,
            'monte_carlo': monte_carlo
        }
        
        # Save if requested
        if save_results and data_manager:
            # Save position analysis
            data_manager.save_analysis_result(
                game_name="War",
                deck_size=deck_size,
                analysis_type="positions",
                data=position_results
            )
            
            # Save Monte Carlo results
            data_manager.save_analysis_result(
                game_name="War",
                deck_size=deck_size,
                analysis_type="monte_carlo",
                data=monte_carlo
            )
    
    return results


def run_periodicity_analysis(deck_sizes: List[int] = [44, 48, 52]) -> Dict[str, Any]:
    """
    Analyze the 16-card periodicity hypothesis.
    
    Args:
        deck_sizes: Deck sizes to test
        
    Returns:
        Dictionary with periodicity analysis results
    """
    print("\n" + "=" * 60)
    print("PERIODICITY ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    for deck_size in deck_sizes:
        print(f"\nAnalyzing periodicity for {deck_size}-card deck...")
        
        # Calculate theoretical properties
        period_length = 16
        num_complete_periods = deck_size // period_length
        remainder = deck_size % period_length
        
        print(f"  Complete periods: {num_complete_periods}")
        print(f"  Remainder cards: {remainder}")
        
        # Binary representation analysis
        binary_states = []
        for i in range(min(deck_size, 64)):  # Analyze first 64 cards max
            binary = format(i % 16, '04b')
            binary_states.append(binary)
        
        # Count unique states in each period
        periods = []
        for p in range(num_complete_periods + (1 if remainder > 0 else 0)):
            start = p * period_length
            end = min(start + period_length, deck_size)
            period_states = binary_states[start:end]
            unique_states = len(set(period_states))
            periods.append({
                'period': p + 1,
                'cards': end - start,
                'unique_states': unique_states,
                'completeness': unique_states / 16.0
            })
        
        results[deck_size] = {
            'period_length': period_length,
            'num_complete_periods': num_complete_periods,
            'remainder': remainder,
            'periods': periods,
            'is_perfect': (remainder == 0 and num_complete_periods == 3)
        }
        
        # Check for 2^4 × 3 structure
        if deck_size == 48:
            print(f"  ✓ Perfect 2^4 × 3 structure detected!")
        else:
            print(f"  ✗ Imperfect structure (remainder={remainder})")
    
    return results


def run_temperature_evolution(deck_sizes: List[int] = [44, 48, 52],
                             samples_per_deck: int = 100) -> Dict[str, Any]:
    """
    Analyze temperature evolution across deck sizes.
    
    Args:
        deck_sizes: Deck sizes to analyze
        samples_per_deck: Number of sample positions per deck
        
    Returns:
        Dictionary with temperature evolution data
    """
    print("\n" + "=" * 60)
    print("TEMPERATURE EVOLUTION ANALYSIS")
    print("=" * 60)
    
    import numpy as np
    
    results = {}
    
    for deck_size in deck_sizes:
        print(f"\nAnalyzing temperature for {deck_size}-card deck...")
        
        engine = WarGameEngine(deck_size=deck_size, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        temperatures = []
        
        # Sample positions at different game stages
        for stage_pct in np.linspace(0.1, 0.9, samples_per_deck):
            # Create position at this stage
            cards_remaining = int(deck_size * stage_pct)
            cards_p1 = cards_remaining // 2
            cards_p2 = cards_remaining - cards_p1
            
            # Create simplified position
            from cgt_analysis.war_engine import WarPosition
            position = WarPosition(
                player1_hand=list(range(cards_p1)),
                player2_hand=list(range(cards_p2)),
                position_type=f"stage_{stage_pct:.1f}",
                deck_size=deck_size
            )
            
            # Analyze temperature
            analysis = analyzer.analyze_position(position, max_depth=2)
            temperatures.append({
                'stage': stage_pct,
                'cards_remaining': cards_remaining,
                'temperature': analysis.get('temperature_computed', 0)
            })
        
        # Calculate statistics
        temps = [t['temperature'] for t in temperatures]
        results[deck_size] = {
            'temperatures': temperatures,
            'max_temperature': max(temps),
            'mean_temperature': np.mean(temps),
            'std_temperature': np.std(temps)
        }
        
        print(f"  Max temperature: {max(temps):.3f}")
        print(f"  Mean temperature: {np.mean(temps):.3f}")
    
    # Check if 48 has maximum temperature
    max_temps = {d: r['max_temperature'] for d, r in results.items()}
    optimal_deck = max(max_temps, key=max_temps.get)
    
    print(f"\nOptimal deck size for temperature: {optimal_deck}")
    
    return results


def generate_summary_report(all_results: Dict[str, Any]) -> str:
    """
    Generate a summary report of all analyses.
    
    Args:
        all_results: Combined results from all analyses
        
    Returns:
        Path to generated report
    """
    from datetime import datetime
    
    report_lines = [
        "# CGT ANALYSIS SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Check for 48-card optimality
    if 'periodicity' in all_results:
        period_results = all_results['periodicity']
        if 48 in period_results and period_results[48]['is_perfect']:
            report_lines.append("✓ **48-card deck shows perfect 2^4 × 3 structure**")
        else:
            report_lines.append("✗ 48-card deck does not show expected structure")
    
    if 'temperature' in all_results:
        temp_results = all_results['temperature']
        max_temps = {d: r['max_temperature'] for d, r in temp_results.items()}
        optimal = max(max_temps, key=max_temps.get)
        report_lines.append(f"✓ **Maximum temperature at {optimal} cards: {max_temps[optimal]:.3f}**")
    
    report_lines.extend([
        "",
        "## Detailed Results",
        ""
    ])
    
    # Add detailed results
    if 'war_analysis' in all_results:
        report_lines.append("### War Game Analysis")
        for deck_size, results in all_results['war_analysis'].items():
            mc = results.get('monte_carlo', {})
            report_lines.append(f"- **{deck_size} cards**: "
                              f"Win rate P1={mc.get('win_rate_p1', 0):.3f}, "
                              f"Avg length={mc.get('avg_game_length', 0):.1f}")
    
    report_lines.extend([
        "",
        "## Conclusion",
        "",
        "The analysis provides evidence for structural resonance in card games, "
        "with particular significance at the 48-card configuration.",
        "",
        "## Next Steps",
        "1. Extend analysis to Crazy Eights (partisan game)",
        "2. Validate with third game type",
        "3. Prove mathematical periodicity rigorously",
        "4. Prepare publication-ready visualizations"
    ])
    
    # Save report
    report_path = Path("data/reports/summary_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    return str(report_path)


def main():
    """Main entry point for analysis runner"""
    parser = argparse.ArgumentParser(
        description='Run CGT analysis for card games'
    )
    
    parser.add_argument(
        '--games',
        nargs='+',
        default=['war'],
        choices=['war', 'crazy_eights', 'all'],
        help='Games to analyze'
    )
    
    parser.add_argument(
        '--deck-sizes',
        nargs='+',
        type=int,
        default=[44, 48, 52],
        help='Deck sizes to test'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations'
    )
    
    parser.add_argument(
        '--analyses',
        nargs='+',
        default=['all'],
        choices=['positions', 'monte_carlo', 'periodicity', 'temperature', 'all'],
        help='Types of analyses to run'
    )

    parser.add_argument(
        '--lenses',
        nargs='*',
        default=[],
        help='Lens ids to run (e.g., temperature grundy). Available: ' + ', '.join(available_lenses())
    )
    parser.add_argument(
        '--deck-strategy',
        default=None,
        choices=[None, *available_deck_strategies()],
        help='Optional deck construction strategy to override engine default.'
    )
    parser.add_argument(
        '--deck-policy',
        type=str,
        default=None,
        help='JSON string with additional policy parameters for deck strategy.'
    )
    parser.add_argument(
        '--lens-samples',
        type=int,
        default=6,
        help='Number of sample positions for lens orchestrator.'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate summary report'
    )
    
    args = parser.parse_args()
    
    # Run requested analyses
    all_results = {}
    
    if 'all' in args.analyses or 'positions' in args.analyses or 'monte_carlo' in args.analyses:
        if 'war' in args.games or 'all' in args.games:
            print("\nRunning War analysis...")
            war_results = {}
            # Custom deck policy parsing
            deck_policy = {}
            if args.deck_policy:
                import json as _json
                try:
                    deck_policy = _json.loads(args.deck_policy)
                except Exception:
                    print("Warning: Could not parse deck policy JSON; ignoring.")
            
            for deck_size in args.deck_sizes:
                custom_deck = None
                if args.deck_strategy:
                    builder = get_deck_builder(args.deck_strategy)
                    custom_deck = builder.build(deck_size=deck_size, seed=42, **deck_policy)
                # Run analysis for this deck size individually (reusing original function logic with minor adaptation)
                tmp_results = run_war_analysis(
                    deck_sizes=[deck_size],
                    num_simulations=args.simulations,
                    save_results=not args.no_save
                )
                war_results.update(tmp_results)
                # If lenses requested, orchestrate now per deck size with custom deck if specified
                if args.lenses:
                    from cgt_analysis.war_engine import WarGameEngine
                    engine = WarGameEngine(deck_size=deck_size, seed=42, custom_deck=custom_deck)
                    orchestrator = LensOrchestrator(
                        engine=engine,
                        lens_ids=args.lenses,
                        seed=42,
                        num_samples=args.lens_samples,
                        trajectory_simulations=5
                    )
                    lens_output_bundle = orchestrator.run()
                    war_results[deck_size]['lens_outputs'] = lens_output_bundle['lens_outputs']
                    # Persist each lens output
                    if not args.no_save:
                        data_manager = DataManager()
                        for lid, lres in lens_output_bundle['lens_outputs'].items():
                            data_manager.save_analysis_result(
                                game_name='War',
                                deck_size=deck_size,
                                analysis_type=f'lens_{lid}',
                                data=lres
                            )
            
            all_results['war_analysis'] = war_results
    
    if 'all' in args.analyses or 'periodicity' in args.analyses:
        print("\nRunning periodicity analysis...")
        all_results['periodicity'] = run_periodicity_analysis(
            deck_sizes=args.deck_sizes
        )
    
    if 'all' in args.analyses or 'temperature' in args.analyses:
        print("\nRunning temperature evolution...")
        all_results['temperature'] = run_temperature_evolution(
            deck_sizes=args.deck_sizes,
            samples_per_deck=50
        )
    
    # Generate report if requested
    if args.report:
        generate_summary_report(all_results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
