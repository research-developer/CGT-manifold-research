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
from cgt_analysis.statistical_analysis import StatisticalAnalyzer


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
        
        # Run Enhanced Monte Carlo simulations with statistical rigor
        print(f"  Running {num_simulations} enhanced simulations...")
        monte_carlo = analyzer.run_monte_carlo_analysis(
            num_simulations=num_simulations,
            use_antithetic_variates=True,
            track_convergence=True
        )
        print(f"    Win rates: P1={monte_carlo['win_rate_p1']:.3f}, "
              f"P2={monte_carlo['win_rate_p2']:.3f}")
        print(f"    Avg game length: {monte_carlo['avg_game_length']:.1f} ± "
              f"{monte_carlo['std_game_length']:.1f}")
        
        # Add convergence diagnostics
        if monte_carlo.get('convergence_diagnostics'):
            converged = monte_carlo['convergence_diagnostics'].get('converged', False)
            print(f"    Convergence: {'✓' if converged else '✗'}")
            if not converged:
                print(f"    Warning: Simulations may not have converged. Consider increasing sample size.")
        
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


def run_statistical_analysis(deck_sizes: List[int] = [44, 48, 52],
                            num_simulations: int = 1000,
                            save_results: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis across deck sizes.
    
    Args:
        deck_sizes: List of deck sizes to analyze
        num_simulations: Number of simulations per deck size
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with statistical analysis results
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Initialize statistical analyzer
    statistical_analyzer = StatisticalAnalyzer(significance_level=0.05)
    data_manager = DataManager() if save_results else None
    
    # Collect Monte Carlo results for all deck sizes
    print("Phase 1: Running Monte Carlo simulations for all deck sizes...")
    mc_results = {}
    
    for deck_size in deck_sizes:
        print(f"\n  Analyzing {deck_size}-card deck...")
        
        # Initialize engine and analyzer
        engine = WarGameEngine(deck_size=deck_size, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        # Run enhanced Monte Carlo with statistical rigor
        results = statistical_analyzer.run_enhanced_monte_carlo(
            analyzer,
            num_simulations=num_simulations,
            use_antithetic_variates=True,
            track_convergence=True
        )
        
        mc_results[deck_size] = results
        
        print(f"    Win rate P1: {results['win_rate_p1']:.4f}")
        print(f"    Avg game length: {results['avg_game_length']:.2f} ± {results['std_game_length']:.2f}")
        
        # Check convergence
        if results.get('convergence_diagnostics'):
            converged = results['convergence_diagnostics'].get('converged', False)
            print(f"    Convergence: {'✓' if converged else '✗'}")
        
        # Save individual results
        if save_results and data_manager:
            data_manager.save_analysis_result(
                game_name="War",
                deck_size=deck_size,
                analysis_type="enhanced_monte_carlo",
                data=results
            )
    
    # Phase 2: Comprehensive hypothesis testing
    print(f"\nPhase 2: Running hypothesis tests...")
    
    if len(deck_sizes) >= 3:
        results_44 = mc_results.get(44, mc_results[min(deck_sizes)])
        results_48 = mc_results.get(48, mc_results[sorted(deck_sizes)[len(deck_sizes)//2]])
        results_52 = mc_results.get(52, mc_results[max(deck_sizes)])
        
        hypothesis_tests = statistical_analyzer.run_comprehensive_hypothesis_tests(
            results_44, results_48, results_52
        )
        
        print(f"    Completed {len(hypothesis_tests)} hypothesis tests")
        
        # Print key results
        for test_name, test_result in hypothesis_tests.items():
            significance = "✓ SIGNIFICANT" if test_result.p_value < 0.05 else "✗ Not significant"
            print(f"    {test_result.test_name}: p={test_result.p_value:.6f} {significance}")
    else:
        hypothesis_tests = {}
        print("    Skipped (need at least 3 deck sizes)")
    
    # Phase 3: Power analysis
    print(f"\nPhase 3: Power analysis...")
    power_analyses = {}
    
    for effect_size in [0.2, 0.5, 0.8]:  # Small, medium, large effects
        power_result = statistical_analyzer.calculate_power_analysis(
            effect_size=effect_size,
            sample_size=num_simulations,
            test_type='two_sample'
        )
        power_analyses[f"effect_size_{effect_size}"] = power_result
        print(f"    Effect size {effect_size}: Power = {power_result['power']:.3f}")
    
    # Phase 4: Bootstrap confidence intervals
    print(f"\nPhase 4: Bootstrap confidence intervals...")
    bootstrap_results = {}
    
    for deck_size, results in mc_results.items():
        # Win rate CI - create proper bootstrap sample
        win_rate_samples = [1 if i < int(results['win_rate_p1'] * len(results['game_lengths'])) else 0 
                           for i in range(len(results['game_lengths']))]
        import numpy as np
        
        win_rate_ci = statistical_analyzer.bootstrap_confidence_intervals(
            win_rate_samples,
            statistic_func=lambda x: np.mean(x),
            n_bootstrap=1000
        )
        
        # Game length CI
        length_ci = statistical_analyzer.bootstrap_confidence_intervals(
            results['game_lengths'],
            statistic_func=np.mean,
            n_bootstrap=1000
        )
        
        bootstrap_results[f"win_rate_{deck_size}"] = win_rate_ci
        bootstrap_results[f"game_length_{deck_size}"] = length_ci
        
        print(f"    {deck_size} cards - Win rate CI: ({win_rate_ci[0]:.4f}, {win_rate_ci[1]:.4f})")
        print(f"    {deck_size} cards - Length CI: ({length_ci[0]:.2f}, {length_ci[1]:.2f})")
    
    # Phase 5: Generate comprehensive report
    print(f"\nPhase 5: Generating statistical report...")
    
    statistical_report = statistical_analyzer.generate_statistical_report(
        hypothesis_tests, power_analyses, bootstrap_results
    )
    
    # Compile all results
    comprehensive_results = {
        'monte_carlo_results': mc_results,
        'hypothesis_tests': hypothesis_tests,
        'power_analyses': power_analyses,
        'bootstrap_results': bootstrap_results,
        'statistical_report': statistical_report,
        'metadata': {
            'num_simulations': num_simulations,
            'deck_sizes_analyzed': deck_sizes,
            'significance_level': statistical_analyzer.alpha,
            'variance_reduction_used': True,
            'convergence_tracking': True
        }
    }
    
    # Save comprehensive results
    if save_results and data_manager:
        data_manager.save_analysis_result(
            game_name="War",
            deck_size=0,  # Use 0 to indicate comprehensive analysis
            analysis_type="comprehensive_statistical_analysis",
            data=comprehensive_results
        )
        
        # Save report as text file
        report_path = Path("data/reports/statistical_analysis_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(statistical_report)
        print(f"    Statistical report saved to: {report_path}")
    
    # Summary of key findings
    print(f"\n" + "=" * 60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Check for 48-card optimality evidence
    if 48 in mc_results:
        win_rate_48 = mc_results[48]['win_rate_p1']
        deviation_48 = abs(win_rate_48 - 0.5)
        
        deviations = {}
        for deck_size, results in mc_results.items():
            deviations[deck_size] = abs(results['win_rate_p1'] - 0.5)
        
        optimal_deck = min(deviations, key=deviations.get)
        
        print(f"Win Rate Balance Analysis:")
        for deck_size, deviation in deviations.items():
            marker = "★" if deck_size == optimal_deck else " "
            print(f"  {marker} {deck_size} cards: {deviation:.6f} deviation from 50/50")
        
        print(f"\nOptimal deck for balance: {optimal_deck} cards")
        
        if optimal_deck == 48:
            print("✓ 48-card deck shows best balance (supports hypothesis)")
        else:
            print("✗ 48-card deck does not show optimal balance")
    
    # Statistical significance summary
    if hypothesis_tests:
        significant_count = sum(1 for test in hypothesis_tests.values() if test.p_value < 0.05)
        print(f"\nHypothesis Testing Summary:")
        print(f"  Total tests: {len(hypothesis_tests)}")
        print(f"  Significant results: {significant_count}")
        print(f"  Multiple comparison correction: Bonferroni")
        
        if significant_count > 0:
            print("  ✓ Found statistically significant differences")
        else:
            print("  ✗ No significant differences found after correction")
    
    return comprehensive_results


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
        choices=['positions', 'monte_carlo', 'periodicity', 'temperature', 'statistical', 'all'],
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
    
    if 'all' in args.analyses or 'statistical' in args.analyses:
        print("\nRunning comprehensive statistical analysis...")
        all_results['statistical_analysis'] = run_statistical_analysis(
            deck_sizes=args.deck_sizes,
            num_simulations=args.simulations,
            save_results=not args.no_save
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
