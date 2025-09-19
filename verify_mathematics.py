#!/usr/bin/env python3
"""
Mathematical Verification Runner

This script runs comprehensive verification of the CGT mathematical methods
to identify and report inconsistencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cgt_analysis.mathematical_verification import MathematicalVerifier
from cgt_analysis.war_engine import WarGameEngine
from cgt_analysis.base import CGTAnalyzer


def run_verification():
    """Run comprehensive mathematical verification"""
    print("MATHEMATICAL VERIFICATION OF CGT METHODS")
    print("=" * 60)
    print()
    
    verifier = MathematicalVerifier()
    
    # Test with different deck sizes
    deck_sizes = [44, 48, 52]
    all_positions = []
    
    for deck_size in deck_sizes:
        print(f"Analyzing {deck_size}-card deck...")
        engine = WarGameEngine(deck_size=deck_size, seed=42)
        analyzer = CGTAnalyzer(engine)
        
        # Create test positions as WarPosition objects first
        war_positions = {
            'A': engine.create_position_a(),
            'B': engine.create_position_b(), 
            'C': engine.create_position_c(),
            'D': engine.create_position_d(),
            'E': engine.create_position_e()
        }
        
        # Convert to CGTPosition objects properly
        positions = {}
        for name, war_pos in war_positions.items():
            cgt_pos = analyzer._convert_to_cgt_position(war_pos, 0, 2)  # Limited depth for testing
            positions[name] = cgt_pos
        
        # Verify each position
        for name, position in positions.items():
            pos_name = f"{name}_{deck_size}"
            position.position_name = pos_name  # Ensure unique naming
            
            print(f"  Verifying Position {pos_name}...")
            result = verifier.verify_position_consistency(position)
            
            if result['verification_passed']:
                print(f"    ✅ Position {pos_name} verification PASSED")
            else:
                print(f"    ❌ Position {pos_name} verification FAILED")
                for issue in result['inconsistencies_found']:
                    print(f"       - {issue}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    ⚠️  Warning: {warning}")
            
            all_positions.append(position)
    
    print()
    print("GENERATING COMPREHENSIVE REPORT...")
    print("-" * 40)
    
    # Generate full verification report
    report = verifier.generate_verification_report(all_positions)
    print(report)
    
    # Save report to file
    report_path = Path(__file__).parent.parent / "mathematical_verification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Full report saved to: {report_path}")
    print()
    
    # Test specific mathematical properties
    print("TESTING SPECIFIC MATHEMATICAL PROPERTIES")
    print("-" * 50)
    
    # Test Grundy number consistency
    print("Testing Grundy number calculations...")
    grundy_issues = 0
    
    for position in all_positions[:5]:  # Test first 5 positions
        try:
            grundy_result = verifier.grundy_calc.verify_grundy_calculation(position)
            if not grundy_result['verification_passed']:
                print(f"  ❌ Grundy verification failed for {position.position_name}")
                grundy_issues += 1
            else:
                print(f"  ✅ Grundy verification passed for {position.position_name}")
        except Exception as e:
            print(f"  ❌ Grundy calculation error for {position.position_name}: {e}")
            grundy_issues += 1
    
    # Test temperature consistency  
    print("\nTesting temperature calculations...")
    temp_issues = 0
    
    for position in all_positions[:5]:
        try:
            temp = verifier.temp_calc.compute_temperature(position)
            if temp < 0 or not math.isfinite(temp):
                print(f"  ❌ Invalid temperature {temp} for {position.position_name}")
                temp_issues += 1
            else:
                print(f"  ✅ Temperature {temp:.3f} valid for {position.position_name}")
        except Exception as e:
            print(f"  ❌ Temperature calculation error for {position.position_name}: {e}")
            temp_issues += 1
    
    print()
    print("VERIFICATION SUMMARY")
    print("-" * 30)
    print(f"Grundy calculation issues: {grundy_issues}")
    print(f"Temperature calculation issues: {temp_issues}")
    
    if grundy_issues == 0 and temp_issues == 0:
        print("✅ All core mathematical methods appear consistent!")
    else:
        print("❌ Mathematical inconsistencies detected - review required")
        return 1
    
    return 0


if __name__ == "__main__":
    import math
    sys.exit(run_verification())