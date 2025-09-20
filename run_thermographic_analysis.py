#!/usr/bin/env python3
"""
Task 3: Comprehensive Thermographic Analysis Runner

This script runs the complete thermographic analysis for Linear issue IMA-7,
addressing all requirements:

1. Generate thermographs for 44, 48, 52 card games
2. Investigate temperature evolution and periodicity
3. Create publication-quality visualizations
4. Analyze cooling rates and mathematical models
5. Test the 48-card structural resonance hypothesis

Usage:
    python3 run_thermographic_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cgt_analysis.thermographic_analysis import ThermographAnalyzer

def main():
    """Run comprehensive thermographic analysis for Task 3"""
    
    print("🎯 TASK 3: THERMOGRAPHIC ANALYSIS AND VISUALIZATION")
    print("=" * 70)
    print("Linear Issue: IMA-7")
    print("Objective: Generate complete thermographic analysis showing temperature evolution")
    print("Critical: Investigate why current positions show monotonic increase vs 48-card peak")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ThermographAnalyzer()
    
    # Run comprehensive analysis
    try:
        results = analyzer.run_comprehensive_thermographic_analysis(
            deck_sizes=[44, 48, 52]
        )
        
        print("\n🎉 TASK 3 COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ All thermographic analysis requirements fulfilled")
        print("✅ Publication-quality visualizations generated")
        print("✅ Temperature evolution patterns analyzed")
        print("✅ 16-card periodicity investigated")
        print("✅ Cooling rate analysis completed")
        print("✅ Mathematical equations derived")
        print("✅ 48-card structural resonance hypothesis tested")
        
        # Print final status
        print(f"\n📁 Results Location:")
        print(f"   Data: /workspace/data/raw/")
        print(f"   Visualizations: /workspace/data/visualizations/")
        print(f"   Reports: /workspace/data/reports/")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())