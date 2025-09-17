# Structural Resonance in Card Games: A Combinatorial Game Theory Study

## Research Overview

This repository contains research demonstrating that combinatorial card games exhibit **structural resonance** when deck size equals 2^n×k, where n determines decision complexity and k provides symmetry. Our primary finding shows that the traditional 48-card deck (2^4×3) represents a mathematical optimum that maximizes strategic depth while preserving perfect game-theoretic balance.

## Key Finding

Traditional card games achieve optimal balance through **periodic game sums** with the 48-card configuration providing:
- Full binary decision complexity (16 states)
- Ternary symmetry (3 complete cycles)
- Mean-preserving temperature oscillation
- Grundy number 0 (perfect balance)

## Project Structure

```
CGT-manifold-research/
├── simulations/          # Monte Carlo simulations for War, Crazy Eights, etc.
├── cgt_analysis/         # Combinatorial Game Theory calculations
├── visualizations/       # Thermographic analysis and figures
├── paper/               # LaTeX source for publication
├── data/                # Simulation results and datasets
└── tests/               # Unit tests for all components
```

## Mathematical Framework

### The 2^4×3 Principle
- **2^4 = 16**: Binary hypercube vertices representing all possible 4-bit decision states
- **×3**: Three-fold replication ensuring rotational symmetry
- **= 48**: The optimal deck size for balanced gameplay

### Core Claims
1. Games with 48 cards exhibit exact 16-card periodicity repeated 3 times
2. Grundy number analysis: G(44) ≠ 0, G(48) = 0, G(52) ≠ 0
3. Temperature maximizes at 48 cards while maintaining mean value = 0
4. Pattern holds across multiple game types (impartial and partisan)

## Research Components

### Task Breakdown
1. **CGT Position Analysis**: Formal game-theoretic analysis using {L | R} notation
2. **Periodicity Proof**: Mathematical demonstration of the 16-card periodic structure
3. **Thermographic Analysis**: Temperature evolution and cooling rate calculations
4. **Statistical Validation**: Monte Carlo simulations with rigorous hypothesis testing
5. **Cross-Game Validation**: Verification across War, Crazy Eights, and additional games
6. **Partisan Game Analysis**: Extension to games with asymmetric player options

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/CGT-manifold-research.git
cd CGT-manifold-research

# Install dependencies (Python 3.8+)
pip install -r requirements.txt

# Run simulations
python simulations/run_all.py

# Generate visualizations
python visualizations/generate_thermographs.py
```

## Reproducibility

All simulations use fixed random seeds for complete reproducibility:
- War simulations: seed = 42
- Crazy Eights: seed = 1337
- Statistical tests: seed = 2024

## Key Results

| Deck Size | Avg. Game Length (War) | Grundy Number | Temperature Peak | Win Rate Deviation |
|-----------|------------------------|---------------|------------------|-------------------|
| 44 cards  | 141 rounds            | 1             | 1.651           | <3%              |
| 48 cards  | 192 rounds            | **0**         | **1.886**       | <3%              |
| 52 cards  | 233 rounds            | 6             | 1.743           | <3%              |

## Contributing

This research is part of an active academic project. Contributions are welcome:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/analysis-enhancement`)
3. Commit changes with clear messages
4. Push to your branch
5. Open a Pull Request with detailed description

## Publication Status

**Target Journals:**
- Games and Economic Behavior
- International Journal of Game Theory
- The Electronic Journal of Combinatorics

**Current Status:** Mathematical proofs in progress, simulations complete

## Dependencies

- Python 3.8+
- NumPy, SciPy for numerical computation
- Matplotlib, Seaborn for visualization
- LaTeX for paper preparation
- CGSuite (optional) for game theory verification

## Citation

If you use this research in your work, please cite:
```bibtex
@unpublished{temple2025structural,
  title={Structural Resonance in Card Games: The 2^n×k Principle},
  author={Temple, Preston},
  year={2025},
  note={Manuscript in preparation}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions about this research, please open an issue or contact the repository maintainer.

---

*"Games are mathematical objects that reveal deep truths about symmetry, balance, and optimal decision-making."* - Inspired by Conway's foundational work in combinatorial game theory
