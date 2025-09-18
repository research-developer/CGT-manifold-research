# CGT Analysis Framework Documentation

## Project Structure After Refactoring

```
CGT-manifold-research/
├── cgt_analysis/              # Core CGT framework (DO NOT MODIFY WITHOUT REVIEW)
│   ├── __init__.py           # Module exports
│   ├── base.py               # Abstract base classes and interfaces ⭐
│   ├── cgt_position.py       # CGT position representation
│   ├── grundy_numbers.py     # Grundy number calculations
│   ├── temperature_analysis.py # Temperature calculations
│   ├── thermographic_analysis.py # Thermographic analysis
│   ├── war_engine.py         # War game implementation ✓
│   └── crazy_eights_engine.py # Crazy Eights skeleton (TODO: Task 6)
│
├── scripts/                   # Analysis scripts
│   ├── analysis/             # Core analysis scripts
│   │   ├── analyze_war_positions.py  # Original Task 1 analysis
│   │   └── simplified_analysis.py    # Simplified version
│   └── visualization/        # Visualization scripts
│       └── create_visualizations.py  # Generate charts
│
├── data/                     # Data storage
│   ├── raw/                  # Raw analysis results (JSON)
│   ├── processed/            # Processed data
│   ├── visualizations/       # Generated charts
│   └── reports/              # Analysis reports
│
├── tests/                    # Unit tests
│   └── test_cgt_framework.py # Framework tests
│
├── run_analysis.py          # Main analysis runner ⭐
└── requirements.txt         # Python dependencies
```

## For Agents Working on Future Tasks

### Key Interfaces You MUST Follow

#### 1. Implementing a New Game (Tasks 5, 6)

All games MUST inherit from the base classes in `cgt_analysis/base.py`:

```python
from cgt_analysis.base import GameEngine, GameState, Player

class YourGameState(GameState):
    # Must implement ALL abstract methods:
    # - is_terminal()
    # - get_winner()
    # - get_player_to_move()
    # - get_state_hash()
    # - get_game_value()
    # - to_dict()

class YourGameEngine(GameEngine):
    # Must implement ALL abstract methods:
    # - create_initial_state()
    # - get_possible_moves()
    # - apply_move()
    # - get_next_states()
    # - simulate_game()
    # - evaluate_position()
    # - game_name (property)
    # - is_impartial (property)
```

See `war_engine.py` for a complete example implementation.

#### 2. Running Analysis (All Tasks)

Use the main runner for standardized analysis:

```bash
# Run complete analysis for War
python run_analysis.py --games war --deck-sizes 44 48 52 --analyses all

# Run only periodicity analysis
python run_analysis.py --analyses periodicity --deck-sizes 44 48 52

# Run with more simulations
python run_analysis.py --games war --simulations 5000

# Generate summary report
python run_analysis.py --games war --report
```

#### 3. Using the CGT Analyzer

```python
from cgt_analysis.base import CGTAnalyzer
from cgt_analysis.war_engine import WarGameEngine

# Initialize engine and analyzer
engine = WarGameEngine(deck_size=48, seed=42)
analyzer = CGTAnalyzer(engine)

# Analyze a position
position = engine.create_position_a()
analysis = analyzer.analyze_position(position, max_depth=3)

# Run Monte Carlo simulations
results = analyzer.run_monte_carlo_analysis(num_simulations=1000)
```

#### 4. Data Management

Use the DataManager for consistent data storage:

```python
from cgt_analysis.base import DataManager

dm = DataManager()

# Save results
dm.save_analysis_result(
    game_name="War",
    deck_size=48,
    analysis_type="cgt_analysis",
    data=your_results_dict
)

# Load results
results = dm.load_analysis_results(
    game_name="War",
    deck_size=48
)
```

### Task-Specific Instructions

#### Task 2: Mathematical Proof of 16-Card Periodicity
- Use the periodicity analysis in `run_analysis.py` as a starting point
- Add rigorous mathematical proofs to `cgt_analysis/periodicity.py` (create new file)
- Implement group theory analysis showing C₃ × Z₂⁴ structure
- Create visualization of the hypercube structure

#### Task 3: Thermographic Analysis
- Extend `cgt_analysis/thermographic_analysis.py`
- Use the temperature evolution analysis in `run_analysis.py`
- Generate publication-quality thermographs
- Save to `data/visualizations/` with descriptive names

#### Task 4: Statistical Analysis
- Monte Carlo framework already in `CGTAnalyzer.run_monte_carlo_analysis()`
- Add hypothesis testing to `run_analysis.py`
- Use scipy.stats for statistical tests
- Save p-values and confidence intervals with results

#### Task 5: Third Game Validation
1. Create new file: `cgt_analysis/[game_name]_engine.py`
2. Follow the pattern in `war_engine.py` exactly
3. Add game to `run_analysis.py` main function
4. Run same analyses as War for comparison

#### Task 6: Partisan CGT Analysis (Crazy Eights)
- Complete the skeleton in `cgt_analysis/crazy_eights_engine.py`
- Pay special attention to partisan game differences
- Implement proper {L | R} notation with different moves for each player
- Add partisan-specific analysis methods

#### Task 7: Paper Writing
- Use data from `data/processed/` and `data/reports/`
- LaTeX files go in `paper/` directory
- Reference visualizations from `data/visualizations/`
- Use the DataManager to load all results for integration

#### Task 8: Final Review
- Run all tests: `pytest tests/ -v`
- Run full analysis: `python run_analysis.py --games all --analyses all --report`
- Check all results in `data/reports/summary_report.md`
- Verify reproducibility with fixed seeds

### Important Data Points from Task 1

Key results to build upon:

1. **Grundy Numbers** (Position A):
   - 44 cards: G = 0
   - 48 cards: G = 1 
   - 52 cards: G = 2

2. **Temperature Patterns**:
   - Position A shows increasing temperature with deck size
   - Positions B-E show stable temperatures
   - Need to investigate why 48 doesn't show peak temperature

3. **Mean Value Preservation**:
   - Successfully demonstrated across all positions
   - Near-zero mean values for balanced positions

4. **Files to Reference**:
   - `data/simplified_analysis_results.json` - Numerical results
   - `data/Simplified_CGT_Analysis_Report.md` - Detailed analysis
   - `data/*.png` - Visualizations

### Testing Your Code

Always write tests for new functionality:

```python
# In tests/test_your_game.py
import pytest
from cgt_analysis.your_game_engine import YourGameEngine

def test_your_game_initialization():
    engine = YourGameEngine(deck_size=48)
    assert engine.deck_size == 48
    assert engine.game_name == "YourGame"

# Run with: pytest tests/test_your_game.py -v
```

### Collaboration Guidelines

1. **DO NOT MODIFY** core framework files without discussion:
   - `cgt_analysis/base.py`
   - `cgt_analysis/cgt_position.py`
   - `cgt_analysis/grundy_numbers.py`

2. **ALWAYS USE** the established interfaces - don't create parallel implementations

3. **SAVE ALL RESULTS** using the DataManager for consistency

4. **DOCUMENT YOUR WORK** with clear comments and docstrings

5. **TEST EVERYTHING** - add tests for any new functionality

### Common Pitfalls to Avoid

❌ Don't create analysis scripts in the root directory
❌ Don't modify the base classes without approval
❌ Don't use different data formats - stick to JSON
❌ Don't hardcode paths - use relative paths
❌ Don't forget to set random seeds for reproducibility

### Getting Help

- Review `war_engine.py` for implementation patterns
- Check `test_cgt_framework.py` for usage examples
- Run `python run_analysis.py --help` for options
- Look at Task 1 deliverables in `DELIVERABLES_CHECKLIST.md`

## Next Priority Actions

Based on Task 1 results, the following need investigation:

1. **Temperature Maximum**: Task 1 found monotonic increase (52 > 48 > 44) instead of peak at 48. This needs deeper analysis with more complex positions.

2. **Periodicity Proof**: The mathematical proof of 16-card periodicity (Task 2) is now critical to establish the theoretical foundation.

3. **Crazy Eights Implementation**: Complete the partisan game analysis (Task 6) to show the principle works for both game types.

Remember: The goal is to prove that 48 cards creates "structural resonance" through the 2^4×3 pattern!
