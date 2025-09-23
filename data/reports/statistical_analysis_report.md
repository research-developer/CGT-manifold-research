# STATISTICAL ANALYSIS REPORT
============================================================

## HYPOTHESIS TESTING RESULTS

### Game Length ANOVA
- **Null Hypothesis**: Mean game lengths are equal across deck sizes
- **Alternative Hypothesis**: At least one deck size has different mean game length
- **Test Statistic**: 1.5605
- **P-value**: 1.000000
- **Effect Size**: 0.0010
- **95% CI**: (0.0001, 0.0045)
- **Result**: Non-significant difference in game lengths (Bonferroni corrected)
- **Significant**: No

### Win Rate Balance Test
- **Null Hypothesis**: 48-card deck has 50/50 win rate
- **Alternative Hypothesis**: 48-card deck deviates from 50/50 win rate
- **Test Statistic**: 1.6000
- **P-value**: 1.000000
- **Effect Size**: 0.0400
- **95% CI**: (0.4890, 0.5508)
- **Result**: 48-card deviation: 0.0200, 44-card: 0.0010, 52-card: 0.0250 (Bonferroni corrected)
- **Significant**: No

### Temperature Variance Test
- **Null Hypothesis**: Temperature variance is equal across deck sizes
- **Alternative Hypothesis**: Temperature variance differs between deck sizes
- **Test Statistic**: 1.5605
- **P-value**: 1.000000
- **Effect Size**: 3.3797
- **95% CI**: (0.0000, 4.0556)
- **Result**: Non-significant variance differences (Bonferroni corrected)
- **Significant**: No

### Periodicity Evidence Test
- **Null Hypothesis**: No evidence of 16-card periodicity in 48-card deck
- **Alternative Hypothesis**: 48-card deck shows 16-card periodicity patterns
- **Test Statistic**: 998022.0000
- **P-value**: 1.000000
- **Effect Size**: 0.0370
- **95% CI**: (0.0296, 0.0444)
- **Result**: 48-card autocorr: -0.0091, not highest (Bonferroni corrected)
- **Significant**: No

### Grundy Consistency Test
- **Null Hypothesis**: Win rates are consistent across deck sizes (balanced games)
- **Alternative Hypothesis**: Win rates vary significantly (unbalanced games)
- **Test Statistic**: 0.0219
- **P-value**: 1.000000
- **Effect Size**: 0.0219
- **95% CI**: (0.0000, 0.0328)
- **Result**: Coefficient of variation: 0.0219 (Bonferroni corrected)
- **Significant**: No

## POWER ANALYSIS RESULTS

### effect_size_0.2
- **Statistical Power**: 1.000
- **Effect Size**: 0.200
- **Sample Size**: 1000
- **Minimum N for 80% Power**: 200
- **Interpretation**: Power = 1.000 (adequate)

### effect_size_0.5
- **Statistical Power**: 1.000
- **Effect Size**: 0.500
- **Sample Size**: 1000
- **Minimum N for 80% Power**: 40
- **Interpretation**: Power = 1.000 (adequate)

### effect_size_0.8
- **Statistical Power**: nan
- **Effect Size**: 0.800
- **Sample Size**: 1000
- **Minimum N for 80% Power**: 20
- **Interpretation**: Power = nan (inadequate)

## BOOTSTRAP CONFIDENCE INTERVALS

### win_rate_44
- **95% Bootstrap CI**: (0.4680, 0.5290)

### game_length_44
- **95% Bootstrap CI**: (99.2880, 99.8160)

### win_rate_48
- **95% Bootstrap CI**: (0.4900, 0.5510)

### game_length_48
- **95% Bootstrap CI**: (99.5220, 99.8821)

### win_rate_52
- **95% Bootstrap CI**: (0.4950, 0.5570)

### game_length_52
- **95% Bootstrap CI**: (99.6839, 99.9540)

## STATISTICAL CONCLUSIONS

- **Total tests performed**: 5
- **Significant results**: 0
- **Multiple comparison correction**: Bonferroni
- **Overall significance level**: 0.05

**No statistically significant differences found after correction.**