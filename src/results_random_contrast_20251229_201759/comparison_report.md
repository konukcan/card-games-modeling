# Random Contrast Task Encoding Variants: Comparison Report

**Generated:** 2025-12-29 20:17:59

**Total variants tested:** 12


**Successful:** 12, **Failed:** 0


## Main Results Table

| Variant | R@5 | R@10 | MRR | Prob Ratio | Conv. Epoch |
|---------|-----|------|-----|------------|-------------|
| random_augmented_lambda0.5 | 0.092±0.05 | 0.173 | 0.074±0.02 | 0.9 | 99 |
| enhanced+attention+triple | 0.090±0.14 | 0.109 | 0.100±0.11 | 2.1 | 99 |
| enhanced+random_augmented | 0.083±0.04 | 0.145 | 0.081±0.03 | 12.4 | 99 |
| triple_contrast_standard | 0.083±0.11 | 0.118 | 0.070±0.04 | 1.5 | 99 |
| pos_vs_random_enhanced | 0.078±0.04 | 0.143 | 0.073±0.02 | 1.7 | 99 |
| random_augmented_50random | 0.077±0.04 | 0.145 | 0.069±0.02 | 0.8 | 99 |
| baseline_standard | 0.071±0.03 | 0.143 | 0.081±0.02 | 1.7 | 99 |
| random_augmented_lambda1.0 | 0.066±0.03 | 0.147 | 0.076±0.03 | 2.2 | 99 |
| baseline_enhanced | 0.062±0.02 | 0.145 | 0.063±0.02 | 0.9 | 99 |
| pos_vs_random_standard | 0.055±0.03 | 0.128 | 0.064±0.02 | 2.1 | 99 |
| triple_contrast_50random | 0.021±0.02 | 0.041 | 0.045±0.02 | 2.4 | 99 |
| triple_contrast_enhanced | 0.011±0.02 | 0.028 | 0.039±0.01 | 0.7 | 99 |

## Task Encoder Analysis

| Encoder Type | Mean R@5 | Mean MRR | Count |
|--------------|----------|----------|-------|
| pos_vs_random | 0.066 | 0.068 | 2 |
| random_augmented | 0.079 | 0.075 | 4 |
| standard | 0.067 | 0.072 | 2 |
| triple | 0.051 | 0.063 | 4 |

## Key Insights

1. **Best performing variant:** random_augmented_lambda0.5 (R@5=0.092)
2. **Improvement over baseline:** +28.8%

