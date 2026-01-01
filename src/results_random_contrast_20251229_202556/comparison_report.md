# Random Contrast Task Encoding Variants: Comparison Report

**Generated:** 2025-12-29 20:25:56

**Total variants tested:** 12


**Successful:** 12, **Failed:** 0


## Main Results Table

| Variant | R@5 | R@10 | MRR | Prob Ratio | Conv. Epoch |
|---------|-----|------|-----|------------|-------------|
| baseline_enhanced | 0.657±0.02 | 0.788 | 0.466±0.01 | 17034.9 | 55 |
| triple_contrast_standard | 0.644±0.03 | 0.792 | 0.454±0.01 | 3876.6 | 54 |
| random_augmented_lambda0.5 | 0.640±0.05 | 0.786 | 0.461±0.01 | 58731.5 | 52 |
| baseline_standard | 0.635±0.02 | 0.793 | 0.459±0.01 | 7805554.4 | 48 |
| random_augmented_lambda1.0 | 0.627±0.03 | 0.791 | 0.455±0.01 | 16536.3 | 50 |
| pos_vs_random_standard | 0.620±0.02 | 0.804 | 0.462±0.01 | 99912.3 | 53 |
| pos_vs_random_enhanced | 0.619±0.04 | 0.789 | 0.457±0.01 | 11989.6 | 56 |
| enhanced+random_augmented | 0.619±0.02 | 0.791 | 0.454±0.01 | 7960.0 | 53 |
| random_augmented_50random | 0.606±0.04 | 0.783 | 0.448±0.01 | 6647.7 | 55 |
| enhanced+attention+triple | 0.604±0.02 | 0.793 | 0.459±0.01 | 1870497.3 | 57 |
| triple_contrast_50random | 0.600±0.07 | 0.797 | 0.459±0.02 | 389992.1 | 60 |
| triple_contrast_enhanced | 0.588±0.03 | 0.776 | 0.456±0.02 | 132431832.9 | 51 |

## Task Encoder Analysis

| Encoder Type | Mean R@5 | Mean MRR | Count |
|--------------|----------|----------|-------|
| pos_vs_random | 0.620 | 0.460 | 2 |
| random_augmented | 0.623 | 0.455 | 4 |
| standard | 0.646 | 0.462 | 2 |
| triple | 0.609 | 0.457 | 4 |

## Key Insights

1. **Best performing variant:** baseline_enhanced (R@5=0.657)
2. **Improvement over baseline:** +0.0%

