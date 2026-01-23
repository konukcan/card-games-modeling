# Experimental Design: Warm-Start Pretraining for Card Game Rule Learning

**Version:** 1.0
**Date:** December 2024
**Author:** Methodologist Agent

---

## 1. Executive Summary

This document specifies a rigorous experimental methodology to test whether **warm-start pretraining** on simpler card game rules (poker, blackjack, solitaire-inspired) improves the DreamCoder system's ability to learn more complex experimental rules.

### Core Hypothesis

> H1: A recognition model pretrained on 44 "familiar" card game rules will solve more catalogue rules (and solve them faster) than a model trained from scratch, even when controlling for total compute budget.

### Rule Sets

| Set | Count | Source | Complexity | Purpose |
|-----|-------|--------|------------|---------|
| Pretraining Rules | 44 | `pretraining_rules.py` | Level 1-2 (simpler) | Warm-start training |
| Catalogue Rules | 45 | `catalogue.py` | Level 1-4 (harder) | Main evaluation |

---

## 2. Experimental Conditions

### 2.1 Primary Conditions

| Condition | Abbrev. | Description |
|-----------|---------|-------------|
| **Control: Cold Start** | `COLD` | Train directly on catalogue rules with no prior exposure |
| **Treatment: Warm Start** | `WARM` | Pretrain on pretraining rules, then fine-tune on catalogue rules |

### 2.2 Secondary Conditions (Ablations)

| Condition | Abbrev. | Description |
|-----------|---------|-------------|
| **Oracle Pretraining** | `WARM-O` | Warm-start using oracle programs (no enumeration during pretraining) |
| **Shallow Pretraining** | `WARM-S` | Warm-start with reduced pretraining budget (1/2 iterations) |
| **Deep Pretraining** | `WARM-D` | Warm-start with extended pretraining budget (2x iterations) |
| **Library Only** | `LIB-ONLY` | Transfer learned library but reset recognition model |
| **Recognition Only** | `REC-ONLY` | Transfer recognition model but reset library |

### 2.3 Baseline Conditions

| Condition | Abbrev. | Description |
|-----------|---------|-------------|
| **Extended Cold Start** | `COLD+` | Cold start with total compute equal to WARM (pretraining + main) |
| **Random Pretraining** | `RANDOM` | Pretrain on random tasks (control for general training effects) |

---

## 3. Independent Variables

### 3.1 Pretraining Configuration

| Variable | Values | Rationale |
|----------|--------|-----------|
| **Pretraining Iterations** | {0, 5, 10, 15, 20} | Dose-response curve for pretraining benefit |
| **Pretraining Budget per Iteration** | {100K, 200K, 300K} | Search effort during pretraining |
| **Pretraining Max Depth** | {8, 9, 10} | Program complexity during pretraining |
| **Pretraining Method** | {enumeration, oracle} | Whether to enumerate or use ground-truth programs |

### 3.2 Main Training Configuration

| Variable | Values | Rationale |
|----------|--------|-----------|
| **Main Iterations** | {10, 15, 20} | Training duration on catalogue rules |
| **Main Budget per Iteration** | {200K, 300K, 500K} | Search effort during main training |
| **Recognition Blend Factor** | {0.3, 0.5, 0.7} | Neural guidance strength |

### 3.3 Architecture Parameters (Held Constant)

| Parameter | Value | Source |
|-----------|-------|--------|
| Hidden Dimension | 128 | `neural_recognition.py` default |
| Learning Rate | 1e-3 | Standard |
| Max Examples | 20 | Per task |
| Max Cards | 8 | Hand size |
| N Workers | 4 | Parallelism |

---

## 4. Dependent Variables

### 4.1 Primary Metrics (Solve Performance)

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Solve Rate** | Fraction of catalogue rules solved | `n_solved / 45` |
| **Solve Rate by Level** | Fraction solved per complexity level | Per-level breakdown |
| **Solve Rate by Family** | Fraction solved per rule family | Per-family breakdown |
| **Time to First Solution** | Wall-clock time until first solution per rule | Seconds |
| **Iteration to First Solution** | Training iteration when first solved | Integer |

### 4.2 Search Efficiency Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Programs Enumerated per Solution** | Total programs tried before finding solution | Count |
| **Programs per Second** | Enumeration throughput | Count/sec |
| **Search Depth at Solution** | Program AST depth of solution | Integer |
| **Search Effort Ratio** | `(programs_tried / theoretical_minimum)` | Ratio |

### 4.3 Library Learning Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Abstractions Discovered** | Number of invented primitives in library | Count |
| **Abstraction Reuse Rate** | Avg. times each invention is used | Count |
| **Library Size Growth** | `final_library / initial_library` | Ratio |
| **Compression Gain** | Bits saved by library learning | Float |

### 4.4 Recognition Model Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Training Loss** | Cross-entropy loss on solved tasks | Float |
| **Prediction Accuracy** | Hit rate for used primitives in top-k | `hits@k / total` |
| **Prediction Entropy** | Uncertainty in primitive predictions | Bits |
| **Embedding Similarity** | Cosine similarity of related task embeddings | [-1, 1] |

### 4.5 Transfer Learning Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Transfer Efficiency** | `(solve_rate_WARM - solve_rate_COLD) / pretraining_cost` | Ratio |
| **Negative Transfer Rate** | Rules solved by COLD but not WARM | Count |
| **Positive Transfer Rate** | Rules solved by WARM but not COLD | Count |
| **Transfer Benefit by Family** | Per-family difference in solve rates | Dict |

---

## 5. Experimental Protocol

### 5.1 Phase 1: Pretraining (Treatment Conditions Only)

```
FOR warm-start conditions:
    1. Initialize:
       - Fresh grammar from primitives.py
       - Fresh NeuralRecognitionModel
       - Empty frontier dict

    2. Pretraining Loop (N_PRETRAIN iterations):
       a. Task Selection:
          - Use all 44 pretraining rules
          - Generate 20 examples per rule

       b. Enumeration Phase:
          - Budget: PRETRAIN_BUDGET programs
          - Max depth: PRETRAIN_DEPTH
          - Use recognition-guided search if iteration > 1

       c. Compression Phase:
          - Extract abstractions from solutions
          - Add to library if reused

       d. Recognition Training:
          - Train on solved tasks
          - PRETRAIN_EPOCHS per iteration

       e. Dreaming Phase (optional):
          - Generate synthetic tasks
          - Train recognition on dreams

       f. Checkpoint:
          - Save model weights
          - Save library state
          - Log all metrics

    3. Output:
       - Pretrained recognition model
       - Expanded library (if any inventions)
       - Pretraining metrics log
```

### 5.2 Phase 2: Main Training (All Conditions)

```
FOR all conditions:
    1. Initialize:
       - Grammar: base primitives + pretraining inventions (WARM) or base primitives only (COLD)
       - Recognition: pretrained (WARM) or fresh (COLD)
       - Empty frontier dict for catalogue rules

    2. Main Training Loop (N_MAIN iterations):
       a. Task Selection:
          - All 45 catalogue rules
          - Generate 20 examples per rule

       b. Enumeration Phase:
          - Budget: MAIN_BUDGET programs
          - Max depth: MAIN_DEPTH
          - Use recognition-guided search with blend_factor

       c. Compression Phase:
          - Extract abstractions from solutions
          - Add to library if reused

       d. Recognition Training:
          - Train on ALL solved tasks (including pretraining solutions if WARM)
          - MAIN_EPOCHS per iteration

       e. Dreaming Phase:
          - Generate DREAM_COUNT synthetic tasks
          - Train recognition on dreams

       f. Checkpoint:
          - Save model weights
          - Save library state
          - Log all metrics

    3. Output:
       - Final solve rate
       - Per-rule solution times
       - Recognition model state
       - Library state
```

### 5.3 Compute Budget Fairness

**Critical Design Decision:** To ensure fair comparison, we must equalize total compute.

#### Option A: Time-Based Equalization
```
Total Time Budget = T hours

COLD:   T hours on catalogue rules
WARM:   T/3 hours pretraining + 2T/3 hours on catalogue rules
COLD+:  T hours on catalogue rules (same as COLD, for reference)
```

#### Option B: Enumeration-Based Equalization
```
Total Enumeration Budget = B programs

COLD:   B programs on catalogue rules
WARM:   B/3 programs pretraining + 2B/3 programs on catalogue rules
COLD+:  B programs on catalogue rules
```

#### Option C: Iteration-Based Equalization (Recommended)
```
Total Iterations = I

COLD:   I iterations on catalogue rules
WARM:   I/2 iterations pretraining + I/2 iterations on catalogue rules
COLD+:  I iterations on catalogue rules

Each iteration has same enumeration budget B.
```

**Rationale for Option C:**
- Most interpretable
- Matches how DreamCoder naturally operates
- Easy to implement checkpointing
- Allows iteration-by-iteration comparison

### 5.4 Statistical Design

#### Sample Size
- **Runs per condition:** 5 (with different random seeds)
- **Total conditions:** 9 (primary + ablations + baselines)
- **Total runs:** 45

#### Random Seeds
```python
SEEDS = [42, 123, 456, 789, 1010]
```

#### Stratification
- All conditions use identical:
  - Random seeds for hand generation
  - Example generation procedures
  - Evaluation functions

---

## 6. Fair Comparison Framework

### 6.1 What Makes a Fair Comparison?

| Aspect | Fairness Criterion |
|--------|-------------------|
| **Compute** | Total enumeration budget equal across conditions |
| **Time** | Wall-clock time reported but not equalized (hardware-dependent) |
| **Information** | Same rule definitions and evaluation functions |
| **Initialization** | Same grammar primitives (57 from primitives.py) |

### 6.2 Accounting for Pretraining Compute

The key question: **Does the benefit of pretraining exceed its cost?**

We measure:
```
Efficiency = (solve_rate_WARM - solve_rate_COLD) / (pretrain_cost / total_cost)

Where:
  pretrain_cost = pretraining iterations × budget
  total_cost = pretraining + main training compute
```

If `Efficiency > 1`, pretraining is beneficial even accounting for its cost.

### 6.3 Transfer Learning Analysis

#### Positive Transfer Analysis
```
For each catalogue rule R:
    - Was R similar to any pretraining rule? (measured by primitives overlap)
    - Was R solved faster in WARM than COLD?
    - Did WARM use inventions from pretraining?
```

#### Negative Transfer Analysis
```
For each catalogue rule R:
    - Was R solved by COLD but not WARM?
    - Did pretraining "crowd out" useful primitives?
    - Did pretraining inventions hurt compression?
```

---

## 7. Baselines and Comparisons

### 7.1 Primary Comparison: WARM vs COLD

| Hypothesis | Test | Significant if |
|------------|------|----------------|
| WARM has higher solve rate | Two-proportion z-test | p < 0.05 |
| WARM solves faster | Mann-Whitney U test on times | p < 0.05 |
| WARM uses fewer programs | Mann-Whitney U test on counts | p < 0.05 |

### 7.2 Compute-Controlled Comparison: WARM vs COLD+

```
COLD+: Same total compute as WARM, all on catalogue rules

If WARM > COLD+: Pretraining structure helps, not just extra compute
If WARM = COLD+: Pretraining benefit is just from extra training
If WARM < COLD+: Pretraining is actually harmful (negative transfer)
```

### 7.3 Component Ablations

| Comparison | What it Tests |
|------------|---------------|
| WARM vs LIB-ONLY | Value of pretrained recognition model |
| WARM vs REC-ONLY | Value of learned library abstractions |
| WARM vs WARM-O | Value of enumeration (vs oracle programs) |
| WARM vs RANDOM | Specificity of pretraining (vs generic training) |

---

## 8. Analysis Plan

### 8.1 Primary Analyses

#### Analysis 1: Overall Solve Rate Comparison
```python
# Compare solve rates across conditions
solve_rates = {condition: solved_count / 45 for condition in conditions}

# Statistical test: Chi-squared test for proportions
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Effect size: Cohen's h
from statsmodels.stats.proportion import proportions_chisquare
```

#### Analysis 2: Time to Solution
```python
# For rules solved by both conditions, compare solution times
from scipy.stats import mannwhitneyu

cold_times = [time for rule, time in cold_solutions.items() if rule in warm_solutions]
warm_times = [time for rule, time in warm_solutions.items() if rule in cold_solutions]

U, p = mannwhitneyu(cold_times, warm_times, alternative='greater')
```

#### Analysis 3: Learning Curves
```python
# Track cumulative solve rate over iterations
for iteration in range(max_iterations):
    cold_curve[iteration] = count_solved_by_iteration(cold_results, iteration)
    warm_curve[iteration] = count_solved_by_iteration(warm_results, iteration)

# Compare areas under learning curves (AUC)
cold_auc = np.trapz(cold_curve)
warm_auc = np.trapz(warm_curve)
```

### 8.2 Transfer Analysis

#### Analysis 4: Transfer Matrix
```python
# Create similarity matrix between pretraining and catalogue rules
similarity_matrix = np.zeros((len(pretrain_rules), len(catalogue_rules)))

for i, p_rule in enumerate(pretrain_rules):
    for j, c_rule in enumerate(catalogue_rules):
        similarity_matrix[i, j] = primitive_overlap(p_rule, c_rule)

# Correlate with transfer benefit
transfer_benefit = warm_solve_times - cold_solve_times
max_similarity = similarity_matrix.max(axis=0)

correlation = np.corrcoef(max_similarity, transfer_benefit)[0, 1]
```

#### Analysis 5: Which Rules Benefit Most?
```python
# Group catalogue rules by family
families = ['LOCAL', 'COUNT', 'AP', 'SCORE', 'HIER', 'LANG', 'PAL',
            'ALTCLR', 'COPY', 'SHIFT', 'ADJ', 'PARITY', 'CENTER']

for family in families:
    family_rules = [r for r in catalogue_rules if r.family == family]
    cold_rate = sum(r in cold_solved for r in family_rules) / len(family_rules)
    warm_rate = sum(r in warm_solved for r in family_rules) / len(family_rules)
    benefit = warm_rate - cold_rate
    print(f"{family}: COLD={cold_rate:.2f}, WARM={warm_rate:.2f}, Δ={benefit:+.2f}")
```

### 8.3 Visualization Plan

#### Figure 1: Learning Curves
```
- X-axis: Iteration number
- Y-axis: Cumulative solve rate
- Lines: COLD, WARM, COLD+ (with confidence bands)
- Annotation: Pretraining phase boundary for WARM
```

#### Figure 2: Per-Rule Solution Times (Heatmap)
```
- Rows: Catalogue rules (sorted by family)
- Columns: Conditions
- Color: Time to solution (log scale) or "unsolved"
```

#### Figure 3: Transfer Benefit vs Rule Similarity
```
- X-axis: Maximum similarity to any pretraining rule
- Y-axis: Time speedup (COLD - WARM)
- Points: Individual catalogue rules
- Color: Rule family
```

#### Figure 4: Library Evolution
```
- X-axis: Iteration
- Y-axis: Library size (number of primitives + inventions)
- Lines: COLD, WARM
- Annotation: Key inventions discovered
```

#### Figure 5: Recognition Model Predictions
```
- For select tasks, show:
  - Top-10 predicted primitives
  - Actual primitives used in solution
  - Comparison COLD vs WARM predictions
```

---

## 9. Expected Outcomes and Predictions

### 9.1 Primary Predictions

| Prediction | Basis | Measurement |
|------------|-------|-------------|
| WARM > COLD in solve rate | Transfer from familiar patterns | χ² test |
| WARM faster than COLD on similar rules | Direct skill transfer | Correlation analysis |
| Pretraining inventions reused | Shared structure | Invention usage counts |
| Recognition accuracy higher in WARM | More training signal | Hit rate comparison |

### 9.2 Secondary Predictions

| Prediction | Basis | Measurement |
|------------|-------|-------------|
| PAL family benefits most | Palindrome patterns in pretraining | Per-family analysis |
| HIER family benefits moderately | Halves patterns in pretraining | Per-family analysis |
| LANG family benefits least | Bracket matching unique to catalogue | Per-family analysis |

### 9.3 Null Hypothesis Scenarios

| Scenario | Interpretation |
|----------|---------------|
| WARM = COLD | Pretraining has no effect (rules too different) |
| WARM < COLD | Negative transfer (pretraining interferes) |
| WARM = COLD+ | Benefit is from extra training, not transfer |

---

## 10. Implementation Details

### 10.1 Code Structure

```
card-games-modelling/
├── src/
│   ├── experiments/
│   │   ├── run_warmstart_experiment.py      # Main experiment runner
│   │   ├── analyze_warmstart_results.py     # Analysis scripts
│   │   └── generate_warmstart_report.py     # HTML report generator
│   └── ...
├── results/
│   └── warmstart_experiment/
│       ├── run_001_COLD_seed42/
│       ├── run_002_WARM_seed42/
│       └── ...
└── docs/
    └── experimental_design_warmstart_pretraining.md  # This document
```

### 10.2 Configuration Template

```python
@dataclass
class WarmStartExperimentConfig:
    # Identification
    condition: str  # 'COLD', 'WARM', 'COLD+', etc.
    seed: int
    run_id: str

    # Pretraining (ignored for COLD)
    pretrain_iterations: int = 10
    pretrain_budget: int = 200_000
    pretrain_depth: int = 9
    pretrain_epochs: int = 15
    use_oracle_programs: bool = False

    # Main Training
    main_iterations: int = 15
    main_budget: int = 300_000
    main_depth: int = 10
    main_epochs: int = 10
    recognition_blend: float = 0.5

    # Architecture
    hidden_dim: int = 128
    learning_rate: float = 1e-3

    # Parallelism
    n_workers: int = 4
    use_pypy: bool = True

    # Logging
    log_dir: str = "results/warmstart_experiment"
    checkpoint_every: int = 1
    log_embeddings: bool = True
```

### 10.3 Execution Plan

```bash
# Run all conditions with all seeds
for CONDITION in COLD WARM COLD+ WARM-O WARM-S WARM-D LIB-ONLY REC-ONLY RANDOM; do
    for SEED in 42 123 456 789 1010; do
        nohup caffeinate -d -i -s python3 src/experiments/run_warmstart_experiment.py \
            --condition $CONDITION \
            --seed $SEED \
            > logs/warmstart_${CONDITION}_${SEED}.log 2>&1 &
    done
done

# Wait for all runs to complete (estimated: 5-10 hours each)

# Analyze results
python3 src/experiments/analyze_warmstart_results.py \
    --results-dir results/warmstart_experiment

# Generate report
python3 src/experiments/generate_warmstart_report.py \
    --results-dir results/warmstart_experiment \
    --output docs/warmstart_results.html
```

---

## 11. Success Criteria

### 11.1 Minimum Success (H1 supported)
- WARM solve rate > COLD solve rate by at least 10 percentage points
- p < 0.05 on primary comparison

### 11.2 Strong Success
- WARM > COLD+ (benefit exceeds extra compute)
- At least 3 catalogue rules solved only by WARM
- Pretraining inventions reused in at least 5 catalogue solutions

### 11.3 Theoretical Success
- Clear correlation between pretraining similarity and transfer benefit
- Identifiable family-level transfer patterns

---

## 12. Limitations and Threats to Validity

### 12.1 Internal Validity
- **Randomness:** Addressed by multiple seeds, but stochastic search may still vary
- **Hyperparameter sensitivity:** Grid search is limited; optimal settings may not be found
- **Implementation bugs:** Pre-flight validation helps, but subtle issues possible

### 12.2 External Validity
- **Domain specificity:** Card game rules may not generalize to other domains
- **Rule set design:** Pretraining rules were manually designed to be "familiar"
- **Scale:** 45 catalogue rules is small compared to typical program synthesis benchmarks

### 12.3 Construct Validity
- **"Similarity" measurement:** Primitive overlap may not capture true conceptual similarity
- **"Solving" definition:** Binary solved/unsolved ignores partial progress

### 12.4 Mitigations
- Multiple random seeds reduce variance
- Ablation conditions isolate different components
- Per-family analysis provides finer-grained understanding

---

## 13. Timeline and Resources

### 13.1 Estimated Runtime

| Phase | Duration per Run | Total Runs | Total Time |
|-------|------------------|------------|------------|
| Pretraining (WARM conditions) | 3-4 hours | 25 | 75-100 hours |
| Main Training | 5-7 hours | 45 | 225-315 hours |
| Analysis | 1 hour | 1 | 1 hour |

**Total:** ~300-400 hours of compute (can be parallelized)

### 13.2 Resource Requirements
- **CPU:** 4+ cores recommended for parallel enumeration
- **Memory:** 8GB+ for recognition model and example storage
- **Storage:** ~10GB for checkpoints and logs
- **GPU:** Optional (recognition model is small, CPU is fine)

---

## 14. Appendices

### Appendix A: Pretraining Rule Families

| Family | Count | Examples |
|--------|-------|----------|
| poker | 6 | has_pair, flush, straight, same_color, two_suits |
| blackjack | 2 | sum_even, sum_odd |
| rummy | 4 | run_of_3, set_of_3, all_different, three_ranks |
| solitaire | 5 | alternating, v_shape, ascending, sorted, same_suit_seq |
| simple | 11 | first_red, last_black, ends_suit, has_spade, all_even, ... |
| symmetry | 4 | suits_palindrome, periodic_colors, ranks_palindrome, halves_suits |
| counting | 4 | two_red, more_red, three_suit, majority_color |
| compositional | 8 | left_half_uniform_color, right_half_has_pair, skip1_same_color, ... |

### Appendix B: Catalogue Rule Families

| Family | Count | Examples |
|--------|-------|----------|
| LOCAL | 4 | Sorted_by_rank, S_before_H, Ends_same_suit, Ends_same_color |
| COUNT | 6 | Has_pair_ranks, Uniform_color, Exactly_two_suits, ... |
| AP | 3 | AP_len3_anywhere_anyk, AP_len3_step2_anywhere, AP_len4_step2_anywhere |
| SCORE | 2 | Half_sum_diff_geN, Half_sum_one_side_ge_2x_other |
| HIER | 5 | Halves_uniform_color_equal, Halves_AP_step1_equal, ... |
| LANG | 3 | Well_formed_brackets_by_suit, Even_opens_next_closes, Odd_opens_next_closes |
| PAL | 3 | Suits_palindrome, Colors_palindrome, Ranks_palindrome |
| ALTCLR | 3 | AltColor1_palindrome, AltColor2_palindrome, Ends_same_altcolor1 |
| COPY | 6 | Halves_copy_suits, Halves_copy_colors, Halves_copy_ranks, ... |
| SHIFT | 2 | Shift_half_plus_two, Shift_half_ge |
| ADJ | 4 | Adj_same_rank_or_suit, Skip2_same_rank_or_suit, Adj_rank_gap_le3, Adj_same_rank_or_color |
| PARITY | 2 | Only_one_odd_rank, Uniform_rank_parity |
| CENTER | 2 | Halves_radial_nonincreasing, Global_radial_no_dominance |

### Appendix C: Expected Primitive Overlap

| Pretraining Family | Catalogue Family | Shared Primitives | Expected Transfer |
|-------------------|------------------|-------------------|-------------------|
| symmetry | PAL | reverse, eq, map | High |
| compositional | HIER, COPY | first_half, second_half, eq | High |
| poker | COUNT | unique, length, eq | Medium |
| solitaire | LOCAL | head, last, eq | Medium |
| blackjack | SCORE | sum_ranks, + | Medium |
| simple | ADJ | eq, or | Low |

---

## 15. References

1. Ellis, K., et al. (2021). DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. *PLDI 2021*.

2. Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. *Science*, 350(6266), 1332-1338.

3. Hewitt, L., et al. (2020). Learning to learn generative programs with Memoised Wake-Sleep. *UAI 2020*.

---

*End of Experimental Design Document*
