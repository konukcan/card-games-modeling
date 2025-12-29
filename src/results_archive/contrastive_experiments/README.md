# Contrastive Experiments

Experiments testing contrastive wake-sleep, memoized enumeration, and Set Transformer recognition.

## Runner Scripts
- `src/run_experimental_rules.py` (ACTIVE) - Set Transformer on catalogue rules
- `src/run_overnight_set_transformer.py` (ACTIVE) - Set Transformer overnight
- `legacy_runners/run_5iter_memoized.py` - Memoized enumeration test

## Directory Contents

### results_contrastive_full/
**Date**: December 21, 2024
**Focus**: Full contrastive learning runs

Contains:
- `contrastive_balanced_lean_plus_fold/` - Best performing configuration
- `experiment_*.log` - Detailed execution logs

Key insight: Contrastive dreaming improves task discrimination.

### results_lean_contrastive_test/
**Date**: December 21-22, 2024
**Focus**: Lean primitives with contrastive model

Testing the contrastive recognition model without fold primitives.

### results_5iter/
**Date**: December 23, 2024
**Focus**: 5-iteration memoized enumeration

Configuration:
- 5 wake-sleep iterations
- 45 tasks (catalogue rules)
- Enumeration budget: 50,000
- Memoized enumeration enabled

Results:
- Tasks solved: 7/45 (15.5%)
- Grammar growth: 59 → 84 primitives
- 25 abstractions learned
- Total time: ~20 minutes (much faster than expected)

### results_memoized_test/
**Date**: December 22-23, 2024
**Focus**: Memoized enumeration testing

Quick tests of the memoization system:
- Validates memoization works correctly
- Measures speedup from caching

### results_deep_enum/
**Date**: December 23, 2024
**Focus**: Deep enumeration testing

Testing deeper search configurations:
- Higher depth limits
- Larger budgets
- Looking for harder rules

### contrastive_test/
**Date**: December 19, 2024
**Focus**: Initial contrastive model testing

Early validation of contrastive recognition model.

## Key Innovations Tested

1. **Memoized Enumeration**: Cache partial programs to avoid redundant work
2. **Contrastive Recognition**: Learn to discriminate between similar tasks
3. **Set Transformer**: Better handling of set-structured inputs (hands of cards)
4. **Structural Similarity Loss**: Auxiliary loss for better embeddings

## Results JSON Format

```json
{
  "config": {...},
  "summary": {
    "tasks_solved": 7,
    "tasks_total": 45,
    "final_grammar_size": 84
  },
  "learning_curve": [
    {"iteration": 0, "tasks_solved": 7, "recognition_loss": 3.62, ...}
  ]
}
```

## How to Read Results

1. Check `*_results_*.json` for summary metrics
2. Look at `learning_curve` for performance over iterations
3. `final_grammar` shows learned abstractions
