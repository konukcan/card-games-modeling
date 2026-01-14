# Lingering Issues and Future Work

This document reviews known issues, potential problems, and future improvement directions for the DreamCoder wake-sleep implementation. Created January 2026.

**Related Documentation:**
- `KNOWN_ISSUES.md` - Historical bugs that have been fixed
- `ARCHITECTURE.md` - Parallelization trade-offs and options
- `TASK_GENERATION.md` - Task creation system details

---

## 1. Dreaming Phase Issues

### 1.1 Low Dream Training Success Rate

**Status**: PARTIALLY FIXED (Jan 2026)

**Current Behavior:**
- 20 dreams generated per iteration
- Only ~2 dreams successfully train recognition
- Many dreams fail example generation

**Root Cause:**
We implemented `sample_requiring_variable()` which ensures programs syntactically use `$0`, but doesn't guarantee *meaningful* variation in output. A program like `(λ all ((λ false)) (unique $0))` uses `$0` but always returns `false` regardless of input.

**Evidence from Test Run:**
```
Generating 10 contrastive dreams...
  Dream 1: (λ has_suit (second_half $0) SPADES)... (10 examples, 2 near-miss pairs)
  Failed to generate examples for: (λ all ((λ false)) (unique $0))...
```

**Potential Solutions:**

1. **Behavioral Diversity Check** (Recommended)
   ```python
   def passes_behavioral_diversity(program, sample_fn, n_samples=20, min_variation=0.2):
       """Reject programs whose output doesn't vary across inputs."""
       results = []
       for _ in range(n_samples):
           hand = sample_fn()
           try:
               result = program.evaluate([])(hand)
               results.append(result)
           except:
               return False

       unique_ratio = len(set(results)) / len(results)
       return unique_ratio >= min_variation
   ```

2. **Relaxed Example Balance Requirements**
   - Current: Require 50/50 positive/negative split
   - Proposed: Accept 60/40 or even 70/30 splits
   - Trade-off: Less balanced training data, but more dreams used

3. **Retry with Different Programs**
   - If example generation fails, sample a new program
   - Set maximum retries per dream (e.g., 5)
   - May increase dream generation time

**Estimated Fix Time**: 2-4 hours

---

### 1.2 Dream Training Epochs

**Status**: POTENTIAL ISSUE

**Current Behavior:**
Dreams train recognition for just 1 epoch:
```python
self.recognition.train_on_frontiers(
    tasks=dream_tasks,
    frontiers=dream_frontiers,
    epochs=1,  # Very short!
    batch_size=min(8, len(dream_tasks))
)
```

**Concern:**
With only ~2 successful dreams and 1 epoch, the contribution to recognition learning may be negligible.

**Potential Solutions:**
1. Increase epochs to 3-5 for dream training
2. Accumulate dreams across iterations before training
3. Weight dream loss contribution relative to real task loss

**Investigation Needed**: Compare recognition loss improvement with/without dream training.

---

## 2. Recognition Model Issues

### 2.1 Saturated Probability Predictions

**Status**: OBSERVED (Jan 2026)

**Current Behavior:**
All top primitive predictions show probability 1.00:
```
Top predictions: #((λ (λ lt (n_u(1.00), #((λ (λ (λ lt ((1.00), ...
```

**Concern:**
If all predictions are saturated at 1.0, the model isn't providing discriminative guidance. This could indicate:
- Softmax temperature too low
- Model overconfident after few training iterations
- All learned abstractions equally probable

**Potential Solutions:**

1. **Temperature Scaling**
   ```python
   # In predict_grammar_weights():
   logits = self.primitive_head(task_embedding)
   probs = F.softmax(logits / temperature, dim=-1)  # Add temperature parameter
   ```

2. **Entropy Regularization**
   - Add entropy term to loss to prevent overconfidence
   - Encourages model to spread probability mass

3. **Dropout During Prediction** (for calibration)
   - MC Dropout gives uncertainty estimates
   - More conservative predictions

**Estimated Investigation Time**: 1-2 hours

---

### 2.2 Recognition Hidden Dimension

**Status**: POSSIBLY SUBOPTIMAL

**Current Setting**: `recognition_hidden_dim: int = 32`

**Context:**
The Set Transformer architecture was previously tested with larger dimensions (128-256). The current 32-dim model may lack capacity for complex task representations.

**Historical Reference** (from `runner_scripts_history.md`):
> Key parameters: 256 hidden dim, 150 dreams/iteration, 20 recognition epochs.

**Trade-off:**
- Larger: Better task discrimination, slower training
- Smaller: Faster training, risk of underfitting

**Recommendation**: Conduct ablation comparing 32, 64, 128 hidden dims.

---

## 3. Unsolved Task Patterns

### 3.1 Tasks Requiring Missing Primitives

**Status**: FUNDAMENTAL LIMITATION

**21 unsolved tasks** share patterns that may be inexpressible with current primitives:

| Task Pattern | Example | Missing Primitive |
|--------------|---------|-------------------|
| Symmetry | `sym_suits_palindrome` | `reverse`, `eq_lists` |
| Sorting | `sol_ascending`, `sol_sorted` | `is_sorted`, `sorted` |
| Exact counting | `count_two_red` | `eq` with specific constants |
| Periodicity | `sym_periodic_colors` | `cycle_detection`, `modular` |

**Evidence:**
```
# These tasks remain unsolved across iterations:
[ ] poker_three_of_kind    - Needs specific rank counting
[ ] poker_straight         - Needs consecutive rank detection
[ ] rummy_run_3            - Same as straight
[ ] sol_alternating        - Needs pairwise comparison
```

**Potential Solutions:**

1. **Add Missing Primitives**
   - `is_sorted: list[int] → bool`
   - `all_consecutive: list[int] → bool` (for straights)
   - `count_exact: (list[T] → int) → int → bool` (for exact counts)

2. **DSL Extension Study**
   - Systematically identify which primitives would solve remaining tasks
   - Measure cost/benefit of adding each primitive

**Reference**: See `archived/special_purpose/` for previous DSL expansion experiments.

---

### 3.2 Search Budget Distribution

**Status**: POTENTIAL INEFFICIENCY

**Current Behavior:**
All tasks get equal enumeration budget (100,000 in quick mode, 500,000 in overnight).

**Observation:**
Easy tasks solve quickly (672 programs for `poker_same_color`), while hard tasks exhaust budget (100,000 for `poker_straight`).

**Potential Solutions:**

1. **Adaptive Budget Allocation**
   - Start with small budget for all tasks
   - Allocate remaining budget to promising unsolved tasks
   - Based on recognition model confidence

2. **Early Termination with Confidence**
   - If recognition model is very uncertain, terminate early
   - Reallocate budget to more tractable tasks

---

## 4. Code Quality Issues

### 4.1 Remaining Bare Exception Handlers

**Status**: PARTIALLY FIXED

**Current State:**
Most bare `except:` blocks fixed, but some remain in experiment scripts:
```
src/experiments/run_transfer_study.py:284:        except:
src/experiments/run_transfer_study.py:337:            except:
src/experiments/archive/diagnose_*.py                multiple
```

**Risk:**
Silent failures can mask bugs. The `train_on_dream()` bug was caused by exactly this pattern.

**Fix**: Replace all bare `except:` with specific exception types.

**Estimated Fix Time**: 30 minutes

---

### 4.2 Hardcoded Paths

**Status**: LOW PRIORITY

**Affected Files:**
- Some scripts have hardcoded paths that assume specific directory structure
- May break when running from different locations

**Recommendation**: Use `Path(__file__).parent` pattern consistently.

---

## 5. Parallelization Gap

### 5.1 PyPy Workers Without Neural Guidance

**Status**: ARCHITECTURAL DEBT

**The Problem** (from `ARCHITECTURE.md`):
```python
# CURRENT (incomplete - weights never passed):
def enumerate_task(task_data, grammar_productions, ...):
    grammar = build_lean_grammar()  # Always default weights!
    if grammar_productions:         # This is ALWAYS None!
        pass  # Never executed
```

PyPy worker infrastructure exists but neural guidance weights are never serialized and passed.

**Historical Context:**
Original DreamCoder solves this with JSON serialization to OCaml workers:
> "The Python frontend communicates a request to the OCaml backend in JSON format after serializing the current library to JSON."

**Impact:**
- PyPy workers enumerate blindly (no neural guidance)
- Defeats purpose of recognition training
- Current sequential approach is actually better for research

**Future Implementation Path:**

1. **Option B: CPython Workers** (~1 day)
   - Use ProcessPoolExecutor with CPython
   - Serialize weights via pickle
   - ~4x speedup, full guidance

2. **Option C: PyPy + Pre-computed Weights** (~2 days)
   - Serialize weights to JSON
   - PyPy worker rebuilds guided grammar
   - ~7-15x speedup with guidance

**Recommendation**: Document but defer until research needs production-scale runs.

---

## 6. Compression Phase Considerations

### 6.1 Quality Filter Tuning

**Status**: RECENTLY FIXED (Jan 2026)

**Current State:**
Three quality filters now active:
- `is_nontrivial()` - Requires meaningful complexity
- `is_eta_reducible()` - Blocks redundant wrappers
- `is_single_task_abstraction()` - From Stitch

**Potential Issue:**
Filters may be too aggressive, rejecting useful abstractions.

**Evidence Needed:**
- Log rejected abstractions with reasons
- Compare rejected vs. accepted abstraction utility

---

### 6.2 Forward-Looking Scoring

**Status**: ENABLED

**Current Behavior:**
```python
# corpus_guidance_alpha: float = 0.7  # Weight for unsolved task fit
```

Compression scores abstractions by potential utility on unsolved tasks.

**Potential Improvements:**
- Weight by task family similarity
- Consider recognition model's predictions for unsolved tasks
- Multi-objective: MDL savings + recognition guidance fit

---

## 7. Logging and Diagnostics

### 7.1 Per-Iteration Model Analysis

**Status**: NOT IMPLEMENTED

**What's Missing** (from `KNOWN_ISSUES.md`):
```python
# DESIRED but not saved:
iteration_checkpoint = {
    'task_embeddings': {task_name: tensor},
    'attention_weights': {task_name: weights},
    'primitive_predictions': {task_name: logprobs},
    'feature_importance': {task_name: {feature: score}},
}
```

**Impact:**
Cannot analyze how recognition model evolves, which limits debugging and interpretability.

**Recommendation**: Add optional detailed logging mode.

---

### 7.2 Dream Success Rate Tracking

**Status**: PARTIALLY IMPLEMENTED

**Current:**
- Logs "Trained recognition on N dreams"
- Does not log *why* other dreams failed

**Improvement:**
```python
dream_failures = {
    'example_generation': 0,
    'all_positive': 0,
    'all_negative': 0,
    'exception': 0,
}
```

Would help diagnose dream generation bottleneck.

---

## 8. Task Generation Considerations

### 8.1 Near-Miss Generation for Rare Rules

**Status**: DOCUMENTED (see `TASK_GENERATION.md`)

**The Issue:**
Some rules have very low positive rates. Near-miss generation may fail:
```json
{
  "failures": [
    {"rule_id": "sym_ranks_palindrome", "reason": "Could not find enough positives..."}
  ]
}
```

**Current Mitigation:**
- Up to 200,000 sampling attempts
- Proportional adjustment if not enough positives
- Skip rule if < 80% target found

**Future Consideration:**
For very rare rules, consider:
- Constructive generation (generate hands that satisfy rule)
- Larger sampling budget
- Separate "rare rule" handling

---

## Priority Ranking

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| Dream behavioral diversity check | HIGH | 2-4h | **1** |
| Bare `except:` cleanup | MEDIUM | 30min | **2** |
| Recognition temperature tuning | MEDIUM | 1-2h | **3** |
| Dream epoch increase | LOW | 15min | **4** |
| DSL expansion study | HIGH | 1-2 days | **5** |
| Parallelization implementation | MEDIUM | 1-2 days | **6** |
| Per-iteration logging | LOW | 2-3h | **7** |

---

## References

- `KNOWN_ISSUES.md` - Historical bug documentation
- `ARCHITECTURE.md` - Parallelization trade-offs
- `TASK_GENERATION.md` - Task creation details
- `runner_scripts_history.md` - Evolution of experiment scripts
- `archived/` - Historical implementations and ablation studies

---

*Last updated: January 13, 2026*
