# Recognition Model Code Review - Progress Document

**Created:** December 30, 2024
**Purpose:** Track progress of interactive code review session to prevent losing context

---

## Current Status: Component 9 (ContextualGrammarNetwork) - INTEGRATED

We discovered that three key components from the original DreamCoder were missing or not fully integrated:
1. **BigramHead** - Exists but was never trained ✅ (NOW FIXED)
2. **CountHead** - Was trained but never evaluated ✅ (NOW FIXED)
3. **ContextualGrammarNetwork** - Did not exist ✅ (NOW INTEGRATED)

Comparison scripts created to measure improvement from adding these components.

---

## Components Reviewed

### Component 1: hand_to_tensors() ✅
- **File:** `recognition_variants.py:32-70`
- **Function:** Converts Card objects to tensor indices (suit, rank, position)
- **Key insight:** Uses categorical indices, not continuous values
- **Decision:** None needed, this is foundational

### Component 2: Card Encoders ✅
- **Files:**
  - `contrastive_recognition.py:77-137` (FactoredCardEncoder)
  - `recognition_variants.py:77-154` (EnhancedCardEncoder)
- **Key finding:** EnhancedCardEncoder (R@5=0.657) beats FactoredCardEncoder (R@5=0.635)
- **Difference:** EnhancedCardEncoder adds color embedding + continuous rank value
- **TODO added:** Line 72-75 in contrastive_recognition.py - consider removing FactoredCardEncoder

### Component 3: Hand Encoders ✅
- **Files:**
  - `contrastive_recognition.py:188-236` (HandEncoder - mean pooling)
  - `recognition_variants.py:161-272` (SelfAttentionHandEncoder)
- **Key finding:** Simple mean pooling BEATS self-attention
- **Why:** Contrastive task encoding already captures relationships; attention adds noise
- **MLP explained:** Transforms independent features into combined features before pooling
- **Position concern:** Position info is "baked in" through MLP nonlinearity before averaging

### Component 4: Task Encoders ✅
- **Files:**
  - `contrastive_recognition.py:243-310` (ContrastiveTaskEncoder - standard)
  - `recognition_variants.py:477-633` (Random contrast variants)
- **Key formula:** τ = mean(positive_hands) - mean(negative_hands)
- **Key finding:** Standard contrast is BEST; random contrast variants hurt performance
- **TODOs added:** Lines 477, 534, 585 in recognition_variants.py - delete failed variants

### Component 5: Normalization Layer ⏸️ (PAUSED FOR DEEP DIVE)
- **File:** `contrastive_recognition.py:677-688, 850-854`
- **Current approach:** LayerNorm(τ) × learned_scale (post-hoc fix)
- **Problem:** τ has tiny magnitude (~0.05), causing uniform predictions
- **Discussion in progress:** Comparing to "cleaner" L2 normalization approach from SimCLR/CLIP

---

## Components Remaining

### Component 6: Prediction Heads ⏸️ (IN PROGRESS - MISSING COMPONENTS FOUND)
- PrimitiveHead (sigmoid) - best performer
- SoftmaxPrimitiveHead
- PrimitiveEmbeddingHead
- **BigramHead** - EXISTS but never trained (NOW FIXED)
- **CountHead** - trained but never evaluated (NOW FIXED)

### Component 7: Loss Functions
- BCE loss for primitive prediction
- MSE loss for CountHead (λ=0.1)
- BCE loss for BigramHead (λ=0.1) (NOW ADDED)
- **Missing:** Focal loss not implemented
- **Missing:** Structural similarity loss is O(n²)

### Component 8: Full Forward Pass
- How all components connect
- Training loop

### Component 9: ContextualGrammarNetwork ✅ (INTEGRATED)
- **File:** `dreamcoder_core/contextual_grammar.py`
- **Integration:** `dreamcoder_core/enumeration.py` (TopDownEnumerator)
- **Test Script:** `experiments/test_contextual_grammar_integration.py`
- Predicts P(primitive | task, parent, argument_position)
- Two variants: 'mask' (additive) and 'full' (multiplicative)

**Integration Completed (December 31, 2024):**
1. ✅ Modified `TopDownEnumerator.__init__()` to accept `contextual_grammar`, `task_embedding`, `contextual_weight`
2. ✅ Added `get_contextual_log_prob()` method for blending base + contextual predictions
3. ✅ Added `_apply_with_holes_contextual()` to track parent context for new holes
4. ✅ Updated `_expand_first_hole()` to use contextual predictions when available
5. ✅ Added `hole_contexts` tracking in `PriorityItem`

**Remaining TODOs:**
- Update `run_overnight_v3.py` to pass trained ContextualGrammarNetwork to enumerator
- Add training for ContextualGrammarNetwork in wake-sleep loop
- Measure improvement in search efficiency with trained model

---

## Key Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Card encoder | EnhancedCardEncoder | +3.5% R@5 improvement |
| Hand encoder | Mean pooling | Simpler, better performance |
| Task encoder | Standard (pos-neg) | Random variants hurt performance |
| hand_size default | 6 (standardized) | Was inconsistent (5 in some files) |

---

## Files Modified During Review

1. **contrastive_recognition.py** - Added TODO for FactoredCardEncoder
2. **contrastive_recognition_v1_baseline.py** - Added legacy TODO
3. **recognition_variants.py** - Added TODOs for 3 failed task encoders
4. **Multiple experiment files** - Fixed hand_size=5 → hand_size=6
5. **docs/random_contrast_report.md** - Updated hand size note

### Files Modified for Missing Component Integration (December 31, 2024)

6. **contrastive_recognition.py** - Added BigramHead training + CountHead/BigramHead evaluation methods
7. **neural_recognition.py** - Added BigramHead, CountHead, extract_bigrams, build_bigram_vocabulary + evaluation methods (parity with contrastive model)
8. **contextual_grammar.py** (NEW) - ContextualGrammarNetwork implementation with Mask/Full variants

### New Experiment Scripts Created

9. **experiments/compare_with_new_heads.py** - Re-run previous comparisons WITH bigram training and count evaluation
10. **experiments/compare_bigram_training.py** - Proper Pipeline A bigram test with real program parsing

### ContextualGrammarNetwork Integration (December 31, 2024)

11. **dreamcoder_core/enumeration.py** - Added contextual grammar support:
    - `PriorityItem.hole_contexts` field for tracking (parent, position) per hole
    - `TopDownEnumerator.__init__()` accepts `contextual_grammar`, `task_embedding`, `contextual_weight`
    - `get_contextual_log_prob()` method for blending predictions
    - `_apply_with_holes_contextual()` for context-aware hole creation
    - `_expand_first_hole()` uses contextual predictions when available
12. **experiments/test_contextual_grammar_integration.py** (NEW) - Integration test script

---

## Completed Deep Dive: L2 Normalization vs LayerNorm ✅

**Experiment conducted:** `experiments/compare_normalization_strategies.py`

**Three approaches tested:**
1. **LayerNorm+Scale (current)** - LayerNorm(τ) × learned_scale
2. **L2Norm+Temperature (CLIP-style)** - L2 normalize τ and primitives, cosine similarity / temperature
3. **L2Norm+SimpleHead (hybrid)** - L2 normalize τ, standard MLP head

**Results with small dataset (20 train / 15 test):**
| Approach | R@5 | vs Baseline |
|----------|-----|-------------|
| LayerNorm+Scale | 0.097 | -- |
| L2Norm+Temperature | 0.194 | +100% |
| **L2Norm+SimpleHead** | **0.528** | **+443%** |

**Results with larger dataset (35 train / 30 test):**
| Approach | R@5 | MRR | vs Baseline |
|----------|-----|-----|-------------|
| **LayerNorm+Scale** | **0.653** | 0.669 | -- |
| L2Norm+Temperature | 0.354 | 0.465 | -45.7% |
| L2Norm+SimpleHead | 0.566 | **0.743** | -13.3% |

**Key findings:**
1. **L2Norm+SimpleHead learns faster** with limited data (dominant in small tests)
2. **LayerNorm+Scale catches up** and surpasses with more training
3. **L2Norm+Temperature (pure CLIP-style) doesn't work** for multi-label prediction
4. **L2Norm+SimpleHead has best MRR** (ranks correct primitives higher)

**Conclusion:** Current LayerNorm+Scale is adequate for production. L2Norm+SimpleHead is a valid alternative if training data is limited. No immediate refactoring needed.

---

## To Resume Review

1. ✅ Finished normalization comparison (no refactoring needed)
2. ✅ Designed wake-sleep comparison experiment (see below)
3. ✅ Component 6: Prediction Heads (reviewed)
4. ✅ Component 7: Loss Functions (reviewed + ablation experiment)
5. ✅ Component 8: Full Forward Pass (reviewed)
6. **NEXT**: Final cleanup decisions

---

## Completed: Lambda Ablation Experiment ✅

**Date:** December 31, 2024
**Script:** `experiments/ablate_lambda_weights.py`

### Purpose
Test sensitivity to lambda hyperparameters for the combined loss function.

### Results

| Condition | R@5 | vs Baseline | Interpretation |
|-----------|-----|-------------|----------------|
| baseline | 0.141 ± 0.045 | (reference) | λ_struct=0.3, λ_count=0.1, λ_bigram=0.1 |
| no_struct | 0.056 ± 0.027 | **-60.5%** | Removing structural loss hurts significantly |
| high_struct | 0.067 ± 0.059 | **-52.6%** | λ_struct=1.0 is too much |
| no_count | 0.111 ± 0.047 | -21.1% | Count loss helps moderately |
| no_bigram | 0.089 ± 0.024 | -36.8% | Bigram loss helps |
| pred_only | 0.048 ± 0.053 | **-65.8%** | Just prediction loss is worst |

### Key Findings
1. **Structural loss is critical** - removing it drops R@5 by 60%
2. **Goldilocks zone** - λ_struct=0.3 is better than both 0.0 and 1.0
3. **All auxiliary losses contribute** - each one improves performance
4. **Bigram effect > count effect** - 36.8% vs 21.1% drop when removed

---

## Completed: Component 7 (Loss Functions) ✅

**Date:** December 31, 2024

### Loss Functions Summary

| Loss | Formula | Purpose | Default λ |
|------|---------|---------|-----------|
| L_pred | BCE(pred, target) | Primitive prediction | 1.0 |
| L_struct | Σ(cos(τᵢ,τⱼ) - Jaccard(Pᵢ,Pⱼ))² | Task clustering | 0.3 |
| L_count | MSE(pred_count, actual) | Primitive counting | 0.1 |
| L_bigram | BCE(pred_bigram, target) | Bigram prediction | 0.1 |

### Structural Loss Deep Dive
- Forces tasks with similar primitives to have similar embeddings
- Jaccard similarity: |A ∩ B| / |A ∪ B| measures set overlap
- O(n²) complexity - becomes expensive with many tasks

### Focal Loss
- Implemented but not used in main model
- Addresses class imbalance (different problem than structural)
- Down-weights easy examples: (1-p)^γ focal weight

---

## Completed: Component 8 (Full Forward Pass) ✅

**Date:** December 31, 2024

### Data Flow
```
Card → FactoredEmbedding(32) → CardMLP(32→64→32) → card_feature
Hand → MeanPool(card_features) → h ∈ ℝ³²
Task → (mean(h|pos) - mean(h|neg)) → LayerNorm×20 → τ ∈ ℝ³²
τ → PrimitiveHead → P(primitives)
τ → CountHead → primitive_count
τ → BigramHead → P(bigrams)
```

### Training Loop
1. Collect training data from solved frontiers
2. For each batch: encode tasks, compute predictions
3. Combined loss: λ_pred×L_pred + λ_count×L_count + λ_struct×L_struct + λ_bigram×L_bigram
4. Adam optimizer, lr=1e-3

### Key Locations
- Forward pass: `contrastive_recognition.py:772-854`
- Training loop: `contrastive_recognition.py:1010-1147`
- Grammar blending: `contrastive_recognition.py:908-934` (50% original + 50% predicted)

---

## Wake-Sleep Comparison Experiment Design

**Created:** December 31, 2024
**Script:** `experiments/compare_normalization_wakesleep.py`

### Purpose
Test whether L2 normalization's faster learning with limited data translates to better wake-sleep performance in early iterations.

### Configuration
| Parameter | Value |
|-----------|-------|
| Rules | 35 pretraining rules |
| Iterations | 5 wake-sleep cycles |
| Enumeration budget | 100,000 programs/task |
| Recognition epochs | 15 per iteration |
| Hidden dim | 64 |

### Metrics Collected
1. **Per iteration:** Tasks solved (cumulative and new), programs enumerated, wall time, training loss
2. **Learning curve:** tasks solved vs iteration for each normalization strategy
3. **Crossover detection:** when does one strategy overtake the other?

### To Run
```bash
cd src
python3 experiments/compare_normalization_wakesleep.py --quick   # Validate (~1 min)
python3 experiments/compare_normalization_wakesleep.py           # Full experiment (~2-3 hrs)
```

### Expected Outcomes
- **Early iterations (1-2):** L2 normalization should solve more tasks (faster learning)
- **Later iterations (3-5):** LayerNorm+Scale may catch up with more training data
- **Crossover point:** Approximately iteration 2-3 (to be determined empirically)

---

## Completed Deep Dive: BigramHead Training ✅

**Date:** December 31, 2024
**Script:** `experiments/compare_bigram_training.py`

### Methodology
- Used real parsed programs from cached overnight runs (27 parseable programs)
- Extracted actual bigrams from AST structure using `extract_bigrams()`
- 3-fold cross-validation, 30 epochs
- Compared: Model WITHOUT bigram loss vs Model WITH bigram loss (λ=0.1)

### Results

| Metric | WITHOUT Bigram | WITH Bigram | Difference |
|--------|----------------|-------------|------------|
| **R@5** | 0.278 ± 0.109 | 0.319 ± 0.255 | **+15.0%** |
| R@10 | 0.630 ± 0.210 | 0.639 ± 0.104 | +1.5% |
| MRR | 0.146 ± 0.038 | 0.173 ± 0.065 | +17.9% |

Bigram prediction quality: P@5=0.0, R@5=0.0 (see note below)

### Key Finding: Auxiliary Task Regularization

**Paradox:** BigramHead training improves R@5 (+15%) even though bigram predictions themselves are poor (0% precision/recall on test set).

**Explanation:** This is a multi-task learning phenomenon:
1. The bigram loss acts as a **regularizer** on the shared task encoder
2. The encoder learns richer representations useful for BOTH primitive AND bigram prediction
3. Even though bigram predictions are poor, the **learning signal** from trying to predict them improves the shared representations
4. Similar to "auxiliary task learning" - practicing one skill improves another through shared representations

**Analogy:** Learning to draw faces by also practicing hands. Hand drawings may still be bad, but the practice improves overall artistic skill (shared visual understanding), which makes faces better.

### Files Updated
- Removed `experiments/evaluate_head_improvements.py` (used fake primitives)
- Removed `experiments/test_bigram_with_cached_programs.py` (superseded)
- Created `experiments/compare_bigram_training.py` (proper Pipeline A with real parsing)

### TODO for Future Investigation
- Test with larger dataset (more overnight runs pooled)
- Investigate why bigram predictions are poor (vocabulary mismatch? sparse signal?)
- Consider curriculum learning for bigram loss

---

## Completed Integration: ContextualGrammarNetwork ✅

**Date:** December 31, 2024
**Test Script:** `experiments/test_contextual_grammar_integration.py`

### Architecture

ContextualGrammarNetwork predicts P(primitive | task, parent, arg_position):
- **Input:** task embedding τ (32-dim), parent primitive index, argument position
- **Output:** log-probabilities over primitives
- **Variants:** 'mask' (base + context bias) or 'full' (dedicated MLP per context)

### Integration with TopDownEnumerator

The enumerator now accepts optional contextual guidance:
```python
enumerator = TopDownEnumerator(
    grammar=grammar,
    contextual_grammar=cgn,        # ContextualGrammarNetwork
    task_embedding=task_embedding, # From recognition model
    contextual_weight=0.5          # Blend: (1-w)*base + w*contextual
)
```

### Test Results

**Test 1-2:** ContextualGrammarNetwork creation and context-dependent predictions work
- Different contexts produce different log-probability distributions
- Mean abs difference between root vs filter context: 0.06

**Test 3-4:** Enumeration works both without and with contextual grammar
- 50 programs enumerated in both cases
- **Key result:** Ordering changes with contextual guidance

**Test 5: Ordering Differs With Context**
| Comparison | Same-Position Matches |
|------------|----------------------|
| base vs ctx1 | 3.33% |
| base vs ctx2 | 13.33% |
| ctx1 vs ctx2 | 16.67% |

This confirms that contextual guidance significantly changes program enumeration order.

**Test 6:** Log probability blending works correctly

### Next Steps

1. Train ContextualGrammarNetwork on solved programs from overnight runs
2. Integrate with wake-sleep loop in `run_overnight_v3.py`
3. Measure improvement in search efficiency (programs enumerated to find solution)

---

## Reference: Performance Results

| Configuration | R@5 | Notes |
|---------------|-----|-------|
| baseline_enhanced (best) | 0.657 | EnhancedCardEncoder + sigmoid + mean pooling |
| baseline_standard | 0.635 | FactoredCardEncoder + sigmoid + mean pooling |
| **+ BigramHead training** | +15% relative | Auxiliary task regularization effect |
| triple_contrast | 0.609 | -5.7% vs standard |
| random_augmented | 0.623 | -3.6% vs standard |
| self-attention | 0.604 | Worse than mean pooling |
