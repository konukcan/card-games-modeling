# DreamDecompiler Reference Guide

This document summarizes the DreamDecompiler algorithm (Palmarini et al., ICML 2024) for recognition-guided compression, adapted for our card game modeling project.

## Paper Citation

```bibtex
@inproceedings{palmarini2024dreamdecompiler,
  title={Bayesian Program Learning by Decompiling Amortized Knowledge},
  author={Palmarini, Alessandro B. and Lucas, Christopher G. and Siddharth, N.},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  pages={39042--39055},
  year={2024},
  publisher={PMLR},
  volume={235}
}
```

**arXiv:** https://arxiv.org/abs/2306.07856
**GitHub:** https://github.com/abpalmarini/dreamdecompiler

---

## Core Insight

Standard DreamCoder uses the recognition model only during enumeration (to guide search). The compression phase operates purely symbolically based on MDL (Minimum Description Length).

**DreamDecompiler's innovation:** Use the recognition model's learned preferences to guide *which abstractions to add to the library* - not just which programs to enumerate.

> "The amortized knowledge learnt to reduce search breadth is now also used to reduce search depth."

---

## The Algorithm

### 1. Caching Benefit Measure

The central formula measures how beneficial it would be to cache a fragment `f` in the library:

```
C(q, ρ, f) = (n_ρ^f × log q(f)) / log q(ρ)
```

Where:
- `n_ρ^f` = number of times fragment `f` appears in program `ρ`
- `q(f)` = recognition model's probability of generating `f`
- `q(ρ)` = recognition model's probability of generating complete program `ρ`

**Intuition:** This ratio measures what fraction of the "cost" (negative log probability) of generating `ρ` is attributable to generating `f`. If `f` is hard to generate (low `q(f)`) but appears multiple times, caching it provides high benefit.

### 2. Chunking Probability (DDC-PC)

The probability that a fragment `f` should be chunked (added to library):

```
p(c=1 | f; φ) ∝ E_x[ q(f|x) × Σ_ρ [ 1[ρ solves x] × C(q, ρ, f) × q(ρ|x) ] ]
```

Components:
1. `q(f|x)` — How much the recognition model prefers `f` for task `x`
2. `C(q, ρ, f)` — The caching benefit ratio
3. `q(ρ|x)` — Recognition model's preference for program `ρ` on task `x`

### 3. Two Variants

| Variant | Description | Selection Criterion |
|---------|-------------|---------------------|
| **DDC-Avg** | Average recognition probability across tasks | Simple but biases toward small fragments |
| **DDC-PC** | Full probabilistic chunking with caching benefit | Principled; maintains larger abstractions |

---

## Integration with DreamCoder

DreamDecompiler replaces the compression phase:

```
Standard DreamCoder:
  Wake → Dreaming → Compression (MDL-based)

With DreamDecompiler:
  Wake → Dreaming → DreamDecompiler (recognition-guided)
```

### Entry Points (from their code)

```python
# Simple version (fragment proposal)
DreamDecompiler.consolidate(
    grammar,
    recognitionModel,
    frontiers,
    useProgramPrimArgTypeCounts=True,
    fromRoot=False,
    chunkWeighting="raw",
    pseudoCounts=1.0,
    arity=3,
    CPUs=1
)

# Version space version (more efficient)
DreamDecompiler.consolidateVS(
    grammar,
    recognitionModel,
    frontiers,
    useProgramPrimArgTypeCounts=True,
    fromRoot=False,
    chunkWeighting="raw",
    numConsolidate=2,  # top-k or threshold
    maximumFrontier=10,
    pseudoCounts=1.0,
    arity=3,
    topK=2,
    topI=300,
    bs=1000000,
    CPUs=1
)
```

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunkWeighting` | How to weight programs in beam: "raw", "prop", or "uniform" | "raw" |
| `fromRoot` | Whether to compute likelihood from root or marginalize over contexts | False |
| `numConsolidate` | Number of candidates to chunk (int) or probability threshold (float) | 2 |
| `useProgramPrimArgTypeCounts` | Weight by frequency of (parent, arg, type) in programs | True |
| `arity` | Maximum arity for candidate fragments | 3 |

---

## Experimental Results (from paper)

| Domain | Improvement (DDC-PC vs DreamCoder) |
|--------|-----------------------------------|
| List Processing | +13.25% |
| Block Towers | +17%+ |
| Symbolic Regression | +17%+ |
| LOGO Graphics | +17%+ |
| Text Editing | ~comparable |
| Regexes | moderate |

**Key finding:** Improvements are most pronounced in cycles 3-5 (early learning).

---

## Adaptation for Our System

### Our Recognition Model Interface

```python
class ContrastiveRecognitionModel:
    def predict_primitives_dict(self, task) -> Dict[str, float]:
        """Returns {primitive_name: probability} for task."""

    def predict_grammar_weights(self, task) -> Grammar:
        """Returns task-specific grammar with adjusted weights."""
```

### Simplified Adaptation

For our contrastive recognition model, we can compute a simplified "recognition score":

```python
def recognition_score(fragment, unsolved_tasks, recognition_model):
    """
    Score a candidate fragment by recognition model predictions.

    Higher score = recognition model predicts these primitives
    are useful for unsolved tasks.
    """
    # Extract primitive names from fragment
    primitives_in_fragment = collect_primitive_names(fragment)

    scores = []
    for task in unsolved_tasks:
        # Get recognition predictions
        preds = recognition_model.predict_primitives_dict(task)

        # Sum predictions for primitives in fragment
        task_score = sum(preds.get(p, 0.0) for p in primitives_in_fragment)
        scores.append(task_score)

    return np.mean(scores) if scores else 0.0
```

### Combined Scoring

```python
def combined_score(fragment, backward_mdl_savings, forward_recognition_score, alpha=0.7):
    """
    Combine backward (MDL) and forward (recognition) scores.

    alpha = 1.0: Pure backward (original DreamCoder)
    alpha = 0.7: Mostly backward, some forward (recommended)
    alpha = 0.0: Pure forward (experimental)
    """
    # Normalize both scores to [0, 1] for combination
    norm_backward = backward_mdl_savings / max_backward_savings
    norm_forward = forward_recognition_score / max_forward_score

    return alpha * norm_backward + (1 - alpha) * norm_forward
```

---

## Key Differences from Our Current Compression

| Aspect | Current (MDL-only) | With DreamDecompiler |
|--------|-------------------|---------------------|
| Abstraction selection | Based on compression savings | Based on recognition predictions |
| Unsolved tasks | Not considered | Scored for predicted utility |
| Recognition model | Only used in enumeration | Also used in compression |
| Forward-looking | No | Yes |

---

## Implementation Checklist

- [x] Add `collect_primitive_names()` utility to extract primitives from Program
      → Implemented in `compression.py:collect_primitives_from_body()`
- [x] Add `recognition_score()` function to compute forward score
      → Implemented as `compression.py:compute_recognition_score()`
- [x] Add `combined_score()` function to blend MDL and recognition scores
      → Implemented as `compression.py:compute_combined_score()`
- [x] Add `compress_frontiers_recognition()` to accept optional recognition guidance
      → Implemented in `compression.py` (~150 lines)
- [x] Update `ContrastiveWakeSleep._run_iteration()` to pass unsolved tasks
      → Modified compression phase with conditional logic
- [x] Add tests for recognition-guided compression
      → Integration tests pass with both enabled/disabled
- [ ] Run ablation comparing MDL-only vs recognition-guided
      → TODO: Overnight comparison experiment

---

## Files to Reference

**DreamDecompiler source:**
- `/tmp/dreamdecompiler/dreamcoder/dreamdecompiler.py` - Main algorithm (1600+ lines)
- `/tmp/dreamdecompiler/dreamcoder/dreamcoder.py:716-784` - Integration with wake-sleep

**Our files to modify:**
- `src/dreamcoder_core/compression.py` - Add recognition guidance
- `src/dreamcoder_core/contrastive_wake_sleep.py` - Pass unsolved tasks
- `src/dreamcoder_core/contrastive_recognition.py` - Interface for scoring

---

## Connection to Compression-Based Objectives

Under a **uniform recognition model** (q(f) = constant for all f), the caching benefit simplifies to:

```
p(c | f) ∝ size(f) × Σ_ρ n_ρ^f × w(ρ)
```

This **recovers compression-based objectives** but emerges organically. Compression-based library learning is a special case when the recognition model hasn't learned anything yet.
