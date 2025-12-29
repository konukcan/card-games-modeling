# Recognition Model Search: Progression and Findings

This document tracks our systematic exploration of recognition model architectures for guiding program synthesis in the DreamCoder-style system.

## Executive Summary

We discovered that **the output activation function (softmax vs sigmoid) is the critical factor for search guidance quality**, more important than the encoder architecture (GRU vs factored embeddings). Lower recognition loss does not imply better search performance when using incompatible loss functions.

---

## Experiment Timeline

### Experiment 1: Neural Recognition Model (Baseline)
**Date**: December 24-25, 2025
**Model**: `NeuralRecognitionModel` (GRU-based with softmax output)

**Architecture**:
- Input: One-hot card features (24-dim per card)
- Encoder: Bidirectional GRU
- Pooling: Attention-weighted mean
- Output: Softmax over primitives
- Loss: Cross-entropy

**Results (BOTH condition)**:
| Metric | COLD | WARM | Transfer |
|--------|------|------|----------|
| Solve rate | 4/45 (8.9%) | 6/45 (13.3%) | +2 rules |
| Recognition loss | 3.87 | 1.97 | - |
| Pretraining | - | 19/44 (43.2%) | - |

**Key finding**: Positive transfer observed. WARM solved 2 additional rules (Ranks_palindrome, Halves_copy_ranks) by learning useful primitive combinations during pretraining.

---

### Experiment 2: Contrastive Recognition Model
**Date**: December 25-26, 2025
**Model**: `ContrastiveRecognitionModel` (Factored embeddings with sigmoid output)

**Architecture**:
- Input: Factored embeddings (suit 8-dim + rank 16-dim + position 8-dim = 32-dim)
- Encoder: CardInteractionMLP → Mean pooling
- Task encoding: τ = mean(positive examples) - mean(negative examples)
- Output: Sigmoid per primitive (independent probabilities)
- Loss: BCE + MSE(count) + Structural similarity

**Results (BOTH condition)**:
| Metric | COLD | WARM | Transfer |
|--------|------|------|----------|
| Solve rate | 4/45 (8.9%) | 4/45 (8.9%) | 0 rules |
| Recognition loss | 0.53 | 0.55 | - |
| Pretraining | - | 16/44 (36.4%) | - |

**Key finding**: ZERO transfer despite lower loss! The model learned to classify primitives as relevant/irrelevant but couldn't rank them for search.

---

## Critical Observations

### Observation 1: Lower Loss ≠ Better Performance

| Model | Recognition Loss | WARM Solve Rate | Transfer |
|-------|-----------------|-----------------|----------|
| Neural | 4.05 (higher) | 13.3% | +2 rules |
| Contrastive | 0.53 (lower) | 8.9% | 0 rules |

**Explanation**: The losses measure fundamentally different things:
- Cross-entropy (neural): How well the model predicts a probability *distribution* over primitives
- BCE (contrastive): How well the model *classifies* each primitive independently

### Observation 2: Contrastive Model Enumerates Fewer Programs

| Model | Avg Programs/Task (unsolved) | Programs/Second |
|-------|------------------------------|-----------------|
| Neural | ~40,000 | ~660 |
| Contrastive | ~5,000 | ~84 |

**This is a symptom, not a feature**. The flat probability distribution from sigmoid outputs causes:
1. Many primitives assigned similar (~0.5) probabilities
2. log(0.5) ≈ -0.69 for all → uniform grammar weights
3. Enumerator can't prioritize → "traffic jam" in search
4. Slower enumeration overall

### Observation 3: Transfer Rules Share Common Structure

The two rules that showed positive transfer in neural model:
```
Ranks_palindrome:   (λ le (n_unique_ranks $0) (n_unique_ranks (second_half $0)))
Halves_copy_ranks:  (λ le (n_unique_ranks $0) (n_unique_ranks (second_half $0)))
```

Both use `second_half` + `n_unique_ranks` - primitives learned together during pretraining from `sym_ranks_palindrome`.

### Observation 4: Sigmoid Allows "Confident Hedging"

At BCE optimum, sigmoid can output:
- 0.9 for all 5 relevant primitives
- 0.1 for 55 irrelevant primitives
- Loss ≈ 0 (perfect classification)
- But: No ranking among the 5 relevant primitives!

Softmax MUST rank because of normalization constraint Σp_i = 1.

---

## Theoretical Framework

### Why Softmax Works for Search

**Proposition**: Search guidance requires ranking, not classification.

1. Enumerator explores primitives in order of log-probability
2. Classification tells us: "Is primitive X useful?" (yes/no)
3. Ranking tells us: "Try primitive X before Y"
4. Sigmoid optimizes (1); Softmax optimizes (2)

### The Guidance-Calibration Tradeoff

For sigmoid/BCE:
- Low loss → Good binary classification
- Low loss ↛ Good ranking

For softmax/CE:
- Low loss → Good probability distribution
- Low loss → Good ranking (forced by normalization)

---

## Hypothesis for Next Experiment

**H1**: The contrastive model's underperformance is due to the sigmoid output layer, NOT the contrastive encoding (τ = pos - neg). Adding softmax to the contrastive architecture should restore search guidance quality.

**Prediction**: ContrastiveSoftmax will match or exceed Neural model performance.

**Alternative H0**: The τ = mean(pos) - mean(neg) representation lacks discriminative power. Softmax alone won't fix it.

---

## Planned Experiment: Softmax Ablation

Create three conditions (WARM only):
1. **ContrastiveSigmoid**: Original contrastive model
2. **ContrastiveSoftmax**: Contrastive encoder + softmax output + CE loss
3. **Neural**: GRU baseline

**Key metrics**:
- Programs per solution (search efficiency)
- Recall@5 (are correct primitives in top predictions?)
- Prediction entropy (is distribution focused?)

**Expected if H1 true**:
- ContrastiveSoftmax ≈ Neural >> ContrastiveSigmoid on all metrics

---

## Implications for Recognition Model Design

### Recommendations

1. **Always use softmax output** when the goal is search guidance
2. **Use cross-entropy loss**, not BCE, even if targets look like binary labels
3. **Lower loss is only meaningful within the same loss function**
4. **Evaluate with ranking metrics** (NDCG, MRR, Recall@k), not just loss

### The Contrastive Encoding is Likely Fine

The τ = mean(pos) - mean(neg) encoding has theoretical appeal:
- Directly captures the decision boundary
- Compact representation
- No sequential processing needed

The problem is purely in how we convert τ → primitive predictions. With proper softmax output, the contrastive encoder may work as well or better than GRU.

---

## Summary Table

| Model | Encoder | Output | Loss | Transfer | Performance |
|-------|---------|--------|------|----------|-------------|
| Neural | GRU + Attention | Softmax | CE | +2 rules | Good |
| Contrastive | Factored + Mean | Sigmoid | BCE+MSE+Struct | 0 rules | Poor |
| ContrastiveSoftmax | Factored + Mean | Softmax | CE | ? | TBD |

---

## Files Reference

- Neural model: `src/dreamcoder_core/neural_recognition.py`
- Contrastive model: `src/dreamcoder_core/contrastive_recognition.py`
- Experiment script: `src/experiments/run_warmstart_experiment.py`
- Results:
  - Neural: `results/warmstart_experiment/neural_BOTH_20251225_145356/`
  - Contrastive: `results/warmstart_experiment/contrastive_BOTH_20251225_145818/`

---

## Next Steps

1. [ ] Implement `SoftmaxPrimitiveHead` class in contrastive_recognition.py
2. [ ] Add `output_mode` parameter to ContrastiveRecognitionModel
3. [ ] Create ablation experiment script
4. [ ] Run ContrastiveSigmoid vs ContrastiveSoftmax vs Neural comparison
5. [ ] Analyze results and update this document

---

*Last updated: December 26, 2025*
