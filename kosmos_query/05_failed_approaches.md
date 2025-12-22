# Failed Approaches and Analysis

This document describes approaches we've tried that haven't worked, to help focus the search for solutions.
***IMPORTANT NOTE***: it is possible that some of these approaches have failed because of minor implementation bugs. However, we've tried to be careful and the failures seem consistent across multiple runs and hyperparameter settings. At the hear of the issue here is that I don't have a good high-level representation of what would be the right architecture for this task, and this is what I need to understand.

## Approach 1: Standard BiGRU (Original DreamCoder Style)

### Architecture
```
Card features (24-dim) → GRU encoder → hand embedding (64-dim)
Hand embedding + label (2-dim) → concatenate → example embedding
Examples → attention pooling → task embedding
Task embedding → MLP → primitive log-probabilities
```

### Why It Failed
- **Evidence**: Cosine similarity between different task embeddings: 0.95-0.99
- **Root cause**: The label (True/False) provides only 2 bits of information
- In original DreamCoder, the OUTPUT is informative (e.g., `[1,2,3] → [6]` for sum)
- In our classification task, the OUTPUT is just a label (True/False)
- The GRU encodes card sequences well, but has no way to learn *why* some hands are True

### Metrics
| Metric | Observed | Expected |
|--------|----------|----------|
| Prediction entropy std | 0.0005 | Much higher |
| Top primitive diversity | 1 (same for all) | 8-20 different |
| Task embedding similarity | 0.95-0.99 | < 0.5 |

---

## Approach 2: Attention-Weighted Pooling

### Modification
Added attention mechanism over examples to weight them before pooling:
```python
attention_weights = softmax(W @ example_embeddings)
task_embedding = sum(attention_weights * example_embeddings)
```

### Why It Failed
- Slight improvement in embedding variance
- Still collapsed: similarity 0.90-0.95
- Attention doesn't help if individual example embeddings are already too similar

---

## Approach 3: Set Transformer Architecture

### Architecture
Based on Lee et al. (2019) Set Transformer:
```
Cards → SAB (self-attention blocks) → PMA (pooling by multihead attention) → hand embedding
Examples → SAB → PMA → task embedding
```

### Benefits
- Better at modeling card-card interactions
- Learnable positional encoding for position-sensitive rules
- More expressive pooling via PMA

### Why It Still Failed
- Better at encoding hands, but still weak task discrimination
- Similarity dropped to 0.85-0.95 (improvement but insufficient)
- The fundamental issue remains: label signal is too weak

---

## Approach 4: FiLM Conditioning

### Modification
Used Feature-wise Linear Modulation to condition hand embedding on label:
```python
gamma = label_to_gamma(label)  # (64-dim)
beta = label_to_beta(label)    # (64-dim)
modulated_hand = gamma * hand_embedding + beta
```

### Why It Failed
- FiLM works when the conditioning signal is rich (e.g., language description)
- With just True/False, gamma and beta are essentially constant
- Model learns: gamma ≈ [1,1,1,...], beta ≈ [0.1,-0.1,0.1,...] for True
- Same for False with slightly different values
- Result: modulation is near-identity, doesn't discriminate

---

## Approach 5: Raw Feature Correlation

### Idea
Bypass learned embeddings entirely. Compute correlation between raw card features and labels:
```python
for each feature_dim in [suit_onehot, rank_onehot, color_onehot, ...]:
    correlation[dim] = corr(feature_values, labels)
```

### Why It's Partial Success
- **Best so far**: Actually captures some discriminative signal
- For "uniform_color" task: color features highly correlated with True/False
- For "ends_same_suit" task: position×suit interaction correlated

### Limitations
- Linear correlations miss non-linear interactions
- First-order features only (no card-card relationships)
- Manual feature engineering, doesn't generalize

---

## Analysis: What the Model Needs

### The Core Challenge

Different rules attend to different feature subspaces:

| Rule | Relevant Features | Irrelevant Features |
|------|-------------------|---------------------|
| Uniform color | color | position, rank, specific suit |
| Sorted by rank | position, rank | suit, color |
| Halves copy suits | position, suit | rank, color |
| At least 3 hearts | suit (specifically ♥) | position, rank, other suits |

The model must learn to **selectively attend** to relevant features for each task.

### Why This Is Hard

1. **No explicit supervision**: We don't tell the model which features are relevant
2. **Feature space is large**: 4 suits × 13 ranks × 6 positions = 312 possible feature combinations per card
3. **Relevant features are sparse**: For any rule, most features are noise
4. **Cross-task interference**: Negative examples look similar across tasks

### What Would Help

1. **Contrastive learning**: Train to distinguish tasks, not just predict primitives
2. **Prototype learning**: Learn what "typical" positive examples look like per task
3. **Feature selection mechanism**: Attention over feature dimensions, not just examples
4. **Minimal-change negatives**: Negative examples that differ minimally from positives

---

## Quantitative Baseline

For reference, here's what a "random baseline" produces:

| Metric | Random Model | Our Best Model | Desired |
|--------|--------------|----------------|---------|
| Task embedding cosine sim | 0.99 | 0.85-0.95 | < 0.5 |
| Prediction entropy std | 0.001 | 0.005 | > 0.5 |
| Top-1 primitive diversity | 1 | 2-3 | 15-20 |
| Enumeration speedup | 1.0x | 1.1x | 2-10x |

---

## Key Insight

The standard "encode examples → predict primitives" pipeline fails because:

1. **The output space is uninformative**: True/False vs structured outputs
2. **The training signal is indirect**: Primitives in solution, not explicit features
3. **The task distribution is adversarial**: Tasks look similar but need different primitives

A new approach must either:
- **Enrich the output signal** (e.g., explain *why* hands are True/False)
- **Learn task-specific feature attention** directly
- **Use meta-learning** to rapidly adapt to new tasks
- **Employ retrieval** from similar solved tasks instead of direct prediction
