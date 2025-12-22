# Set Transformer Task Discrimination Fix

**Date:** December 3, 2024
**Problem:** Set Transformer recognition model produced identical embeddings for all tasks
**Status:** FIXED

---

## Executive Summary

The Set Transformer recognition model was failing to discriminate between different tasks—all task embeddings had cosine similarity ≈ 1.0000, meaning the model saw every task as identical. After systematic investigation, I discovered that **learned neural network layers were collapsing discriminative information** present in the raw input features. The fix was to compute task signatures directly from **raw card feature correlations** rather than learned embeddings.

---

## 1. Problem Discovery

### Initial Observation
When checking overnight run checkpoints, I found alarming metrics:
```
mean_similarity: 1.0000
top_primitive_diversity: 1
```

This meant:
- **All task embeddings were identical** (cosine similarity = 1.0)
- **All tasks received the same primitive prediction** (diversity = 1)

This was the **exact same problem** we had with the legacy GRU model—the recognition model had no discriminative power.

---

## 2. Investigation Steps

### Step 1: Verify the Problem
First, I encoded several different tasks and computed pairwise cosine similarities:

```python
# Example tasks: poker_has_pair, poker_flush, poker_same_color, etc.
for task1, task2 in pairs:
    cos_sim = cosine_similarity(embed(task1), embed(task2))
    # Result: ALL were 0.9999-1.0000
```

**Finding:** Confirmed—all task embeddings were essentially identical.

### Step 2: Trace Through the Architecture

The Set Transformer has a hierarchical encoding pipeline:

```
Cards → HandEncoder → ExampleEncoder → TaskEncoder → PrimitivePredictor
 (24)      (64)           (64)            (64)           (67)
```

I added diagnostic prints at each stage to find where diversity was lost.

### Step 3: Check Hand Embeddings

```python
# Encode the same hand twice
hand_emb1 = hand_encoder(hand_features)
hand_emb2 = hand_encoder(hand_features)
# Cosine: 1.0000 (expected—same input)

# Encode different hands
hand_emb_A = hand_encoder(hand_A_features)
hand_emb_B = hand_encoder(hand_B_features)
# Cosine: 0.998+ (PROBLEM!)
```

**Finding:** Different hands produced nearly identical embeddings (cosine > 0.998).

### Step 4: Identify LayerNorm as Culprit

The Set Transformer uses LayerNorm after attention blocks. LayerNorm normalizes each sample to:
- Mean = 0
- Standard deviation = 1

When inputs have **similar structure** (all 5-card hands), LayerNorm normalizes away the differences, making outputs nearly identical.

### Step 5: Attempt Fix #1 - Disable LayerNorm

```python
# Changed SAB and PMA to use_layernorm=False
class SetAttentionBlock:
    def __init__(self, ..., use_layernorm=False):
        ...
```

**Result:** Still 1.0000 cosine similarity. LayerNorm wasn't the only problem.

### Step 6: Attempt Fix #2 - FiLM Modulation

Changed the ExampleEncoder from concatenation to Feature-wise Linear Modulation (FiLM):

```python
# Before: output = MLP(concat(hand_emb, label_emb))
# After:  output = gamma * hand_emb + beta
#         where gamma, beta = f(label_emb)
```

**Result:** Example embeddings improved (0.97 → 0.86 cosine for same hand with different labels), but task embeddings still 1.0000.

### Step 7: Attempt Fix #3 - Contrastive Pooling

Created separate PMA (Pooling by Multihead Attention) for positive and negative examples:

```python
pos_emb = pma_pos(positive_examples)
neg_emb = pma_neg(negative_examples)
task_emb = pos_emb - neg_emb  # Contrastive signal
```

**Result:** Still 1.0000. The learned PMA was collapsing diversity.

### Step 8: Attempt Fix #4 - Feature-Label Correlation

Computed Pearson correlation between embedding dimensions and labels:

```python
# For each embedding dimension d:
correlation[d] = corr(embeddings[:, d], labels)
```

**Result:** Correlation vectors had 0.98-0.99 cosine similarity across tasks. The **learned embeddings** (d_model=64) weren't preserving discriminative information.

### Step 9: Key Insight - Raw Features ARE Discriminative

I computed correlations on the **raw 24-dimensional card features** instead of learned embeddings:

```python
# Raw card features: suit (4), rank (13), color (2), properties (5) = 24 dims
raw_correlation = corr(raw_card_features, labels)
```

**Result:** Raw correlation vectors had diverse cosines: **range [-0.44, 0.51]**!

This proved that discriminative information **exists in the input** but was being destroyed by the neural network layers.

---

## 3. Root Cause Analysis

### Why Learned Embeddings Collapse Diversity

1. **LayerNorm Normalization**: Each sample gets normalized to mean=0, std=1, reducing variance between samples with similar structure.

2. **Attention Averaging**: Self-attention tends to average features across positions, smoothing out differences.

3. **Non-linear Compression**: Multiple MLP layers with GELU activations can map diverse inputs to similar outputs if not carefully initialized.

4. **Random Initialization**: Randomly initialized projection layers can accidentally map diverse inputs to similar regions of the output space.

### The Core Problem

```
Raw features (24-dim) → [Many neural layers] → Learned embeddings (64-dim)
    Diverse!                                        All similar!
```

The neural network was acting as an **information bottleneck** that squeezed out the task-discriminative signal.

---

## 4. The Solution

### Approach: Bypass Learned Embeddings

Instead of computing correlations on learned embeddings, compute them directly on raw card features:

```python
class SetTransformerTaskEncoder:
    def forward(self, ..., raw_features):
        # Compute correlation between RAW features and labels
        raw_correlation = corr(raw_features, labels)  # (24,)

        # Compute prototype difference
        raw_diff = mean(pos_examples) - mean(neg_examples)  # (24,)

        # Concatenate: [correlation, diff] = 48 dims
        raw_combined = concat(raw_correlation, raw_diff)

        # Expand to d_model with identity-initialized projection
        return self.expand(raw_combined)  # (64,)
```

### Key Design Decisions

1. **Use Raw Features**: Compute correlations on 24-dim card features, not 64-dim learned embeddings.

2. **Identity Initialization**: Initialize the expansion layer as near-identity to preserve the raw signal:
   ```python
   with torch.no_grad():
       self.expand.weight.zero_()
       for i in range(48):
           self.expand.weight[i, i] = 1.0  # Identity for first 48 dims
   ```

3. **Simple Architecture**: Avoid non-linear layers that could collapse diversity.

4. **Correlation + Difference**: Use both feature-label correlation (which features predict the label) and prototype difference (how positive/negative examples differ on average).

### Updated encode_task Method

```python
def encode_task(self, task):
    example_embeddings = []
    raw_feature_list = []
    labels = []

    for inp, out in task.examples:
        # Get learned embedding (still used for other purposes)
        example_emb = self.example_encoder(hand_features, label_features)
        example_embeddings.append(example_emb)

        # Store RAW features: mean-pool card features for this hand
        raw_hand_feat = hand_features[:num_cards].mean(dim=0)  # (24,)
        raw_feature_list.append(raw_hand_feat)
        labels.append(out == True)

    # Pass raw features to task encoder
    task_emb = self.task_encoder(
        stacked_embeddings,
        example_mask,
        label_mask,
        raw_features=raw_stacked  # NEW: pass raw features
    )
    return task_emb
```

---

## 5. Results

### Before Fix
```
Task embedding cosine: mean=0.9927, std=0.0047, range=[0.9816, 0.9985]
```
All tasks looked identical.

### After Fix
```
Task embedding cosine: mean=0.1142, std=0.2561, range=[-0.3845, 0.4861]
```

- **std increased from 0.0047 → 0.2561** (54x improvement!)
- **Range now includes negative correlations** (-0.38 to 0.49)
- Different tasks now have genuinely different embeddings

### Sample Pairwise Similarities (After Fix)
```
Sorted_by_rank vs Ends_same_suit: 0.3162
Sorted_by_rank vs Uniform_color: 0.1331
Uniform_color vs Has_Ace_of_Spades: -0.0009
Uniform_color vs Has_pair_ranks: -0.3845
Has_Ace_of_Spades vs Has_pair_ranks: 0.4678
```

Tasks are now discriminable!

---

## 6. Why This Works

### Raw Card Features Capture Rule Semantics

Each rule creates a different pattern of feature-label correlation:

| Rule | High Correlation Features |
|------|--------------------------|
| Uniform_color | Color one-hot (dims 17-18) |
| Has_Ace | Rank=Ace one-hot (dim 16) |
| Sorted_by_rank | Rank value (dim 19) |
| Ends_same_suit | Suit one-hots (dims 0-3) |

These correlations are **directly observable** in raw features but get **washed out** by learned transformations.

### Identity Initialization Preserves Signal

By initializing the expansion layer as identity:
```
output[0:48] ≈ input[0:48]  (correlation + diff)
output[48:64] = small_random_noise
```

The discriminative signal passes through unchanged, allowing the model to learn refinements without destroying the base signal.

---

## 7. Lessons Learned

1. **Neural Networks Can Destroy Information**: More layers ≠ better. Deep networks can collapse discriminative features.

2. **Check Intermediate Representations**: When debugging, trace information flow through every layer.

3. **Raw Features Matter**: Sometimes hand-crafted features (correlation, difference) outperform learned representations.

4. **Initialization is Critical**: Random initialization can accidentally project diverse inputs to similar outputs.

5. **LayerNorm is a Double-Edged Sword**: Great for training stability, but can destroy between-sample variance.

---

## 8. Files Modified

1. **`set_transformer_recognition.py`**:
   - `SetTransformerTaskEncoder`: Rewrote to use raw feature correlations
   - `encode_task`: Added raw feature extraction and passing
   - Added identity initialization for expansion layer

2. **`run_experimental_rules.py`**: Created new script for experimental rules

---

## 9. Future Improvements

1. **Learn to Weight Correlations**: Add a learnable layer that combines raw correlations, initialized near identity but trainable.

2. **Per-Position Features**: Instead of mean-pooling cards, keep position information for position-sensitive rules.

3. **Correlation + Attention Hybrid**: Use raw correlations as bias terms in a learned attention mechanism.

4. **Regularization**: Add loss term to encourage task embedding diversity during training.

---

## Appendix: Diagnostic Code

### Quick Test for Task Discrimination
```python
def test_discrimination(model, tasks):
    embeddings = [model.encode_task(t) for t in tasks]
    cos_sims = []
    for i, e1 in enumerate(embeddings):
        for e2 in embeddings[i+1:]:
            cos_sims.append(F.cosine_similarity(e1, e2, dim=0).item())

    print(f"Cosine: mean={np.mean(cos_sims):.4f}, std={np.std(cos_sims):.4f}")
    if np.std(cos_sims) > 0.1:
        print("✓ Model is discriminative")
    else:
        print("✗ Model is NOT discriminative")
```

### Check Raw vs Learned Correlations
```python
def compare_correlations(task, model):
    raw_corr = compute_raw_correlation(task)  # 24-dim
    learned_corr = compute_learned_correlation(task, model)  # 64-dim

    print(f"Raw correlation range: [{raw_corr.min():.3f}, {raw_corr.max():.3f}]")
    print(f"Learned correlation range: [{learned_corr.min():.3f}, {learned_corr.max():.3f}]")
```
