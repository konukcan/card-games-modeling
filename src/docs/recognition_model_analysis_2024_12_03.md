# Recognition Model Analysis - December 3, 2024

This document records the comprehensive analysis of the DreamCoder recognition model issues discovered during the overnight run debrief, along with proposed solutions.

## Table of Contents
1. [Bug Findings](#bug-findings)
2. [Architecture Comparison with Original DreamCoder](#architecture-comparison)
3. [Contrastive Learning Pipeline](#contrastive-learning-pipeline)
4. [Set Classification Architectures](#set-classification-architectures)
5. [Minimal-Change Negative Examples](#minimal-change-negative-examples)
6. [Recommendations](#recommendations)

---

## Bug Findings

### Finding I: Embedding Cache Bug (FIXED)

**Location:** `neural_recognition.py:677-682`

**The Bug:** `get_task_embedding()` used a cache (`self._task_embeddings`) that was **never cleared** between iterations. Once an embedding was computed for a task in iteration 1, it was returned unchanged for all subsequent iterations, even after model weights had been updated through training.

**Evidence:**
```
Iter 1:  first 5 vals = [ 1.405   0.7349 -0.3105  0.4909  0.6192]
Iter 21: first 5 vals = [ 1.405   0.7349 -0.3105  0.4909  0.6192]
Cosine similarity = 1.0, Max difference = 0.0
```

**Fix Applied:**
- Changed `get_task_embedding()` default to `use_cache=False`
- Added `clear_embedding_cache()` method
- Automatically clear cache after `train_on_frontiers()` completes

### Finding II: All Tasks Get Nearly Identical Predictions

**Evidence:** Within any single iteration, all tasks produce nearly identical primitive probability distributions:

| Metric | Value | Expected |
|--------|-------|----------|
| Entropy std across tasks | 0.0005 - 0.005 | Should be much higher |
| Max prob difference between tasks | ~0.01-0.03 | Should be significant |
| Top primitive | Same for all tasks | Should vary by task |

**This is NOT a recording bug** - it's a fundamental limitation of the architecture.

---

## Architecture Comparison

### Original DreamCoder (List Tasks)

| Aspect | Implementation |
|--------|---------------|
| Input encoding | Tokenized symbols (0-9, etc.) through shared vocabulary |
| Output encoding | **Tokenized LIST as sequence** (e.g., `[6]` for sum result) |
| Sequence structure | `[1,2,3,ENDOFINPUT,6,ENDING]` - output is part of sequence |
| Hidden dim | 32 (small!) |
| Example aggregation | Simple mean pooling |

**Key insight:** In DreamCoder's list tasks, the **output is informative** - it's the *result* of applying the program to the input (e.g., `[1,2,3] → [6]` for a sum function). The GRU learns to recognize "when inputs map to this kind of output, these primitives are useful."

### Our Implementation (Card Tasks)

| Aspect | Implementation |
|--------|---------------|
| Input encoding | Per-card 24-dim one-hot features |
| Output encoding | **Just True/False (2 bits!)** |
| Sequence structure | Cards only; output is separate binary label |
| Hidden dim | 256 (large) |
| Example aggregation | Attention-weighted pooling |

**The critical difference:** Our output is just a **classification label** (True/False). The network sees:
- Hand A → True
- Hand B → False

But it has no way to learn *why* Hand A is True and Hand B is False from the labels alone.

### Why This Causes Collapse

Even with a fresh, untrained model:
- Cosine similarity between task encodings: 0.95-0.99
- Prediction cosine similarity: 1.0000 (identical!)

The architecture itself collapses task representations before training even begins because:

1. **Weak output signal:** True/False labels provide minimal information about *which primitives* to use
2. **Cross-task confusion:** Negative examples from different tasks are indistinguishable (all "random hands")
3. **Architecture mismatch:** GRU encodes cards, but True/False label is just 2-dim vector combined late

---

## Contrastive Learning Pipeline

### Original DreamCoder Pipeline

```
TRAINING (Wake-Sleep):
  For each solved task (T, program):
    Input: Examples of task T
    Target: Primitives used in program
    Loss: Cross-entropy

INFERENCE:
  New task T' → Recognition model Q → Primitive probabilities → Reweight grammar
```

This fails for us because True/False carries only 2 bits of information.

### Contrastive Approach

Instead of predicting primitives directly, train the model to **embed tasks into a space where similar tasks cluster together**.

```
TRAINING:
  Learn embedding f(task) such that:
  - f(Task_A with examples E₁) ≈ f(Task_A with examples E₂)  [same task]
  - f(Task_A) ≠ f(Task_B)                                     [different tasks]

  Loss: InfoNCE / Triplet loss pushing same-task together, different-task apart

INFERENCE:
  1. Embed new task: z = f(T')
  2. Find k-nearest solved tasks in embedding space
  3. Aggregate primitives from similar solved tasks weighted by similarity
  4. Use aggregated probabilities to guide enumeration
```

**Advantages:**
- Even if True/False labels are uninformative individually, the **joint distribution** of which hands are True vs False is task-specific
- Model learns to detect this pattern by being trained to distinguish tasks
- Can transfer knowledge from similar solved tasks without explicit primitive prediction

### Prototype Learning Variant

Simpler alternative:

```
For each task T:
  1. Embed all TRUE examples: {f(hand) : (hand, True) ∈ T}
  2. Compute prototype: μ_T = mean of TRUE embeddings
  3. Store (T, μ_T, solution(T)) in database

For new task T':
  1. Compute μ_T' from TRUE examples
  2. Find nearest prototype
  3. Use primitives from that solution to guide search
```

---

## Set Classification Architectures

### 1. DeepSets (Zaheer et al., 2017)

**Core principle:** Any permutation-invariant set function decomposes as:
```
f({x₁, ..., xₙ}) = ρ(Σᵢ φ(xᵢ))
```

**Architecture:**
```python
class DeepSetsCardEncoder(nn.Module):
    def __init__(self, card_dim=24, hidden_dim=64):
        self.phi = MLP(card_dim → hidden_dim)  # Per-card encoder
        self.rho = MLP(hidden_dim → output_dim)  # Set decoder

    def forward(self, hand):
        card_embs = self.phi(hand)  # (batch, num_cards, hidden_dim)
        set_emb = card_embs.sum(dim=1)  # Permutation-invariant
        return self.rho(set_emb)
```

**Pros:** Simple, guaranteed permutation invariance, universal approximator
**Cons:** No inter-element interactions, loses count information

**Good for:** "Contains Ace of Spades", "All same color", "Exactly two suits"
**Bad for:** "Ends same suit" (requires position), "Sorted by rank" (requires order)

### 2. Set Transformer (Lee et al., 2019)

**Core insight:** Use self-attention to model interactions between set elements, with learnable "seed" vectors for pooling.

**Components:**
- **SAB (Set Attention Block):** Self-attention over cards - each card attends to all others
- **PMA (Pooling by Multihead Attention):** Learnable "seed" vectors query the set

**Architecture:**
```python
class SetTransformerCardEncoder(nn.Module):
    def __init__(self, card_dim=24, hidden_dim=64, num_heads=4, num_seeds=1):
        self.card_embed = nn.Linear(card_dim, hidden_dim)
        self.sab1 = SetAttentionBlock(hidden_dim, num_heads)
        self.sab2 = SetAttentionBlock(hidden_dim, num_heads)
        self.seed = nn.Parameter(torch.randn(num_seeds, hidden_dim))
        self.pma = PoolingByMultiheadAttention(hidden_dim, num_heads, num_seeds)

    def forward(self, hand):
        x = self.card_embed(hand)
        x = self.sab1(x)  # Card-card interactions
        x = self.sab2(x)
        x = self.pma(x)   # Pool with learnable seeds
        return x.flatten(start_dim=1)
```

**What attention learns:**
- "Ends same suit": First card attends strongly to last card
- "Sorted by rank": Each card attends to neighbors
- "Palindrome colors": Cards attend to mirror positions

**Pros:** Models pairwise interactions, learnable pooling, captures relational patterns
**Cons:** O(n²) complexity (fine for 6 cards), needs more training data

**Position encoding:** Add positional encoding for position-sensitive rules:
```python
x = self.card_embed(hand) + positional_encoding(num_cards)
```

### 3. Graph Neural Networks

**Core insight:** Treat hand as graph where cards are nodes and edges represent relationships.

**Graph construction options:**
- **Sequential:** Edges (1,2), (2,3), ... for position-based rules
- **Fully connected:** All pairs for any-pair rules
- **Feature-based:** Connect cards sharing suit/color/rank

**Architecture:**
```python
class GNNCardEncoder(nn.Module):
    def __init__(self, card_dim=24, hidden_dim=64, num_layers=3):
        self.card_embed = nn.Linear(card_dim, hidden_dim)
        self.convs = [GNNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]

    def forward(self, hand, edge_index):
        x = self.card_embed(hand)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x.mean(dim=1)  # Global pooling
```

**Pros:** Explicitly models card-to-card relationships, flexible structure
**Cons:** Need to choose graph structure, more complex

### Recommended: Hierarchical Set Transformer

For our task (set of examples, each example is set of cards):

```python
class HierarchicalSetRecognition(nn.Module):
    def __init__(self):
        # Level 1: Encode each hand (set of cards)
        self.hand_encoder = SetTransformer(
            card_dim=24, hidden_dim=64, num_heads=4,
            use_positional=True  # KEY: add position for ordered rules
        )

        # Level 2: Encode task (set of examples)
        self.task_encoder = SetTransformer(
            input_dim=66,  # hand_emb + label
            hidden_dim=64, num_heads=4,
            use_positional=False  # examples are unordered
        )

        # Primitive predictor
        self.primitive_head = MLP(64 → num_primitives)
```

---

## Minimal-Change Negative Examples

### Current Setup: Random Negatives

```
Rule: "Uniform color"
Positive: [♠K, ♣7, ♠2, ♣Q, ♠5, ♣9]  ← all black
Negative: [♠K, ♦3, ♣7, ♥2, ♠Q, ♦9]  ← random (many violations)
```

### Proposed: Minimal-Change Negatives

```
Rule: "Uniform color"
Positive: [♠K, ♣7, ♠2, ♣Q, ♠5, ♣9]  ← all black
Negative: [♠K, ♣7, ♥2, ♣Q, ♠5, ♣9]  ← ONE card changed (♠2 → ♥2)
```

### Impact Analysis

#### On Enumeration

| Aspect | Random Negatives | Minimal-Change |
|--------|-----------------|----------------|
| Speed | Fast (many programs work) | Slower (fewer work) |
| Solution quality | May be spurious | Forced to be correct |
| Generalization | Poor | Good |

**Example:** For "Uniform_color", the spurious program `λh. ge (count_color h BLACK) 4`:
- Random negative (3 black): 3 ≥ 4 → False ✓ (correct by accident)
- Minimal-change (5 black): 5 ≥ 4 → True ✗ (exposed as wrong)

#### On Recognition Model

| Aspect | Random Negatives | Minimal-Change |
|--------|-----------------|----------------|
| Signal strength | Weak (FALSE≈random) | Strong (FALSE≈almost TRUE) |
| Task discrimination | Poor (all tasks alike) | Good (task-specific failures) |
| What model learns | "Valid has structure" | "THIS is what breaks validity" |

**Key insight:** Minimal-change negatives encode task-specific failure modes:
- "Uniform_color" FALSE differs by COLOR
- "Ends_same_suit" FALSE differs by POSITION
- Model can learn "color failure → color primitives useful"

#### On Abstraction Learning

| Aspect | Random Negatives | Minimal-Change |
|--------|-----------------|----------------|
| Abstraction quality | Shallow patterns | Deep structural patterns |
| What's learned | "Reject random" | "Minimal distinguishing conditions" |
| Transfer | Limited | Strong (shared structure) |

### Recommended Hybrid Approach

- 50% minimal-change negatives (forces precision)
- 30% moderate negatives (2-3 cards changed)
- 20% random negatives (ensures robustness)

---

## Recommendations

### Immediate Actions

1. ✅ **Fixed embedding cache bug** - Embeddings now refresh after training

2. **Implement Set Transformer architecture** for better task discrimination

3. **Add halves primitives** to enable learning rules like "Halves_copy_colors"

### Medium-Term Improvements

4. **Implement minimal-change negative generation** for stronger training signal

5. **Add contrastive learning objective** for task embedding

6. **Consider prototype-based retrieval** for primitive prediction

### Architecture Changes

7. **Replace GRU with Set Transformer** for hand encoding

8. **Add positional encoding** for position-sensitive rules

9. **Use hierarchical set encoding** (cards → hand → task)

---

## References

- Zaheer et al. (2017). "Deep Sets" - Permutation-invariant neural networks
- Lee et al. (2019). "Set Transformer" - Attention-based set encoding
- Ellis et al. (2021). "DreamCoder" - Original recognition model architecture
- [DreamCoder GitHub](https://github.com/ellisk42/ec)
- [DreamCoder recognition.py](https://github.com/CatherineWong/dreamcoder/blob/main/recognition.py)

---

*Document created: December 3, 2024*
*Related commits: fix: clear embedding cache after training*
