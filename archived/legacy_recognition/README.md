# Legacy Recognition Models (Archived)

This directory contains recognition model architectures that were tested but **did not work well** for the card game domain. They are preserved for reference but should **not be used** for new experiments.

**Current Primary Model**: `src/dreamcoder_core/contrastive_recognition.py`

---

## Archived Models

### 1. neural_recognition.py (GRU-based DreamCoder Model)

**Architecture**:
- Bidirectional GRU encoder over serialized card features
- Attention-weighted pooling over examples
- Softmax output over primitives
- Cross-entropy loss

**Why It Was Superseded**:

1. **Sequential processing is suboptimal for cards**: The GRU processes cards sequentially, but card hands are fundamentally **sets** - the order shouldn't matter for most rules. The model learns spurious positional patterns.

2. **No contrastive signal**: The model encodes tasks by pooling example embeddings, but doesn't explicitly capture **what distinguishes positive from negative examples**.

3. **Limited transfer**: In warmstart experiments, achieved only +2 rule transfer (13.3% solve rate) compared to contrastive model with proper tuning.

**Historical Results** (December 2024):
```
WARM condition: 6/45 (13.3%) solve rate
Recognition loss: 1.97 (after warmstart)
Transfer: +2 rules (Ranks_palindrome, Halves_copy_ranks)
```

**Documentation**: See `src/docs/recognition_model_search.md` for full experiment details.

---

### 2. set_transformer_recognition.py (Set Transformer Model)

**Architecture**:
- Self-attention encoder (Set Attention Blocks)
- Pooling by Multihead Attention (PMA)
- Learned positional encodings
- FiLM modulation for example encoding

**Why It Was Superseded**:

1. **Embedding collapse problem**: All task embeddings had cosine similarity ≈ 1.0. The model couldn't distinguish between different tasks - it saw every task as identical.

2. **Neural layers destroyed discriminative information**: The raw 24-dimensional card features contained discriminative information (correlation range [-0.44, 0.51] between features and labels), but multiple neural network layers collapsed this diversity.

3. **LayerNorm exacerbated collapse**: When inputs have similar structure (all 5-card hands), LayerNorm normalizes away differences.

4. **Fix attempt didn't fully resolve issues**: We tried using raw feature correlations instead of learned embeddings, which improved diversity but still underperformed contrastive model.

**Historical Results** (December 2024):
```
Before fix: All task embeddings had cosine similarity ≈ 0.9999
After fix:  Cosine similarity range [-0.38, 0.49] - improved but still worse than contrastive
```

**Documentation**: See `docs/set_transformer_discrimination_fix_2024_12_03.md` for detailed investigation.

---

## Why ContrastiveRecognitionModel is Primary

### Key Innovation: τ = mean(pos) - mean(neg)

The contrastive model encodes tasks by:
1. Computing mean embedding of **positive examples** (hands satisfying the rule)
2. Computing mean embedding of **negative examples** (hands not satisfying)
3. Taking the **difference**: τ = mean(pos) - mean(neg)

This directly captures **what distinguishes positive from negative examples** - the decision boundary itself.

### Factored Card Embeddings

Instead of flattening card features, the model uses **learned embeddings**:
```
Card → E_suit(suit) ⊕ E_rank(rank) ⊕ E_position(pos)
```

This is more parameter-efficient and captures card structure naturally.

### Output Mode Flexibility

The model supports both:
- `output_mode='sigmoid'`: Independent probabilities per primitive (good for classification)
- `output_mode='softmax'`: Probability distribution over primitives (good for search ranking)

**Key finding**: Softmax output is critical for search guidance because the enumerator needs **ranking**, not just classification.

### Additional Features
- **Count head**: Predicts number of primitives needed
- **Bigram head**: Predicts primitive co-occurrence patterns
- **Structural similarity loss**: Tasks with similar primitives should have similar embeddings

---

## Deleted Files

### contrastive_recognition_v1_baseline.py

This was an older snapshot of `contrastive_recognition.py` without:
- Bigram loss support
- `evaluate_count_head()` method
- Other evaluation methods

It was explicitly marked "TODO: Consider removing" and served no purpose since the main contrastive model supersedes it.

---

## Files That Still Reference These Models

**Warning**: The following files may have broken imports after this archival:

| File | Import | Status |
|------|--------|--------|
| `run_overnight_v3.py` | `NeuralRecognitionModel` | OUTDATED - needs update |
| `resume_overnight_v3.py` | `NeuralRecognitionModel` | OUTDATED - needs update |
| `dreamcoder_original.py` | `NeuralRecognitionModel` | OK - reference implementation |
| `run_experimental_rules.py` | `SetTransformerRecognitionModel` | OUTDATED - needs update |
| `run_overnight_set_transformer.py` | `SetTransformerRecognitionModel` | OUTDATED - should be archived |
| `experiments/run_warmstart_experiment.py` | Both | OK - comparison script |
| `experiments/run_recognition_dream_experiment.py` | Both | OK - comparison script |

Scripts marked "OUTDATED" should either be:
1. Updated to use `ContrastiveRecognitionModel`
2. Archived if they're no longer needed

---

## Key Lessons Learned

### 1. Search Guidance Requires Ranking, Not Classification
- Sigmoid outputs give binary "useful/not useful" predictions
- Softmax outputs give a **ranking** of primitives to try first
- The enumerator needs ranking → always use softmax for search

### 2. Neural Networks Can Destroy Information
- Raw card features contain discriminative information
- Multiple neural layers can collapse this diversity
- Sometimes simpler encodings work better

### 3. The Contrastive Signal is Critical
- τ = mean(pos) - mean(neg) directly captures the decision boundary
- This is more informative than just pooling all examples together
- Combined with factored embeddings and softmax output, this gives best results

---

## References

- `src/docs/recognition_model_search.md` - Full experiment timeline
- `docs/set_transformer_discrimination_fix_2024_12_03.md` - Set Transformer investigation
- `src/docs/recognition_model_variants_report.md` - Architectural variant comparison
- `docs/embedding_fix_report.md` - Embedding magnitude fixes

---

*Archived: January 2025*
