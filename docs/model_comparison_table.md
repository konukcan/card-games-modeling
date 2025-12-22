# DreamCoder Model Versions: Comprehensive Comparison

**Created:** December 4, 2024
**Purpose:** Establish a canonical reference for all architectural variations in the DreamCoder codebase

---

## 1. RECOGNITION MODEL ARCHITECTURES

| Component | Current GRU | Legacy GRU | Set Transformer |
|-----------|-----------|-----------|-----------------|
| **File** | `neural_recognition.py` | `neural_recognition_legacy_gru.py` | `set_transformer_recognition.py` |
| **Encoder Type** | Bidirectional GRU | Bidirectional GRU | Self-Attention (Set Transformer) |
| **Hand Encoding** | GRU over per-card MLPs | GRU over per-card MLPs | SAB layers + PMA |
| **Example Encoding** | Concatenate hand + output | Concatenate hand + output | FiLM (Feature-wise Linear Modulation) |
| **Task Encoding** | Attention-weighted pooling | Mean pooling | Raw card feature correlation |
| **Card Feature Dim** | 24 | 24 | 24 |
| **Hidden Dim / d_model** | 128 (default) | 128 (default) | 64 (default) |
| **LayerNorm Usage** | Post-attention & post-FF | Post-attention & post-FF | **Disabled** in SAB/PMA (preserves variance) |
| **Positional Encoding** | None | None | Learnable positional embedding |
| **num_heads** | N/A | N/A | 4 (default) |
| **num_sab_layers** | N/A | N/A | 2 (default) |
| **num_seeds (PMA)** | N/A | N/A | 1 (default) |
| **Dropout** | 0.1 | 0.1 | 0.1 |
| **Optimizer** | Adam (lr=1e-3) | Adam (lr=1e-3) | AdamW (lr=1e-3, weight_decay=0.01) |
| **LR Scheduler** | None | None | ReduceLROnPlateau (factor=0.5, patience=3) |
| **Gradient Clipping** | None | None | Yes (max_norm=1.0) |
| **Max Examples** | 20 | 20 | 20 |
| **Max Cards** | 8 | 8 | 8 |

### Key Architectural Innovations

**Set Transformer Task Encoding (Critical Fix - Dec 3, 2024)**:
- Problem: Learned embeddings (d_model=64) collapse discriminative information
- Solution: Compute correlations on **raw card features** (24-dim), not learned embeddings
- Method:
  1. `raw_correlation = corr(raw_card_features, labels)` (24 dims)
  2. `raw_diff = mean(pos_examples) - mean(neg_examples)` (24 dims)
  3. Concatenate and expand to d_model via identity-initialized projection
- Result: Task embedding cosine similarity range changed from `[0.98, 1.0]` to `[-0.38, 0.49]`

**FiLM Modulation for Example Encoding**:
- Problem: Concatenation of hand + label washes out label signal
- Solution: `output = gamma * hand_emb + beta` where `gamma, beta = f(label)`
- Ensures True/False labels produce fundamentally different representations

---

## 2. EXAMPLE GENERATION

| Parameter | Random Sampling | Balanced Sampling (Fixed) |
|-----------|----------------|---------------------------|
| **Method** | `sample_hand(hand_size)` | Rejection sampling until 50/50 |
| **Training Examples** | 100 | 100 (50 positive, 50 negative) |
| **Holdout Examples** | 20 | 20 (10 positive, 10 negative) |
| **Hand Size** | 5 | 5 |
| **Problem** | Extreme class imbalance for some rules | Fixed |
| **Rules Skipped** | None | 3 (can't balance: `Halves_AP_len3_any_equal`, `Even_opens_next_closes`, `Odd_opens_next_closes`) |

### Base Rate Analysis (Random 5-card Hands)

| Category | Count | Examples |
|----------|-------|----------|
| **>95% True** | 1 | `Halves_AP_len3_any_equal` (100% True) |
| **>95% False** | 18 | `Even_opens_next_closes`, `Ranks_palindrome`, `Shift_half_plus_two`, etc. |
| **Balanced (5-95%)** | 38 | `Has_pair_ranks` (58.5%), `Ends_same_color` (58%), etc. |

---

## 3. ENUMERATION PARAMETERS

| Parameter | Default | Cython | Notes |
|-----------|---------|--------|-------|
| **Algorithm** | Best-first search | Best-first search | Grammar-weighted by log probability |
| **Max Depth** | Phase-dependent (6-15) | 6-15 | Increases with phase |
| **Max Programs** | Phase-dependent (5K-2M) | Phase-dependent | Budget per task |
| **Timeout per Task** | 60-600 sec | 60-600 sec | Increases with phase |
| **Parallel Workers** | 4 | 4 | PyPy subprocesses |
| **Early Pruning** | All-or-nothing mode | Enabled | Skip partial solutions |
| **Speedup vs Baseline** | 1x | ~7-15x (theoretical) | Actual ~1.5x due to pickle serialization |

---

## 4. WAKE-SLEEP CYCLE

| Component | Implementation |
|-----------|----------------|
| **Wake: Enumeration** | Enumerate programs for each task, keep top-k solutions per frontier |
| **Sleep: Compression** | Anti-unification to find common patterns, add as invented primitives |
| **Sleep: Recognition** | Train neural model on solved tasks for N epochs |
| **Sleep: Dreaming** | Generate synthetic tasks from recognition model (optional) |
| **Grammar Update** | Reweight primitives based on solution frequencies |
| **Library Growth** | Typically 0-5 new abstractions per compression |

---

## 5. PRIMITIVES COMPARISON

| Aspect | card_primitives.py | lean_primitives.py | extended_primitives.py |
|--------|-------------------|-------------------|----------------------|
| **Total Count** | ~61 | ~54 (v3) | ~70+ |
| **Philosophy** | Comprehensive baseline | Cognitive realism | Academic rigor |
| **Constants** | 0-6, suits, colors | 0-5, suits, colors | Base lean + extended |
| **Abstract Combinators** | Yes (compose, flip, const, id) | **No** (low cognitive reality) | Yes |
| **List Construction** | Yes (cons, nil) | **No** (not how humans think) | Yes |
| **Game-Relevant Numbers** | No | **No** (removed in v3) | No |

### Lean Primitives Philosophy
- Primitives should be "directly nameable" in short phrases
- Removed: `compose`, `flip`, `const`, `id`, `cons`, `nil`
- Target: ~54 primitives with high cognitive realism

### Lean Primitives v3 Changes (Dec 2024)
**Removed (unused by any of the 45 active rules):**
- Rank constants: `10`, `11`, `12`, `13`, `14` (face card values)
- Game thresholds: `17`, `21` (Blackjack values)
- Comparison: `neq` (use `not (eq x y)` instead)

**Note**: The following were documented in rule_dependency_tree.py but were never in the actual lean_primitives.py library:
- `cons`, `empty`, `tail`, `is_empty`, `foldr` (DreamCoder-style primitives, available in extended_primitives.py)

### Primitives Added (Dec 4, 2024)
- `half_len`: `len(xs) // 2`
- `first_half`: `xs[:len(xs) // 2]`
- `second_half`: `xs[len(xs) // 2:]`

---

## 6. RUNNER CONFIGURATIONS

### run_overnight_set_transformer.py

| Phase | Name | Iterations | Max Depth | Max Programs | Timeout | Recognition Epochs |
|-------|------|-----------|----------|-------------|---------|-------------------|
| 1 | Quick Exploration | 5 | 6 | 5,000 | 60s | 5 |
| 2 | Medium Exploration | 5 | 6 | 10,000 | 90s | 5 |
| 3 | Deep Search | 10 | 7 | 50,000 | 120s | 12 |
| 4 | Intensive Search | 10 | 8 | 100,000 | 180s | 15 |

### run_experimental_rules.py

| Phase | Name | Iterations | Max Depth | Max Programs | Timeout | Recognition Epochs |
|-------|------|-----------|----------|-------------|---------|-------------------|
| 1 | Quick Exploration | 5 | 7 | 100,000 | 60s | 5 |
| 2 | Medium Search | 10 | 9 | 250,000 | 120s | 8 |
| 3 | Deep Search | 15 | 11 | 500,000 | 180s | 10 |
| 4 | Extended Deep Search | 20 | 13 | 1,000,000 | 300s | 10 |
| 5 | Exhaustive Search | 15 | 15 | 2,000,000 | 600s | 15 |

### run_overnight_v4.py (Part A: Standard Batch)

| Phase | Rule Set | Max Depth | Iterations | Recognition Epochs |
|-------|----------|----------|-----------|-------------------|
| 1-2 | Easy pretraining (level 1) | 6 | 5 each | 5 |
| 3-4 | All pretraining | 7-8 | 10 each | 12-15 |
| 5-7 | Experimental rules (transfer) | 8-10 | 10-15 each | 15-25 |

---

## 7. PARAMETER COMPARISON BY RUNNER

| Parameter | v3 | v4 A | v4 B | Set Transformer | Experimental |
|-----------|-------|-------|-------|-----------------|--------------|
| **Recognition Model** | GRU | GRU | GRU | Set Transformer | Set Transformer |
| **Primitives** | lean | lean | lean | lean | lean |
| **Total Phases** | 4 | 7 | 7 | 4 | 5 |
| **Total Iterations** | 30+ | 60+ | 60+ | 30 | 65 |
| **Max Initial Depth** | 6 | 6 | 6 | 6 | 7 |
| **Max Final Depth** | 10 | 10 | 10 | 8 | 15 |
| **Workers** | 4 | 4 | 4 | 4 | 4 |
| **Mini-batch Compression** | No | No | Yes | No | No |
| **Balanced Sampling** | No | No | No | No | **Yes** (fixed) |

---

## 8. KEY FIXES AND LESSONS LEARNED

### Task-Result Scrambling Bug (v3 Fix)
- **Problem**: `as_completed()` returns futures in arbitrary order, scrambling task-result mappings
- **Solution**: Use dictionary keyed by `task.name` instead of list index

### Task Embedding Collapse (Set Transformer Fix - Dec 3, 2024)
- **Problem**: All task embeddings had cosine similarity ~1.0 (no discrimination)
- **Root Cause**: Neural network layers destroy discriminative information present in raw features
- **Solution**: Compute correlations on raw 24-dim card features, not learned embeddings
- **Result**: Cosine similarity range changed from `[0.98, 1.0]` to `[-0.38, 0.49]`

### Trivial Solutions Passing Verification (Dec 4, 2024 Fix)
- **Problem**: `(λ true)` and `(λ false)` passed holdout verification
- **Root Cause**: Random sampling produces extreme class imbalance for some rules
- **Solution**: Balanced sampling (rejection sampling for 50/50 split)
- **Result**: Most rules can be balanced; a few rules skipped (impossible to balance)

---

## 9. FILES REFERENCE

| Component | Primary File | Notes |
|-----------|-------------|-------|
| Current GRU Recognition | `neural_recognition.py` | Attention-weighted task encoding |
| Legacy GRU Recognition | `neural_recognition_legacy_gru.py` | Mean-pooled task encoding |
| Set Transformer Recognition | `set_transformer_recognition.py` | Raw correlation task encoding |
| DreamCoder Framework | `dreamcoder_v2.py` | Core Task, Frontier, Metrics classes |
| Lean Primitives | `lean_primitives.py` | Cognitive realism optimized |
| Extended Primitives | `extended_primitives.py` | Academic (Ellis et al.) style |
| Card Primitives | `card_primitives.py` | Legacy baseline |
| Compression | `compression.py` | Anti-unification |
| Enumeration | `enumeration.py` | Best-first search |

---

## 10. CHECKLIST FOR NEW EXPERIMENTS

When running a new experiment, verify:

- [ ] **Example Sampling**: Balanced or random? (Use balanced for holdout verification)
- [ ] **Recognition Model**: GRU or Set Transformer?
- [ ] **Primitives**: lean, extended, or card?
- [ ] **Phase Configuration**: Depth, programs, timeout, epochs per phase
- [ ] **Rule Set**: Pretraining (43), experimental (57), or subset?
- [ ] **Workers**: 4 PyPy workers configured?
- [ ] **Checkpointing**: Per-iteration or per-phase?
- [ ] **Transfer**: Pretraining → experimental, or direct evaluation?

---

## APPENDIX: Rule Base Rates (Random 5-Card Sampling)

Rules with extreme base rates (>95% or <5%) require balanced sampling:

| Rule | Base Rate | Category |
|------|-----------|----------|
| `Halves_AP_len3_any_equal` | 100% True | Cannot balance |
| `Even_opens_next_closes` | 0% True | Cannot balance |
| `Odd_opens_next_closes` | 0% True | Cannot balance |
| `Ranks_palindrome` | 0% True | Balanceable with effort |
| `Shift_half_plus_two` | 0% True | Partially balanceable |
| `Shift2_plus3` | 0% True | Partially balanceable |
| `Uniform_color` | 4.5% True | Balanceable |
| `Sorted_by_rank` | 2.5% True | Balanceable |
