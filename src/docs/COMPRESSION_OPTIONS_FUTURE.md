# Future Compression Ablation Options

This document records compression algorithm options that are **implemented but not yet tested** in ablation studies. These are lower priority than the recognition-guided compression ablation but should be tested in future experiments.

## 1. Beam Search Compression

**File**: `dreamcoder_core/compression/compress.py` → `beam_search_compression()`

**Current Default**: Greedy selection (always pick best candidate, repeat)

**What Beam Search Does**:
- Maintains `beam_width` best compression states in parallel
- Explores adding each candidate to each state
- Keeps best overall results
- Avoids local optima that greedy can get stuck in

**Key Parameters**:
```python
beam_search_compression(
    grammar=grammar,
    frontiers=frontiers,
    request_type=arrow(HAND, BOOL),
    beam_width=10,              # How many parallel paths to explore
    max_inventions=5,           # Total abstractions to find
    grammar_weight=1.0,         # MDL parameter
    candidates_per_state=20,    # Candidates to try from each state
    use_arity_search=False      # See below
)
```

**Expected Impact**:
- May find better abstraction combinations
- Especially useful when abstraction order matters for hierarchical patterns
- Trade-off: O(beam_width × candidates) slower than greedy

**Recommended Ablation**:
```python
variants = {
    'greedy': lambda: compress_frontiers(...),  # Current default
    'beam_5': lambda: beam_search_compression(..., beam_width=5),
    'beam_10': lambda: beam_search_compression(..., beam_width=10),
}
```

---

## 2. Arity-Aware Factorization

**File**: `dreamcoder_core/compression/arity_search.py` → `best_factorization()`

**Current Default**: Abstract over ALL free variables

**What Arity Search Does**:
For pattern `(mod (sum_ranks $0) 2)`:
- Option 1: `#mod_sum_2 = λh. (mod (sum_ranks h) 2)` — 1 arg, specific to mod 2
- Option 2: `#mod_sum = λn.λh. (mod (sum_ranks h) n)` — 2 args, generalizes to any modulus

Arity search evaluates ALL factorizations by MDL and picks the best one.

**Integration via Beam Search**:
```python
beam_search_compression_with_arity(
    grammar=grammar,
    frontiers=frontiers,
    request_type=arrow(HAND, BOOL),
    beam_width=10,
    max_inventions=5,
    max_args=4                  # Maximum arity to consider
)
```

**Expected Impact**:
- May find more reusable abstractions (lower arity)
- Or more specific abstractions (higher arity) depending on corpus
- Trade-off: O(2^n) factorizations per candidate where n = free variables

**Recommended Ablation**:
```python
variants = {
    'full_arity': lambda: compress_frontiers(...),  # Current
    'arity_aware_3': lambda: beam_search_compression_with_arity(..., max_args=3),
    'arity_aware_4': lambda: beam_search_compression_with_arity(..., max_args=4),
}
```

---

## 3. MDL-Based Compression

**File**: `dreamcoder_core/compression/compress.py` → `compress_frontiers_mdl()`

**Current Default**: Heuristic scoring `(size-1) × (count-1)`

**What MDL Scoring Does**:
```
MDL = λ × DL(grammar) + Σ DL(program_i | grammar)
```
- Computes actual description length change
- Accounts for grammar expansion cost
- `grammar_weight` controls conservativeness

**Key Parameters**:
```python
compress_frontiers_mdl(
    grammar=grammar,
    frontiers=frontiers,
    request_type=arrow(HAND, BOOL),
    max_inventions=5,
    grammar_weight=1.0,         # Higher = more conservative
    min_mdl_improvement=0.0     # Minimum improvement to accept
)
```

**Expected Impact**:
- More principled selection (no marginal abstractions)
- `grammar_weight > 1`: Very conservative (only clearly beneficial)
- `grammar_weight < 1`: Aggressive (add more abstractions)

**Recommended Ablation**:
```python
variants = {
    'heuristic': lambda: compress_frontiers(...),  # Current
    'mdl_balanced': lambda: compress_frontiers_mdl(..., grammar_weight=1.0),
    'mdl_aggressive': lambda: compress_frontiers_mdl(..., grammar_weight=0.7),
    'mdl_conservative': lambda: compress_frontiers_mdl(..., grammar_weight=1.5),
}
```

---

## 4. Iterative Compression

**File**: `dreamcoder_core/compression/compress.py` → `iterative_compression()`

**Current Default**: Single-pass compression per wake-sleep iteration

**What Iterative Compression Does**:
- Multiple rounds of compression within one call
- After each round, rewrites programs with new abstractions
- Finds patterns in the rewritten programs

**Example**:
```
Round 1: #inc = λ.(+ $0 1)
Round 2 (on rewritten programs): #add2 = λ.(#inc (#inc $0))
```

**Key Parameters**:
```python
iterative_compression(
    grammar=grammar,
    frontiers=frontiers,
    max_rounds=3,                   # Compression rounds
    max_inventions_per_round=3,     # Cap per round
    refactor_programs=True          # Rewrite between rounds
)
```

**Expected Impact**:
- Deeper hierarchical abstractions
- May find composition patterns missed in single pass
- Trade-off: Longer compression time

**Recommended Ablation**:
```python
variants = {
    'single_pass': lambda: compress_frontiers(...),  # Current
    'iterative_2': lambda: iterative_compression(..., max_rounds=2),
    'iterative_3': lambda: iterative_compression(..., max_rounds=3),
}
```

---

## Priority Ordering for Future Experiments

1. **Recognition-Guided Compression** (alpha tuning) ← DOING NOW
2. **MDL vs. Heuristic Scoring** ← Most principled change
3. **Iterative Compression** ← May find hierarchical patterns
4. **Beam Search** ← Avoids local optima
5. **Arity-Aware Search** ← Fine-grained optimization

---

## Code Locations

| Component | File | Function |
|-----------|------|----------|
| Beam Search | `compression/compress.py` | `beam_search_compression()` |
| Arity Search | `compression/arity_search.py` | `best_factorization()` |
| MDL Scoring | `compression/mdl_scoring.py` | `compute_mdl()`, `evaluate_invention_mdl()` |
| Iterative | `compression/compress.py` | `iterative_compression()` |
| Combined | `compression/compress.py` | `beam_search_compression_with_arity()` |

---

*Last updated: January 3, 2026*
