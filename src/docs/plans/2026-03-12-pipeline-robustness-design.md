# Pipeline Robustness — Design Sheet

## Context

The Bayesian rule induction pipeline has grown to include enumeration, injection, extension caching, diagnosticity analysis, and visualization. As we begin experimenting with upstream changes (new exemplars, different priors, LLM-generated hypotheses), we need confidence that changes propagate correctly and stale data doesn't silently corrupt results.

This design addresses five robustness issues identified in a deep audit of the pipeline.

---

## Fix 1: Extension Cache Safety

### Problem
The extension cache (`extension_cache_depth6.json`) maps `fingerprint → (ext_size, base_rate)` but doesn't record which probe configuration produced those fingerprints. If the probe set changes (different seed, different n_probes), fingerprints shift and lookups silently return wrong extension sizes.

### Solution
Add a `_meta` header to the cache JSON:

```json
{
  "_meta": {
    "probe_seed": 42,
    "n_probes": 500,
    "probe_hash": "sha256 of serialized probe set",
    "created": "2026-03-12T...",
    "n_entries": 4135
  },
  "a3f8c2...": [1000, 0.00005],
  ...
}
```

**On load** (`estimate_extensions()`):
- If `_meta` present: compute current probe hash, compare. Mismatch → print warning, discard entire cache, recompute everything.
- If `_meta` absent (old-format cache): treat as valid, print one-time warning suggesting regeneration.

**On save**: always write `_meta`.

### Files to modify
- `analyze.py:estimate_extensions()` — load/save section (~20 lines)

### Probe hash computation
```python
import hashlib, json

def compute_probe_hash(probes):
    """Deterministic hash of a probe set for cache validation."""
    serialized = json.dumps([
        [(c.suit.value, c.rank.value) for c in hand]
        for hand in probes
    ], sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()
```

---

## Fix 2: Rare Predicate Fingerprint Refinement

### Problem
Predicates that fire on ≤2 of 500 random probes are vulnerable to fingerprint collisions. Two semantically different rare predicates could share a fingerprint and merge into one equivalence class, inflating its summed prior. Currently 41 of 3,674 classes (1.1%) hit ≤2 probes.

### Solution
Targeted second pass after main fingerprinting. Only re-fingerprints the ~41 low-hit classes with 2000 additional probes. Splits any classes where members diverge on the refined probe set.

**Algorithm:**

```
1. After main fingerprinting (500 probes), identify "rare classes":
   classes where the canonical predicate fires on ≤ threshold probes
   (threshold = 5, i.e., ≤1% hit rate)

2. Generate 2000 additional refinement probes (deterministic seed)

3. For each rare class with >1 program:
   Recompute fingerprint of each program on the 2000 refinement probes
   If fingerprints diverge: split into sub-classes

4. For rare classes with 1 program: no action needed (nothing to split)
```

**Cost**: ~41 × 2000 = 82K evaluations (negligible vs 1.8M for main pass).

**Extension cache impact**: None for the 3,633 non-rare classes. Rare classes that split get new fingerprints and need extension re-estimation (41 classes × 100K MC samples ≈ 1 min).

### Files to modify
- `analyze.py:build_hypothesis_pool()` — add refinement pass after fingerprinting step
- `hypothesis_table.py` — add `refine_rare_classes()` function

### Refinement probe configuration
- Seed: 4242 (different from main seed 42 to ensure independence)
- n_probes: 2000
- Hit threshold: 5 (classes where canonical predicate fires on ≤5 of 500 main probes)

---

## Fix 3: LLM Translation Verification

### Problem
95 LLM hypotheses were manually translated from Python lambdas to DSL strings, indexed by hard-coded position (0–116). No automated check that DSL matches Python lambda. Index drift if raw file is regenerated.

### Solution: Two parts

#### Part A: Automated equivalence checking
Add a verification step to `translate_hypotheses.py` that runs at translation time:

```python
def verify_translation(python_lambda_str, dsl_predicate, n_test=1000, seed=99):
    """
    Evaluate both the Python lambda and DSL predicate on random hands.
    Return list of disagreeing hands (empty = perfect match).
    """
    rng = random.Random(seed)
    deck = [Card(s, r) for s in Suit for r in Rank]
    disagreements = []

    python_fn = eval(python_lambda_str)  # from raw hypothesis

    for _ in range(n_test):
        hand = rng.sample(deck, 6)
        try:
            py_result = bool(python_fn(hand))
        except Exception:
            py_result = None
        try:
            dsl_result = bool(dsl_predicate(hand))
        except Exception:
            dsl_result = None

        if py_result != dsl_result:
            disagreements.append(hand)

    return disagreements
```

Run after translation, before writing `injected_hypotheses.json`. Any hypothesis with >0 disagreements gets flagged in the output with `"translation_verified": false` and the disagreement count.

#### Part B: Stable keys instead of indices
Change translation dict from integer indices to `(rule_id, hypothesis_text)` tuples:

```python
# BEFORE (fragile):
TRANSLATIONS = {
    0: "(λ ge (max_suit_count $0) 3)",
    1: "(λ all (λ is_even (rank_value $0)) $0)",
    ...
}

# AFTER (stable):
TRANSLATIONS = {
    ("adjacent_share_rank_or_suit", "At least one suit has 3 or more cards."):
        "(λ ge (max_suit_count $0) 3)",
    ("all_even", "All card values are even numbers."):
        "(λ all (λ is_even (rank_value $0)) $0)",
    ...
}
```

This way reordering or regenerating `llm_hypotheses_raw.json` doesn't break translations.

### Files to modify
- `translate_hypotheses.py` — rekey translations, add verification loop
- `injected_hypotheses.json` — regenerated with `translation_verified` field

---

## Fix 4: Approximate True Rules Documentation

### Problem
`suit_brackets_no_cross` and `suit_brackets_nested` have DSL translations that are supersets of the true rule (DSL uses independent counters; true rule requires stack-based bracket matching). Their posterior rank is artificially depressed. Downstream analysis doesn't surface this caveat.

### Solution: Two parts

#### Part A: Propagate flag to results
In `analyze.py:score_rule()`, when the true rule's equivalence class is found, check the `source` field. If `"true_rule_approximate"`, include `"true_rule_approximate": true` in the result dict.

```python
# In score_rule(), after finding true rule:
if true_rule_fingerprint:
    for i, (sh, prob) in enumerate(normalized):
        if sh.fingerprint == true_rule_fingerprint:
            ...
            # Check source
            cls = equiv_classes[cls_idx_for_true]
            is_approx = cls.get("source") == "true_rule_approximate"
            break

return {
    ...
    "true_rule_approximate": is_approx if true_rule_fingerprint else None,
}
```

#### Part B: Top-level documentation in results JSON
Add an `"approximate_rules"` section:

```json
{
  "approximate_rules": [
    {
      "rule_id": "suit_brackets_no_cross",
      "reason": "DSL uses independent suit counters; true rule requires stack-based matching. DSL accepts ~X% more hands.",
      "impact": "Extension size inflated → likelihood lower → posterior rank depressed"
    },
    {
      "rule_id": "suit_brackets_nested",
      "reason": "Same limitation as suit_brackets_no_cross.",
      "impact": "Same"
    }
  ]
}
```

### Files to modify
- `analyze.py:score_rule()` — check source field, add `true_rule_approximate` to result
- `analyze.py:run_analysis()` — add `approximate_rules` to output dict
- Visualization can then show these with an asterisk (future, not in this fix)

---

## Fix 5: Provenance Tracking in Output Files

### Problem
Result JSONs don't record which probe configuration, injection file, exemplar set, or grammar version was used. Can't tell whether two result files are comparable.

### Solution
Add a `"provenance"` block to every output JSON. Computed at run time from actual objects/files used.

```json
{
  "provenance": {
    "probe_seed": 42,
    "n_probes": 500,
    "probe_hash": "sha256...",
    "extension_cache_path": "gallery_analysis/results/extension_cache_depth6.json",
    "inject_path": "gallery_analysis/data/injected_hypotheses.json",
    "inject_hash": "sha256 of injection file contents",
    "exemplar_path": "card-games/rule-gallery/frozen-exemplars.json",
    "exemplar_hash": "sha256 of exemplar file contents",
    "grammar_hash": "sha256 of serialized grammar productions + weights",
    "n_equiv_classes": 3741,
    "timestamp": "2026-03-12T14:30:45Z"
  }
}
```

### Shared utility

Create `gallery_analysis/provenance.py`:

```python
def compute_provenance(
    probe_seed, n_probes, probes,
    inject_path=None, exemplar_path=None,
    grammar=None, n_equiv_classes=None,
):
    """Build a provenance dict for embedding in output JSON."""
    ...
```

This function is called by each script (`analyze.py`, `run_diagnosticity.py`, `depth_mass_analysis.py`) and its output is included in the saved JSON.

### Files to modify
- NEW: `gallery_analysis/provenance.py` — shared hash computation
- `analyze.py:run_analysis()` — compute and include provenance
- `run_diagnosticity.py:main()` — same
- `depth_mass_analysis.py:main()` — same

---

## Implementation Order

Fixes are independent — no ordering constraints. But logically:

1. **Fix 5 (provenance)** first — it's the simplest and immediately useful for tracking which runs are comparable
2. **Fix 1 (cache safety)** — prevents the most dangerous silent failure
3. **Fix 4 (approximate rules)** — small, self-contained
4. **Fix 2 (rare fingerprint refinement)** — medium complexity, touches the core pipeline
5. **Fix 3 (LLM verification)** — most involved, touches the translation pipeline

## Testing Strategy

- **Fix 1**: Unit test: write cache with wrong probe hash, verify it gets discarded on reload
- **Fix 2**: Unit test: create two synthetic predicates that collide on 500 probes but diverge on 2000, verify they get split
- **Fix 3**: Integration test: run verification on all 95 translations, assert 0 disagreements for the ones we believe are correct
- **Fix 4**: Unit test: verify `true_rule_approximate` field appears in results for the two known rules
- **Fix 5**: Unit test: verify provenance block is present in output JSON with all expected fields
