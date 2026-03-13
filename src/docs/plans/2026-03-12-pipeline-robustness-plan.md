# Pipeline Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the Bayesian rule induction pipeline against stale caches, fingerprint collisions, translation errors, and missing provenance metadata.

**Architecture:** Five independent fixes applied to the shared pipeline. Each fix is testable in isolation. Fix 5 (provenance) creates a shared utility; the other four modify existing modules.

**Tech Stack:** Python 3.12, pytest, hashlib, json. No new dependencies.

## Execution Guidelines

- Present 2+ options for each major decision, wait for selection
- Explain code as you write it
- Test each step before proceeding
- Use `print(..., flush=True)` for all logging
- Follow existing import patterns in `gallery_analysis/` (sys.path.insert, try/except)
- JSON output with `indent=2`

---

### Task 1: Provenance Utility Module

**Files:**
- Create: `gallery_analysis/provenance.py`
- Test: `tests/test_provenance.py`

**Step 1: Write the failing test**

```python
"""Tests for provenance tracking utility."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gallery_analysis.provenance import compute_provenance, compute_probe_hash


class TestComputeProbeHash:
    def test_deterministic(self):
        """Same probe set produces same hash."""
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        h1 = compute_probe_hash(probes)
        h2 = compute_probe_hash(probes)
        assert h1 == h2

    def test_different_seeds_different_hash(self):
        """Different probe seeds produce different hashes."""
        from gallery_analysis.exemplars import generate_probe_set
        probes_a = generate_probe_set(10, seed=42)
        probes_b = generate_probe_set(10, seed=99)
        assert compute_probe_hash(probes_a) != compute_probe_hash(probes_b)

    def test_returns_hex_string(self):
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        h = compute_probe_hash(probes)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex


class TestComputeProvenance:
    def test_returns_dict_with_required_keys(self):
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        prov = compute_provenance(
            probe_seed=42,
            n_probes=10,
            probes=probes,
            n_equiv_classes=100,
        )
        assert "probe_seed" in prov
        assert "n_probes" in prov
        assert "probe_hash" in prov
        assert "n_equiv_classes" in prov
        assert "timestamp" in prov

    def test_optional_file_hashes(self):
        """When file paths provided, their hashes are included."""
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        # Use this test file as a stand-in for any real file
        test_file = Path(__file__)
        prov = compute_provenance(
            probe_seed=42, n_probes=10, probes=probes,
            inject_path=str(test_file),
        )
        assert "inject_hash" in prov
        assert prov["inject_hash"] is not None

    def test_missing_file_gives_none(self):
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        prov = compute_provenance(
            probe_seed=42, n_probes=10, probes=probes,
            inject_path="/nonexistent/file.json",
        )
        assert prov["inject_hash"] is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provenance.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'gallery_analysis.provenance'"

**Step 3: Write minimal implementation**

```python
"""Provenance tracking for pipeline output files.

Computes deterministic hashes of probe sets, input files, and grammar
configurations so that output JSONs carry enough metadata to detect
whether two runs used the same inputs.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand


def compute_probe_hash(probes: List[Hand]) -> str:
    """SHA256 hash of a serialized probe set."""
    serialized = json.dumps([
        [(c.suit.value, c.rank.value) for c in hand]
        for hand in probes
    ], sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _file_hash(path: str) -> Optional[str]:
    """SHA256 hash of file contents, or None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_provenance(
    probe_seed: int,
    n_probes: int,
    probes: List[Hand],
    inject_path: Optional[str] = None,
    exemplar_path: Optional[str] = None,
    grammar_hash: Optional[str] = None,
    n_equiv_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a provenance dict for embedding in output JSON.

    All hash fields are computed from the actual objects/files used,
    not from user-provided strings. This ensures two result files
    can be compared to check whether they used the same inputs.

    Args:
        probe_seed: Random seed used for probe generation.
        n_probes: Number of probe hands.
        probes: The actual probe hands (hashed for verification).
        inject_path: Path to injection JSON file (hashed if exists).
        exemplar_path: Path to frozen-exemplars.json (hashed if exists).
        grammar_hash: Pre-computed hash of grammar productions.
        n_equiv_classes: Number of equivalence classes in the pool.

    Returns:
        Dict with provenance metadata.
    """
    return {
        "probe_seed": probe_seed,
        "n_probes": n_probes,
        "probe_hash": compute_probe_hash(probes),
        "inject_path": inject_path,
        "inject_hash": _file_hash(inject_path) if inject_path else None,
        "exemplar_path": exemplar_path,
        "exemplar_hash": _file_hash(exemplar_path) if exemplar_path else None,
        "grammar_hash": grammar_hash,
        "n_equiv_classes": n_equiv_classes,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provenance.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add gallery_analysis/provenance.py tests/test_provenance.py
git commit -m "feat: add provenance tracking utility for pipeline output files"
```

---

### Task 2: Extension Cache Safety

**Files:**
- Modify: `gallery_analysis/analyze.py:199-268` (estimate_extensions function)
- Test: `tests/test_cache_safety.py`

**Step 1: Write the failing test**

```python
"""Tests for extension cache safety (probe hash validation)."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gallery_analysis.provenance import compute_probe_hash
from gallery_analysis.exemplars import generate_probe_set


class TestCacheSafety:
    def test_cache_with_valid_meta_is_used(self, tmp_path):
        """Cache with matching probe hash should be loaded normally."""
        from gallery_analysis.analyze import estimate_extensions

        probes = generate_probe_set(10, seed=42)
        probe_hash = compute_probe_hash(probes)

        # Create a fake equivalence class
        equiv = [{
            "fingerprint": "fake_fp_001",
            "predicate": lambda h: True,
        }]

        # Write cache with valid _meta
        cache_file = tmp_path / "cache.json"
        cache_data = {
            "_meta": {
                "probe_seed": 42,
                "n_probes": 10,
                "probe_hash": probe_hash,
            },
            "fake_fp_001": [1000, 0.05],
        }
        cache_file.write_text(json.dumps(cache_data))

        # Should use cached value
        extensions = estimate_extensions(
            equiv, verbose=0, cache_path=str(cache_file),
            _probe_hash=probe_hash,
        )
        assert extensions[0] == (1000, 0.05)

    def test_cache_with_wrong_meta_is_discarded(self, tmp_path):
        """Cache with wrong probe hash should be discarded entirely."""
        from gallery_analysis.analyze import estimate_extensions

        probes = generate_probe_set(10, seed=42)
        probe_hash = compute_probe_hash(probes)

        equiv = [{
            "fingerprint": "fake_fp_001",
            "predicate": lambda h: len(h) > 0,
        }]

        # Write cache with WRONG probe hash
        cache_file = tmp_path / "cache.json"
        cache_data = {
            "_meta": {
                "probe_seed": 99,
                "n_probes": 10,
                "probe_hash": "wrong_hash_value",
            },
            "fake_fp_001": [9999, 0.99],
        }
        cache_file.write_text(json.dumps(cache_data))

        # Should discard cache and recompute
        extensions = estimate_extensions(
            equiv, verbose=0, cache_path=str(cache_file),
            _probe_hash=probe_hash,
        )
        # Recomputed value should NOT be (9999, 0.99)
        assert extensions[0] != (9999, 0.99)

    def test_saved_cache_includes_meta(self, tmp_path):
        """After saving, cache file should include _meta block."""
        from gallery_analysis.analyze import estimate_extensions

        probes = generate_probe_set(10, seed=42)
        probe_hash = compute_probe_hash(probes)

        equiv = [{
            "fingerprint": "fp_test",
            "predicate": lambda h: True,
        }]

        cache_file = tmp_path / "cache.json"
        estimate_extensions(
            equiv, verbose=0, cache_path=str(cache_file),
            _probe_hash=probe_hash,
        )

        saved = json.loads(cache_file.read_text())
        assert "_meta" in saved
        assert saved["_meta"]["probe_hash"] == probe_hash
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_safety.py -v`
Expected: FAIL — `estimate_extensions` doesn't accept `_probe_hash` parameter yet

**Step 3: Modify `estimate_extensions()` in `analyze.py`**

Add `_probe_hash: str = None` parameter. On load: check `_meta.probe_hash` against `_probe_hash`. On save: write `_meta` block.

Key changes to `analyze.py:estimate_extensions()`:

1. Add parameter: `_probe_hash: str = None`
2. After loading cache JSON, check for `_meta` key:
   - If present and `_meta["probe_hash"] != _probe_hash`: print warning, discard cache (set `cache = {}`)
   - If absent: print one-time warning about old-format cache
3. Before saving, inject `_meta` block into the cache dict
4. In `run_analysis()` and callers: compute probe hash and pass it through

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_safety.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add gallery_analysis/analyze.py tests/test_cache_safety.py
git commit -m "feat: add probe hash validation to extension cache"
```

---

### Task 3: Approximate True Rules Flag

**Files:**
- Modify: `gallery_analysis/analyze.py:281-457` (score_rule function)
- Modify: `gallery_analysis/analyze.py:896-914` (save_results section)
- Test: `tests/test_approximate_rules.py`

**Step 1: Write the failing test**

```python
"""Tests for approximate true rule flagging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestApproximateRuleFlag:
    def test_score_rule_includes_approximate_flag(self):
        """score_rule should return true_rule_approximate when source is approximate."""
        from gallery_analysis.analyze import score_rule
        from rules.cards import Card, Suit, Rank

        # Minimal equiv class marked as approximate true rule
        hand = [Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                                                 Rank.FIVE, Rank.SIX, Rank.SEVEN]]
        equiv = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_approx",
            "predicate": lambda h: True,
            "source": "true_rule_approximate",
        }]
        extensions = [(10_000_000, 0.49)]

        result = score_rule(
            "test_rule", [hand], equiv, extensions,
            true_rule_fingerprint="fp_approx",
        )

        assert result["true_rule_approximate"] is True

    def test_exact_true_rule_not_flagged(self):
        """Exact true rules should have true_rule_approximate=False."""
        from gallery_analysis.analyze import score_rule
        from rules.cards import Card, Suit, Rank

        hand = [Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                                                 Rank.FIVE, Rank.SIX, Rank.SEVEN]]
        equiv = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_exact",
            "predicate": lambda h: True,
            "source": "merged",
        }]
        extensions = [(10_000_000, 0.49)]

        result = score_rule(
            "test_rule", [hand], equiv, extensions,
            true_rule_fingerprint="fp_exact",
        )

        assert result["true_rule_approximate"] is False

    def test_no_true_rule_gives_none(self):
        """When no true rule fingerprint provided, flag should be None."""
        from gallery_analysis.analyze import score_rule
        from rules.cards import Card, Suit, Rank

        hand = [Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                                                 Rank.FIVE, Rank.SIX, Rank.SEVEN]]
        equiv = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_x",
            "predicate": lambda h: True,
        }]
        extensions = [(10_000_000, 0.49)]

        result = score_rule("test_rule", [hand], equiv, extensions)

        assert result["true_rule_approximate"] is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_approximate_rules.py -v`
Expected: FAIL — `true_rule_approximate` key not in result dict

**Step 3: Modify `score_rule()` in `analyze.py`**

After line 404 (where `true_rule_hit_vector` is set after finding the true rule by fingerprint), look up the equivalence class's `source` field:

```python
# After the true-rule tracking loop (around line 404):
true_rule_approximate = None
if true_rule_fingerprint and true_rule_rank is not None:
    # Find the equivalence class for the true rule
    for cls in equivalence_classes:
        if cls["fingerprint"] == true_rule_fingerprint:
            true_rule_approximate = (cls.get("source") == "true_rule_approximate")
            break
elif true_rule_fingerprint:
    true_rule_approximate = None  # true rule not found in pool
```

Add `"true_rule_approximate": true_rule_approximate` to the return dict (after line 456).

Also add to the save section in `run_analysis()` (around line 914):

```python
"true_rule_approximate": rr.get("true_rule_approximate"),
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_approximate_rules.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add gallery_analysis/analyze.py tests/test_approximate_rules.py
git commit -m "feat: propagate approximate true rule flag to results"
```

---

### Task 4: Rare Fingerprint Refinement

**Files:**
- Modify: `gallery_analysis/hypothesis_table.py` (add `refine_rare_classes()`)
- Modify: `gallery_analysis/analyze.py:60-186` (build_hypothesis_pool, add refinement step)
- Test: `tests/test_fingerprint_refinement.py`

**Step 1: Write the failing test**

```python
"""Tests for rare fingerprint refinement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rules.cards import Card, Suit, Rank
from gallery_analysis.exemplars import generate_probe_set


def _make_hand(*specs):
    return [Card(s, r) for s, r in specs]

# A specific rare hand that won't appear in random probes
RARE_HAND = _make_hand(
    (Suit.SPADES, Rank.ACE), (Suit.SPADES, Rank.KING),
    (Suit.SPADES, Rank.QUEEN), (Suit.SPADES, Rank.JACK),
    (Suit.SPADES, Rank.TEN), (Suit.SPADES, Rank.NINE),
)


class TestRefineRareClasses:
    def test_splits_colliding_rare_predicates(self):
        """Two rare predicates that collide on 10 probes should be split
        when refinement probes distinguish them."""
        from gallery_analysis.hypothesis_table import (
            compute_fingerprint, refine_rare_classes,
        )

        # Create two predicates that are both False on all normal probes
        # but differ on specific hands
        pred_a = lambda h: (h[0].suit == Suit.SPADES and h[0].rank == Rank.ACE
                            and h[1].suit == Suit.SPADES and h[1].rank == Rank.KING)
        pred_b = lambda h: (h[0].suit == Suit.HEARTS and h[0].rank == Rank.ACE
                            and h[1].suit == Suit.HEARTS and h[1].rank == Rank.KING)

        # Use very few probes so they collide
        small_probes = generate_probe_set(10, seed=42)
        fp_a = compute_fingerprint(pred_a, small_probes)
        fp_b = compute_fingerprint(pred_b, small_probes)

        # They should collide on 10 probes (both all-False)
        assert fp_a == fp_b, "Test setup: predicates should collide on 10 probes"

        # Create a fake equivalence class with both predicates merged
        equiv_classes = [{
            "canonical_program": "pred_a",
            "canonical_prior": -5.0,
            "summed_prior": -4.3,  # log(exp(-5) + exp(-5))
            "n_expressions": 2,
            "all_programs": ["pred_a", "pred_b"],
            "fingerprint": fp_a,
            "predicate": pred_a,  # canonical
            "_all_predicates": [pred_a, pred_b],
            "_all_priors": [-5.0, -5.0],
        }]

        refined = refine_rare_classes(
            equiv_classes, small_probes,
            hit_threshold=5,
            n_refinement_probes=2000,
            refinement_seed=4242,
        )

        # After refinement, should have 2 classes (split)
        assert len(refined) == 2

    def test_non_rare_classes_unchanged(self):
        """Classes that hit many probes should not be touched."""
        from gallery_analysis.hypothesis_table import refine_rare_classes

        probes = generate_probe_set(100, seed=42)

        # A common predicate (always true)
        equiv_classes = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_common",
            "predicate": lambda h: True,
        }]

        refined = refine_rare_classes(
            equiv_classes, probes,
            hit_threshold=5,
            n_refinement_probes=2000,
            refinement_seed=4242,
        )

        # Should be unchanged
        assert len(refined) == 1
        assert refined[0]["fingerprint"] == "fp_common"

    def test_single_program_class_unchanged(self):
        """Rare classes with only 1 program can't be split."""
        from gallery_analysis.hypothesis_table import (
            compute_fingerprint, refine_rare_classes,
        )

        probes = generate_probe_set(100, seed=42)
        pred = lambda h: False  # rare: fires on nothing
        fp = compute_fingerprint(pred, probes)

        equiv_classes = [{
            "canonical_program": "(λ false)",
            "canonical_prior": -2.0,
            "summed_prior": -2.0,
            "n_expressions": 1,
            "all_programs": ["(λ false)"],
            "fingerprint": fp,
            "predicate": pred,
        }]

        refined = refine_rare_classes(
            equiv_classes, probes,
            hit_threshold=5,
            n_refinement_probes=2000,
            refinement_seed=4242,
        )

        assert len(refined) == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fingerprint_refinement.py -v`
Expected: FAIL — `refine_rare_classes` not found in hypothesis_table

**Step 3: Implement `refine_rare_classes()` in `hypothesis_table.py`**

Add at the end of `hypothesis_table.py`:

```python
def refine_rare_classes(
    equivalence_classes: List[Dict],
    main_probes: List[Hand],
    hit_threshold: int = 5,
    n_refinement_probes: int = 2000,
    refinement_seed: int = 4242,
) -> List[Dict]:
    """
    Second-pass refinement for rare equivalence classes.

    Classes whose canonical predicate fires on <= hit_threshold of the
    main probes are re-fingerprinted with additional refinement probes.
    If members of a class diverge on the refinement probes, the class
    is split into sub-classes.

    Classes with only 1 program (n_expressions == 1) or that hit
    > hit_threshold probes are returned unchanged.

    Args:
        equivalence_classes: Equivalence classes from main fingerprinting.
        main_probes: The probe hands used for main fingerprinting.
        hit_threshold: Max hits on main probes to qualify as "rare".
        n_refinement_probes: Number of additional probes for refinement.
        refinement_seed: Seed for generating refinement probes.

    Returns:
        New list of equivalence classes with rare collisions resolved.
    """
    from gallery_analysis.exemplars import generate_probe_set
    import math

    refinement_probes = generate_probe_set(n_refinement_probes, seed=refinement_seed)

    result = []
    n_split = 0

    for cls in equivalence_classes:
        # Check if this class is rare
        pred = cls["predicate"]
        hits = sum(1 for p in main_probes if _safe_eval(pred, p))

        if hits > hit_threshold or cls["n_expressions"] <= 1:
            # Not rare or can't split — keep as-is
            result.append(cls)
            continue

        # Check if class has stored per-program predicates for splitting
        all_predicates = cls.get("_all_predicates")
        all_priors = cls.get("_all_priors")
        if not all_predicates or len(all_predicates) <= 1:
            # No per-program predicates stored — can't split
            result.append(cls)
            continue

        # Re-fingerprint each program on refinement probes
        sub_groups: Dict[str, List[int]] = {}
        for prog_idx, prog_pred in enumerate(all_predicates):
            refined_fp = compute_fingerprint(prog_pred, refinement_probes)
            sub_groups.setdefault(refined_fp, []).append(prog_idx)

        if len(sub_groups) <= 1:
            # All programs still agree — no split needed
            result.append(cls)
            continue

        # Split into sub-classes
        all_programs = cls["all_programs"]
        for refined_fp, indices in sub_groups.items():
            # Find the highest-prior program as canonical
            sub_priors = [all_priors[i] for i in indices]
            best_idx = indices[sub_priors.index(max(sub_priors))]

            summed = math.log(sum(math.exp(all_priors[i]) for i in indices))

            # Combine main + refinement fingerprint for uniqueness
            combined_fp = compute_fingerprint(
                all_predicates[best_idx],
                main_probes + refinement_probes,
            )

            sub_class = {
                "canonical_program": all_programs[best_idx],
                "canonical_prior": all_priors[best_idx],
                "summed_prior": summed,
                "n_expressions": len(indices),
                "all_programs": [all_programs[i] for i in indices],
                "fingerprint": combined_fp,
                "predicate": all_predicates[best_idx],
            }
            # Preserve any extra metadata
            for key in cls:
                if key not in sub_class and key not in ("_all_predicates", "_all_priors"):
                    sub_class[key] = cls[key]

            result.append(sub_class)

        n_split += 1

    if n_split > 0:
        print(f"  Fingerprint refinement: {n_split} rare classes split, "
              f"total classes: {len(result)} (was {len(equivalence_classes)})",
              flush=True)

    return result


def _safe_eval(predicate, hand) -> bool:
    """Evaluate predicate, returning False on any exception."""
    try:
        return bool(predicate(hand))
    except Exception:
        return False
```

**Important note for the implementer**: The refinement requires per-program predicates (`_all_predicates`) and priors (`_all_priors`) to be stored on each equivalence class during the main fingerprinting step in `build_hypothesis_pool()`. Currently only the canonical predicate is stored. You need to modify the fingerprinting loop in `analyze.py:143-169` to store all predicates:

In `build_hypothesis_pool()`, after line 156 (`canonical_str, canonical_pred, canonical_prior = group[0]`), the equivalence class dict should also include:

```python
"_all_predicates": [pred_fn for _, pred_fn, _ in group],
"_all_priors": [lp for _, _, lp in group],
```

Then call `refine_rare_classes()` after line 172 (after `equivalence_classes.sort()`):

```python
# Refinement pass for rare classes
equivalence_classes = refine_rare_classes(
    equivalence_classes, probes,
    hit_threshold=5,
    n_refinement_probes=2000,
    refinement_seed=4242,
)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fingerprint_refinement.py -v`
Expected: PASS (all 3 tests)

**Step 5: Run existing tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add gallery_analysis/hypothesis_table.py gallery_analysis/analyze.py tests/test_fingerprint_refinement.py
git commit -m "feat: add fingerprint refinement for rare equivalence classes"
```

---

### Task 5: Wire Provenance Into Pipeline Scripts

**Files:**
- Modify: `gallery_analysis/analyze.py:896-914` (save section in main)
- Modify: `gallery_analysis/run_diagnosticity.py` (save section)
- Modify: `gallery_analysis/depth_mass_analysis.py` (save section)

**Step 1: Modify `analyze.py` save section**

After line 898 (`save_results = {`), add provenance computation. Import `compute_provenance` at the top of the file. In `run_analysis()`, compute and return provenance. In the save section, include it.

In `run_analysis()`, before the return statement (~line 626), add:

```python
# Compute provenance
from gallery_analysis.provenance import compute_provenance
probes = generate_probe_set(n_probes=n_probes, seed=42)
provenance = compute_provenance(
    probe_seed=42,
    n_probes=n_probes,
    probes=probes,
    inject_path=inject_path,
    n_equiv_classes=len(equiv_classes),
)
```

Include in the return dict: `"provenance": provenance`.

In the save section (~line 899), add: `"provenance": results["provenance"]`.

**Step 2: Apply same pattern to `run_diagnosticity.py` and `depth_mass_analysis.py`**

Each script computes provenance after building the hypothesis pool and includes it in the output JSON.

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add gallery_analysis/analyze.py gallery_analysis/run_diagnosticity.py gallery_analysis/depth_mass_analysis.py
git commit -m "feat: embed provenance metadata in all pipeline output files"
```

---

### Task 6: LLM Translation Verification

**Files:**
- Modify: `gallery_analysis/translate_hypotheses.py`
- Test: `tests/test_translation_verification.py`

**Step 1: Write the failing test**

```python
"""Tests for LLM translation verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rules.cards import Card, Suit, Rank


class TestVerifyTranslation:
    def test_matching_translation_returns_no_disagreements(self):
        """Identical predicates should produce zero disagreements."""
        from gallery_analysis.translate_hypotheses import verify_translation

        # Both always return True
        python_fn = lambda hand: True
        dsl_fn = lambda hand: True

        disagreements = verify_translation(python_fn, dsl_fn, n_test=100, seed=99)
        assert len(disagreements) == 0

    def test_mismatched_translation_returns_disagreements(self):
        """Different predicates should return disagreeing hands."""
        from gallery_analysis.translate_hypotheses import verify_translation

        python_fn = lambda hand: True
        dsl_fn = lambda hand: False

        disagreements = verify_translation(python_fn, dsl_fn, n_test=100, seed=99)
        assert len(disagreements) == 100

    def test_partial_mismatch(self):
        """Predicates that differ on some hands should return those hands."""
        from gallery_analysis.translate_hypotheses import verify_translation

        python_fn = lambda hand: hand[0].suit == Suit.HEARTS
        dsl_fn = lambda hand: hand[0].suit == Suit.SPADES

        disagreements = verify_translation(python_fn, dsl_fn, n_test=500, seed=99)
        # Should have some disagreements but not all
        assert 0 < len(disagreements) < 500
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_translation_verification.py -v`
Expected: FAIL — `verify_translation` not found

**Step 3: Add `verify_translation()` to `translate_hypotheses.py`**

```python
def verify_translation(
    python_fn,
    dsl_fn,
    n_test: int = 1000,
    seed: int = 99,
) -> list:
    """
    Compare a Python lambda and DSL predicate on random hands.

    Evaluates both functions on n_test random 6-card hands and returns
    a list of hands where they disagree. Empty list = perfect match.

    Args:
        python_fn: The original Python lambda (callable: Hand -> bool).
        dsl_fn: The translated DSL predicate (callable: Hand -> bool).
        n_test: Number of random hands to test.
        seed: Random seed for reproducibility.

    Returns:
        List of (hand, python_result, dsl_result) tuples for disagreements.
    """
    import random
    from rules.cards import Card, Suit, Rank

    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    disagreements = []

    for _ in range(n_test):
        hand = rng.sample(deck, 6)
        try:
            py_result = bool(python_fn(hand))
        except Exception:
            py_result = None
        try:
            dsl_result = bool(dsl_fn(hand))
        except Exception:
            dsl_result = None

        if py_result != dsl_result:
            disagreements.append((hand, py_result, dsl_result))

    return disagreements
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_translation_verification.py -v`
Expected: PASS (all 3 tests)

**Step 5: Rekey translations from indices to stable keys**

This is a larger change to the `_translate_one()` function and the main `translate_all()` loop. Change `_translate_one(idx, ...)` to `_translate_one(rule_id, hypothesis_text, ...)` and rekey the internal dispatch from `if idx == 0:` to `if (rule_id, text) == ("adjacent_share_rank_or_suit", "At least one suit has 3 or more cards."):`.

**This is a mechanical but large refactor** — 95 if-clauses need their keys changed. The implementer should:

1. Extract all `(idx, rule_id, text)` triples from the current code
2. Replace each `if idx == N:` with `if (rule_id, text) == ("...", "..."):`
3. Run `verify_translation()` on each translated pair to confirm no regressions

**Step 6: Commit**

```bash
git add gallery_analysis/translate_hypotheses.py tests/test_translation_verification.py
git commit -m "feat: add translation verification and stable keying for LLM hypotheses"
```

---

### Task 7: Integration Test — Full Pipeline

**Files:**
- Test: `tests/test_pipeline_integration.py`

**Step 1: Write a smoke test that exercises the full pipeline with the new fixes**

```python
"""Integration test: full pipeline with all robustness fixes."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.mark.slow
class TestPipelineIntegration:
    def test_quick_run_produces_valid_output(self, tmp_path):
        """A --quick run should produce valid JSON with provenance."""
        from gallery_analysis.analyze import run_analysis

        results = run_analysis(
            max_depth=5,
            max_programs=50_000,
            max_cost=25.0,
            timeout=120.0,
            extension_samples=10_000,
            verbose=0,
        )

        # Should have provenance
        assert "provenance" in results
        assert "probe_hash" in results["provenance"]

        # Should have rule results
        assert len(results["rule_results"]) > 0

        # Approximate flag should be present
        for rule_id, rr in results["rule_results"].items():
            assert "true_rule_approximate" in rr
```

**Step 2: Run**

Run: `python -m pytest tests/test_pipeline_integration.py -v -m slow`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_pipeline_integration.py
git commit -m "test: add integration test for pipeline robustness fixes"
```
