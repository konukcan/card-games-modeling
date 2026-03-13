"""
Diagnostic: check how many of 60 true gallery rules have a semantically
equivalent program in the enumerated hypothesis space.

For each gallery rule, we:
1. Enumerate programs up to the specified depth
2. Filter trivial programs
3. For each surviving program, check if it agrees with the true rule
   on all 360 exemplar hands + 500 random probe hands

This gives us a baseline of "how many true rules are findable" at each depth.

Usage:
    cd src
    python gallery_analysis/check_true_rule_coverage.py --depth 7 --max-programs 2000000
"""
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank
from gallery_analysis.enumerator import enumerate_hypotheses_with_stats
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.hypothesis_table import filter_trivial
from gallery_analysis.gallery_rules import GALLERY_RULES


def check_coverage(max_depth: int = 7, max_programs: int = 2_000_000):
    """Check which true rules are found in the enumerated hypothesis space."""

    # Load exemplars and build probe set
    exemplars = load_exemplars()
    all_exemplar_hands = []
    for rule_id, exs in exemplars.items():
        all_exemplar_hands.extend(exs["hands_primary"])

    probe_hands = generate_probe_set(500)
    test_hands = all_exemplar_hands + probe_hands  # ~860 hands total

    # Step 1: Enumerate
    print(f"Step 1: Enumerating programs (depth={max_depth}, max={max_programs:,})...")
    t0 = time.time()
    programs, enum_stats = enumerate_hypotheses_with_stats(
        max_depth=max_depth, max_programs=max_programs
    )
    t_enum = time.time() - t0
    print(f"  Accepted: {len(programs):,} in {t_enum:.1f}s")

    # Step 2: Trivial filter
    print("Step 2: Trivial filter...")
    t0 = time.time()
    survivors, trivial_stats = filter_trivial(programs, all_exemplar_hands)
    t_filt = time.time() - t0
    print(f"  Survivors: {len(survivors):,} / {len(programs):,} in {t_filt:.1f}s")

    # Step 3: Pre-compute true rule signatures on test hands
    print("Step 3: Computing true rule signatures...")
    true_sigs = {}
    for rule_id, rule_info in GALLERY_RULES.items():
        pred = rule_info["predicate"]
        sig = tuple(pred(h) for h in test_hands)
        true_sigs[rule_id] = sig

    # Step 4: Check each surviving program against all 60 true rules
    print(f"Step 4: Checking {len(survivors):,} programs against 60 true rules...")
    t0 = time.time()

    found = {}  # rule_id -> (prog_str, log_prior) of first matching program
    found_count = defaultdict(int)  # rule_id -> count of matching programs

    for i, (prog_str, pred_fn, log_prior) in enumerate(survivors):
        if i > 0 and i % 50000 == 0:
            elapsed = time.time() - t0
            n_found = len(found)
            print(f"  {i:,}/{len(survivors):,} checked, {n_found}/60 found so far ({elapsed:.0f}s)")

        # Compute this program's signature
        try:
            prog_sig = tuple(pred_fn(h) for h in test_hands)
        except Exception:
            continue

        # Check against each unfound true rule (and count matches for found ones)
        for rule_id, true_sig in true_sigs.items():
            if prog_sig == true_sig:
                if rule_id not in found:
                    found[rule_id] = (prog_str, log_prior)
                found_count[rule_id] += 1

    t_check = time.time() - t0

    # Report
    print(f"\n{'='*80}")
    print(f"TRUE RULE COVERAGE AT DEPTH {max_depth} ({max_programs:,} program budget)")
    print(f"{'='*80}")
    print(f"\nFound: {len(found)}/60 true rules in hypothesis space")
    print(f"Check time: {t_check:.1f}s\n")

    # Group results
    groups = {1: [], 2: [], 3: []}
    for rule_id, rule_info in GALLERY_RULES.items():
        grp = rule_info["group"]
        if rule_id in found:
            prog_str, log_prior = found[rule_id]
            n_matches = found_count[rule_id]
            groups[grp].append((rule_id, True, prog_str, log_prior, n_matches))
        else:
            groups[grp].append((rule_id, False, None, None, 0))

    for grp in [1, 2, 3]:
        label = {1: "Easy", 2: "Medium", 3: "Hard"}[grp]
        rules = groups[grp]
        n_found_grp = sum(1 for r in rules if r[1])
        print(f"\nGroup {grp} ({label}): {n_found_grp}/{len(rules)} found")
        print("-" * 70)
        for rule_id, is_found, prog_str, log_prior, n_matches in sorted(rules, key=lambda x: (not x[1], x[0])):
            if is_found:
                print(f"  FOUND  {rule_id:<35} logP={log_prior:7.2f}  n={n_matches:4d}  {prog_str[:60]}")
            else:
                print(f"  MISS   {rule_id}")

    # Summary of missing rules
    missing = [r for r in GALLERY_RULES if r not in found]
    if missing:
        print(f"\n{'='*80}")
        print(f"MISSING RULES ({len(missing)}):")
        for rule_id in sorted(missing):
            grp = GALLERY_RULES[rule_id]["group"]
            desc = GALLERY_RULES[rule_id].get("description", "")
            print(f"  [{grp}] {rule_id}: {desc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-programs", type=int, default=2_000_000)
    args = parser.parse_args()
    check_coverage(max_depth=args.depth, max_programs=args.max_programs)
