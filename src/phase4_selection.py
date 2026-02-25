#!/usr/bin/env python3
"""
Phase 4: Selection Optimization

Selects optimal rule subsets for two pilot designs:
  - WIDE pilot: 32 rules (broad coverage of DL × base-rate space)
  - DEEP pilot: 8 rules (intensive study with repeated pairs)

The selection draws from three pools:
  1. Existing catalogue rules (55, already have JS implementations)
  2. Novel template-derived rules (94, interpretable, need JS)
  3. Grammar-enumerated rules (604, need curation for interpretability)

Selection criteria (from the desiderata):
  A1 (HARD): base rate ≥ 1%
  A2 (SOFT): base rate ≤ 60%
  D1: DL diversity — spread across complexity levels
  D2: base rate sweet spot [5%, 50%], ideal [8%, 35%]
  D3: family diversity — represent many structural templates
  D4: pairing feasibility — pairs need non-empty exclusive regions
  D5: confusability spread — some easy, some hard pairs
  D6: interpretability — rules must be explainable to participants

Approach:
  Step 1: Curate the candidate pool (filter, annotate interpretability)
  Step 2: Compute pairwise overlap/confusability via Monte Carlo
  Step 3: Score and rank individual rules
  Step 4: Greedy selection with diversity constraints
  Step 5: Validate selected subsets

Usage:
    python3 phase4_selection.py [--samples 50000] [--deck-size 52]
"""

import sys
import csv
import math
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional, Callable
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))

from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES,
    card_color, sample_hand
)
from rules.catalogue import create_all_rules, Rule


# ============================================================================
# Step 1: Build the curated candidate pool
# ============================================================================

def load_catalogue_rules():
    """Load catalogue rules with Phase 1 calibration data."""
    rules = create_all_rules()
    # Load calibration data for base rates
    cal_path = Path(__file__).parent / "calibration_phase1_deck52_full.csv"
    cal_data = {}
    if cal_path.exists():
        with open(cal_path) as f:
            for row in csv.DictReader(f):
                cal_data[row["token"]] = {
                    "base_rate": float(row["base_rate"]),
                    "base_rate_pct": float(row["base_rate_pct"]),
                    "base_rate_se": float(row["base_rate_se"]),
                    "tree_size": int(row["tree_size"]),
                    "tree_depth": int(row["tree_depth"]),
                    "n_primitives": int(row["n_primitives"]),
                    "level": int(row["level"]),
                    "family": row["family"],
                }

    candidates = []
    for r in rules:
        cal = cal_data.get(r.token, {})
        candidates.append({
            "id": r.id,
            "token": r.token,
            "name": r.name,
            "family": cal.get("family", r.family),
            "base_rate": cal.get("base_rate", None),
            "base_rate_pct": cal.get("base_rate_pct", None),
            "tree_size": cal.get("tree_size", None),
            "level": cal.get("level", r.level),
            "source": "catalogue",
            "predicate": r.predicate,
            "description": r.description,
            "interpretable": True,  # All catalogue rules have descriptions
        })
    return candidates


def load_template_candidates():
    """Load novel template-derived rules with their predicates."""
    from template_generation import (
        gen_ends_rules, gen_palindrome_rules, gen_halves_copy_rules,
        gen_halves_property_rules, gen_adjacent_rules, gen_count_rules,
        gen_shift_rules, gen_global_rules
    )

    # Re-generate to get predicates
    all_template_rules = {}
    for gen_fn in [gen_ends_rules, gen_palindrome_rules, gen_halves_copy_rules,
                   gen_halves_property_rules, gen_adjacent_rules, gen_count_rules,
                   gen_shift_rules, gen_global_rules]:
        for rule in gen_fn():
            all_template_rules[rule["id"]] = rule

    # Load the CSV to get only rules that passed base rate filter
    csv_path = Path(__file__).parent / "template_candidates.csv"
    candidates = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rule_id = row["id"]
            if rule_id not in all_template_rules:
                continue
            template_rule = all_template_rules[rule_id]

            candidates.append({
                "id": rule_id,
                "token": f"t_{rule_id[:20]}",  # synthetic token
                "name": row["description"],
                "family": row["family"],
                "base_rate": float(row["base_rate"]),
                "base_rate_pct": float(row["base_rate_pct"]),
                "tree_size": None,  # Template rules don't have tree metrics
                "level": None,
                "source": "template",
                "predicate": template_rule["predicate"],
                "description": row["description"],
                "interpretable": True,  # Template rules are inherently interpretable
            })
    return candidates


def load_llm_candidates():
    """Load LLM-brainstormed rules with their predicates."""
    from llm_rule_candidates import define_llm_rules

    # Get predicates from the define_llm_rules() function
    llm_rules = define_llm_rules()

    # Load the CSV to get calibrated base rates
    csv_path = Path(__file__).parent / "llm_candidates.csv"
    csv_data = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                csv_data[row["id"]] = row

    candidates = []
    for rule in llm_rules:
        rule_id = rule["id"]
        csv_row = csv_data.get(rule_id)
        if csv_row is None:
            continue  # Skip rules not in the CSV (filtered out)

        candidates.append({
            "id": rule_id,
            "token": f"l_{rule_id[:20]}",  # synthetic token
            "name": rule["description"],
            "family": rule["family"],
            "base_rate": float(csv_row["base_rate"]),
            "base_rate_pct": float(csv_row["base_rate_pct"]),
            "tree_size": None,
            "level": None,
            "source": "llm",
            "predicate": rule["predicate"],
            "description": rule["description"],
            "interpretable": True,  # LLM rules are human-designed and interpretable
        })
    return candidates


def make_deck(deck_size=52):
    all_ranks = list(Rank)
    if deck_size == 52:
        ranks = all_ranks
    elif deck_size == 32:
        ranks = [r for r in all_ranks if RANK_VALUES[r] >= 7]
    else:
        raise ValueError(f"Unsupported deck size: {deck_size}")
    return [Card(suit, rank) for suit in Suit for rank in ranks]


# ============================================================================
# Step 2: Compute pairwise overlap metrics via Monte Carlo
# ============================================================================

def compute_pairwise_overlaps(rules, deck, hand_size, n_samples, seed=42):
    """
    For each pair of rules, estimate the four region probabilities:
      overlap:  P(A ∧ B)
      a_only:   P(A ∧ ¬B)
      b_only:   P(¬A ∧ B)
      neither:  P(¬A ∧ ¬B)

    This is the key input for confusability scoring (BC, E[samples]).

    We evaluate ALL rules on the SAME sample of hands to ensure consistent
    estimates and enable efficient pairwise computation.
    """
    random.seed(seed)
    n_rules = len(rules)

    print(f"  Evaluating {n_rules} rules on {n_samples:,} hands...")

    # Pre-sample all hands
    hands = [tuple(random.sample(deck, hand_size)) for _ in range(n_samples)]

    # Evaluate each rule on each hand → bitmap
    # rule_results[i] is a list of True/False for rule i across all hands
    rule_results = []
    for i, rule in enumerate(rules):
        pred = rule["predicate"]
        results = []
        for hand in hands:
            try:
                r = pred(hand)
                results.append(bool(r) if r is not None else False)
            except Exception:
                results.append(False)
        rule_results.append(results)

        if (i + 1) % 50 == 0:
            print(f"    Evaluated {i+1}/{n_rules} rules...")

    print(f"  Computing pairwise overlaps for {n_rules * (n_rules - 1) // 2} pairs...")

    # Compute pairwise overlaps
    pair_metrics = {}
    for i in range(n_rules):
        for j in range(i + 1, n_rules):
            overlap = 0
            a_only = 0
            b_only = 0
            neither = 0
            for k in range(n_samples):
                a = rule_results[i][k]
                b = rule_results[j][k]
                if a and b:
                    overlap += 1
                elif a and not b:
                    a_only += 1
                elif not a and b:
                    b_only += 1
                else:
                    neither += 1

            pair_metrics[(i, j)] = {
                "overlap": overlap / n_samples,
                "a_only": a_only / n_samples,
                "b_only": b_only / n_samples,
                "neither": neither / n_samples,
            }

    return pair_metrics


# ============================================================================
# Step 3: Confusability metrics (reusing pair_selection.py framework)
# ============================================================================

def bhattacharyya_coefficient(overlap, a_only, b_only):
    """BC = overlap / sqrt(P(A) * P(B))"""
    p_a = a_only + overlap
    p_b = b_only + overlap
    if p_a <= 0 or p_b <= 0:
        return 0.0
    return overlap / math.sqrt(p_a * p_b)


def expected_samples_max(overlap, a_only, b_only):
    """max(E[A→B], E[B→A])"""
    if a_only > 1e-6:
        e_ab = (overlap + a_only) / a_only
    else:
        e_ab = float('inf')
    if b_only > 1e-6:
        e_ba = (overlap + b_only) / b_only
    else:
        e_ba = float('inf')
    return max(e_ab, e_ba)


def pair_feasible(a_only, b_only, threshold=0.005):
    """A pair is feasible if both exclusive regions are large enough."""
    return a_only >= threshold and b_only >= threshold


# ============================================================================
# Step 4: Rule scoring and selection
# ============================================================================

def score_rule(rule):
    """
    Score an individual rule for inclusion in the candidate pool.

    Higher score = more desirable. Combines:
    - Base rate quality: bell-shaped preference for 8-35% range
    - Interpretability: bonus for catalogue/template rules
    - Family novelty: bonus for underrepresented families
    """
    rate = rule["base_rate"]
    if rate is None:
        return 0

    # Base rate preference: bell curve centered at ~15%
    # Peaks at 15%, falls off outside [5%, 50%]
    if rate < 0.01:
        rate_score = 0  # A1 violation
    elif rate < 0.05:
        rate_score = 30 * (rate - 0.01) / 0.04  # Ramp up
    elif rate <= 0.35:
        rate_score = 30  # Sweet spot — full score
    elif rate <= 0.50:
        rate_score = 30 * (0.50 - rate) / 0.15  # Ramp down
    else:
        rate_score = 5  # A2 soft violation

    # Source bonus
    if rule["source"] == "catalogue":
        source_score = 20  # Already has JS, tested in experiments
    elif rule["source"] in ("template", "llm"):
        source_score = 15  # Interpretable, needs JS implementation
    else:
        source_score = 5   # Grammar — needs curation

    return rate_score + source_score


def greedy_select(rules, pair_metrics, n_select,
                  min_families=6, rate_bins=5, max_per_family=None):
    """
    Greedy selection with diversity constraints.

    At each step, add the rule that maximizes:
    1. Individual rule score
    2. Family diversity bonus (new family → big bonus)
    3. Base rate diversity bonus (new rate bin → bonus)
    4. Average pairing quality with already-selected rules

    Args:
        rules: list of rule dicts with 'predicate', 'base_rate', etc.
        pair_metrics: dict mapping (i, j) -> {overlap, a_only, b_only, neither}
        n_select: target number of rules to select
        min_families: minimum number of distinct families
        rate_bins: number of base rate bins for diversity
        max_per_family: max rules from one family (None = unlimited)
    """
    n = len(rules)
    selected = []
    selected_set = set()

    # Precompute individual scores
    indiv_scores = [score_rule(r) for r in rules]

    # Rate bins for diversity: [0-5%, 5-10%, 10-20%, 20-35%, 35-50%]
    bin_edges = [0, 0.05, 0.10, 0.20, 0.35, 0.50]

    def get_rate_bin(rate):
        if rate is None:
            return -1
        for b in range(len(bin_edges) - 1):
            if rate < bin_edges[b + 1]:
                return b
        return len(bin_edges) - 1

    while len(selected) < n_select and len(selected) < n:
        best_score = -float('inf')
        best_idx = None

        selected_families = Counter(rules[i]["family"] for i in selected)
        selected_bins = set(get_rate_bin(rules[i]["base_rate"]) for i in selected)

        for i in range(n):
            if i in selected_set:
                continue

            rule = rules[i]

            # Max per family constraint
            if max_per_family and selected_families[rule["family"]] >= max_per_family:
                continue

            # Base: individual quality
            score = indiv_scores[i]

            # Family diversity: bonus for new families
            if rule["family"] not in selected_families:
                score += 25  # Big bonus for new family
            elif selected_families[rule["family"]] == 1:
                score += 5   # Small bonus for second member

            # Rate bin diversity
            rate_bin = get_rate_bin(rule["base_rate"])
            if rate_bin not in selected_bins:
                score += 15  # Bonus for covering new rate range

            # Pairing quality: average BC with selected rules
            # (lower BC = more distinct = better for pair selection later)
            if selected:
                bcs = []
                feasible_count = 0
                for j in selected:
                    key = (min(i, j), max(i, j))
                    if key in pair_metrics:
                        m = pair_metrics[key]
                        bc = bhattacharyya_coefficient(m["overlap"], m["a_only"], m["b_only"])
                        bcs.append(bc)
                        if pair_feasible(m["a_only"], m["b_only"]):
                            feasible_count += 1
                if bcs:
                    avg_bc = sum(bcs) / len(bcs)
                    # Reward low average BC (more distinct from selected rules)
                    score += 10 * (1 - avg_bc)
                    # Bonus for having many feasible pairs
                    score += 5 * (feasible_count / len(selected))

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)

    return selected


# ============================================================================
# Step 5: Analysis and reporting
# ============================================================================

def analyze_selection(rules, selected_indices, pair_metrics, label="Selection"):
    """Print analysis of a selected rule subset."""
    sel_rules = [rules[i] for i in selected_indices]

    print(f"\n{'='*80}")
    print(f"  {label}: {len(sel_rules)} rules")
    print(f"{'='*80}")

    # Family distribution
    families = Counter(r["family"] for r in sel_rules)
    print(f"\n  Families ({len(families)} distinct):")
    for fam, count in sorted(families.items(), key=lambda x: -x[1]):
        print(f"    {fam:<25} {count}")

    # Source distribution
    sources = Counter(r["source"] for r in sel_rules)
    print(f"\n  Sources: {dict(sources)}")

    # Base rate distribution
    rates = [r["base_rate"] for r in sel_rules if r["base_rate"] is not None]
    rates.sort()
    print(f"\n  Base rate range: {min(rates)*100:.1f}% — {max(rates)*100:.1f}%")
    print(f"  Base rate median: {rates[len(rates)//2]*100:.1f}%")

    bin_edges = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 100)]
    for lo, hi in bin_edges:
        count = sum(1 for r in rates if lo <= r * 100 < hi)
        bar = "█" * count
        print(f"    [{lo:>2}%-{hi:>3}%) {count:>3} {bar}")

    # Pairing analysis
    n = len(selected_indices)
    n_pairs = n * (n - 1) // 2
    feasible_pairs = 0
    bc_values = []

    for a_idx, i in enumerate(selected_indices):
        for b_idx in range(a_idx + 1, len(selected_indices)):
            j = selected_indices[b_idx]
            key = (min(i, j), max(i, j))
            if key in pair_metrics:
                m = pair_metrics[key]
                bc = bhattacharyya_coefficient(m["overlap"], m["a_only"], m["b_only"])
                bc_values.append(bc)
                if pair_feasible(m["a_only"], m["b_only"]):
                    feasible_pairs += 1

    print(f"\n  Pairing analysis:")
    print(f"    Total possible pairs: {n_pairs}")
    print(f"    Feasible pairs (both exclusive regions ≥ 0.5%): {feasible_pairs}")
    if bc_values:
        print(f"    BC range: {min(bc_values):.3f} — {max(bc_values):.3f}")
        print(f"    BC median: {sorted(bc_values)[len(bc_values)//2]:.3f}")

    # Print the selected rules
    print(f"\n  Selected rules (sorted by base rate):")
    print(f"  {'Rate':>6}  {'Family':<25} {'Source':<10} {'ID'}")
    print(f"  {'-'*6}  {'-'*25} {'-'*10} {'-'*40}")
    for r in sorted(sel_rules, key=lambda x: x["base_rate"] or 0):
        rate_s = f"{r['base_rate']*100:.1f}%" if r["base_rate"] else "  ???"
        print(f"  {rate_s:>6}  {r['family']:<25} {r['source']:<10} {r['id'][:50]}")

    return sel_rules


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Selection Optimization")
    parser.add_argument("--samples", type=int, default=50_000,
                        help="Monte Carlo samples for overlap estimation")
    parser.add_argument("--hand-size", type=int, default=6)
    parser.add_argument("--deck-size", type=int, default=52)
    parser.add_argument("--wide", type=int, default=32,
                        help="Number of rules for wide pilot")
    parser.add_argument("--deep", type=int, default=8,
                        help="Number of rules for deep pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="phase4_selections.csv")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 80)
    print("Phase 4: Selection Optimization")
    print("=" * 80)

    # 1. Build curated candidate pool
    print("\n--- Step 1: Building candidate pool ---")

    cat_rules = load_catalogue_rules()
    print(f"  Catalogue: {len(cat_rules)} rules")

    template_rules = load_template_candidates()
    print(f"  Template (novel): {len(template_rules)} rules")

    # Skip grammar-derived rules (raw lambda programs, need manual curation for
    # interpretability). But include LLM-brainstormed rules — they're interpretable.
    llm_rules = load_llm_candidates()
    print(f"  LLM (novel): {len(llm_rules)} rules")

    all_rules = cat_rules + template_rules + llm_rules
    print(f"  Total pool: {len(all_rules)} rules")

    # Filter: A1 (base rate ≥ 1%) and A2 soft (base rate ≤ 60%)
    filtered = [r for r in all_rules
                if r["base_rate"] is not None
                and r["base_rate"] >= 0.01
                and r["base_rate"] <= 0.60]
    print(f"  After A1/A2 filtering: {len(filtered)} rules")

    # Remove extensionally equivalent rules (same base rate ± tolerance)
    # More robust: use fingerprinting
    print("\n  Deduplicating by fingerprint...")
    deck = make_deck(args.deck_size)
    fp_hands = [tuple(random.sample(deck, args.hand_size)) for _ in range(500)]

    seen_fingerprints = {}
    unique_rules = []
    n_deduped = 0

    for r in filtered:
        bits = []
        for hand in fp_hands:
            try:
                result = r["predicate"](hand)
                bits.append('1' if result else '0')
            except Exception:
                bits.append('?')
        fp = ''.join(bits)

        if fp in seen_fingerprints:
            n_deduped += 1
            # Keep the one with better source (catalogue > template > grammar)
            existing = seen_fingerprints[fp]
            source_rank = {"catalogue": 0, "template": 1, "grammar": 2}
            if source_rank.get(r["source"], 9) < source_rank.get(existing["source"], 9):
                # Replace with better source
                unique_rules = [x for x in unique_rules if x["id"] != existing["id"]]
                unique_rules.append(r)
                seen_fingerprints[fp] = r
        else:
            seen_fingerprints[fp] = r
            unique_rules.append(r)

    print(f"  Removed {n_deduped} duplicates, {len(unique_rules)} unique rules remain")

    # 2. Compute pairwise overlaps
    print(f"\n--- Step 2: Computing pairwise overlaps ---")
    pair_metrics = compute_pairwise_overlaps(
        unique_rules, deck, args.hand_size, args.samples, seed=args.seed)
    print(f"  Computed {len(pair_metrics)} pair metrics")

    # 3. Select WIDE pilot
    print(f"\n--- Step 3: Selecting WIDE pilot ({args.wide} rules) ---")
    wide_indices = greedy_select(
        unique_rules, pair_metrics,
        n_select=args.wide,
        min_families=6,
        max_per_family=8  # Allow up to 8 per family for wide
    )
    wide_rules = analyze_selection(unique_rules, wide_indices, pair_metrics,
                                   label="WIDE PILOT")

    # 4. Select DEEP pilot (subset of wide, optimized for pairing)
    print(f"\n--- Step 4: Selecting DEEP pilot ({args.deep} rules) ---")
    # For deep pilot, select from the wide pool with stricter diversity
    deep_indices_in_wide = greedy_select(
        [unique_rules[i] for i in wide_indices],
        # Remap pair metrics to wide-local indices
        {(a, b): pair_metrics[(min(wide_indices[a], wide_indices[b]),
                               max(wide_indices[a], wide_indices[b]))]
         for a in range(len(wide_indices))
         for b in range(a + 1, len(wide_indices))
         if (min(wide_indices[a], wide_indices[b]),
             max(wide_indices[a], wide_indices[b])) in pair_metrics},
        n_select=args.deep,
        min_families=4,
        max_per_family=2  # Strict: at most 2 per family for deep
    )
    deep_global_indices = [wide_indices[i] for i in deep_indices_in_wide]
    deep_rules = analyze_selection(unique_rules, deep_global_indices, pair_metrics,
                                   label="DEEP PILOT")

    # 5. Write output CSV
    output_path = Path(__file__).parent / args.output
    fieldnames = ["id", "source", "family", "base_rate", "base_rate_pct",
                  "name", "description", "selection"]

    rows = []
    for r in wide_rules:
        sel = "wide+deep" if r in deep_rules else "wide"
        rows.append({
            "id": r["id"],
            "source": r["source"],
            "family": r["family"],
            "base_rate": r["base_rate"],
            "base_rate_pct": round(r["base_rate"] * 100, 2) if r["base_rate"] else "",
            "name": r.get("name", ""),
            "description": r.get("description", ""),
            "selection": sel,
        })

    rows.sort(key=lambda r: float(r["base_rate"]) if r["base_rate"] else 0)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} selected rules to {output_path}")

    # 6. Print feasible pair summary for deep pilot
    print(f"\n{'='*80}")
    print("DEEP PILOT — Feasible Pair Details")
    print(f"{'='*80}")

    deep_feasible = []
    for a_idx in range(len(deep_global_indices)):
        for b_idx in range(a_idx + 1, len(deep_global_indices)):
            i, j = deep_global_indices[a_idx], deep_global_indices[b_idx]
            key = (min(i, j), max(i, j))
            if key in pair_metrics:
                m = pair_metrics[key]
                bc = bhattacharyya_coefficient(m["overlap"], m["a_only"], m["b_only"])
                max_e = expected_samples_max(m["overlap"], m["a_only"], m["b_only"])
                feas = pair_feasible(m["a_only"], m["b_only"])

                ra = unique_rules[i]
                rb = unique_rules[j]
                deep_feasible.append({
                    "name_a": ra["id"][:30],
                    "name_b": rb["id"][:30],
                    "family_a": ra["family"],
                    "family_b": rb["family"],
                    "a_only": m["a_only"],
                    "b_only": m["b_only"],
                    "overlap": m["overlap"],
                    "bc": bc,
                    "max_e": max_e,
                    "feasible": feas,
                })

    deep_feasible.sort(key=lambda p: p["bc"], reverse=True)

    feas_count = sum(1 for p in deep_feasible if p["feasible"])
    print(f"\n  {feas_count} feasible / {len(deep_feasible)} total pairs\n")

    print(f"  {'Rule A':<32} {'Rule B':<32} {'A-only':>7} {'B-only':>7} {'Ovlp':>7} {'BC':>6} {'maxE':>6} {'Feas'}")
    print(f"  {'-'*32} {'-'*32} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*4}")

    for p in deep_feasible[:20]:
        me = f"{p['max_e']:.1f}" if not math.isinf(p["max_e"]) else "∞"
        f_mark = "✓" if p["feasible"] else "✗"
        print(f"  {p['name_a']:<32} {p['name_b']:<32} "
              f"{p['a_only']*100:>6.1f}% {p['b_only']*100:>6.1f}% "
              f"{p['overlap']*100:>6.1f}% {p['bc']:>5.3f} {me:>6} {f_mark}")


if __name__ == "__main__":
    main()
