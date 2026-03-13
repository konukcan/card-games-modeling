#!/usr/bin/env python3
"""
Quick validation: compare uniform vs weighted grammar across all 60 rules.

Usage:
    cd src
    python -m gallery_analysis.validate_weights [--depth 6] [--max-programs 500000]

Outputs a side-by-side table showing:
  - True rule rank under uniform vs weighted
  - Entropy under uniform vs weighted
  - Top-1 program under each grammar
  - Count of rules where has_suit/has_color is top-1
"""
import sys
import time
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import (
    build_hypothesis_pool, estimate_extensions, score_rule,
)
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.enumerator import (
    build_gallery_grammar, build_weighted_gallery_grammar,
)
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.gallery_rules import GALLERY_RULES

# Shallow primitives to track
_SHALLOW_PRIMS = {'has_suit', 'has_color'}


def _is_shallow_top1(program_str: str) -> bool:
    """Check if the top-1 program starts with a shallow primitive."""
    for p in _SHALLOW_PRIMS:
        if p in program_str and program_str.count('(') <= 2:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Validate weighted grammar")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-programs", type=int, default=500_000)
    args = parser.parse_args()

    SRC = Path(__file__).parent.parent
    inject_path = SRC / "gallery_analysis" / "data" / "injected_hypotheses.json"

    print("=" * 90)
    print("GRAMMAR VALIDATION: Uniform vs Weighted (4-tier)")
    print("=" * 90)

    # Build hypothesis pool
    print(f"\nBuilding hypothesis pool (depth={args.depth}, max={args.max_programs:,})...")
    t0 = time.time()
    equiv_classes, stats = build_hypothesis_pool(
        max_depth=args.depth, max_programs=args.max_programs,
        max_list_chain=2, verbose=1,
    )
    print(f"  {len(equiv_classes):,} classes ({time.time()-t0:.0f}s)")

    # Inject
    if inject_path.exists():
        grammar_tmp = build_gallery_grammar()
        probes = generate_probe_set(500, seed=42)
        injected = load_and_validate_injections(str(inject_path), grammar=grammar_tmp)
        equiv_classes = merge_injected(equiv_classes, injected, probes)
        print(f"  After injection: {len(equiv_classes):,} classes")

    # Extensions
    extensions = estimate_extensions(
        equiv_classes, verbose=1,
    )

    # True-rule fingerprints
    true_fps = {}
    for cls in equiv_classes:
        for rid in cls.get("true_for_rules", []):
            true_fps[rid] = cls["fingerprint"]
        single = cls.get("true_for_rule")
        if single and single not in true_fps:
            true_fps[single] = cls["fingerprint"]

    # Build grammars
    g_weighted = build_weighted_gallery_grammar()

    # Score all rules under both grammars
    exemplars = load_exemplars()

    print(f"\n{'Rule':<30} {'Grp':>3} | "
          f"{'U_rank':>6} {'U_ent':>5} {'U_top1':<40} | "
          f"{'W_rank':>6} {'W_ent':>5} {'W_top1':<40}")
    print("-" * 150)

    n_shallow_uniform = 0
    n_shallow_weighted = 0
    n_improved = 0
    n_worsened = 0
    n_unchanged = 0

    for rule_id, rule_info in sorted(GALLERY_RULES.items(), key=lambda x: x[1]["group"]):
        if rule_id not in exemplars:
            continue

        hands = exemplars[rule_id]["hands_primary"]
        true_fp = true_fps.get(rule_id)

        # Score under uniform (grammar=None uses stored priors)
        r_u = score_rule(rule_id, hands, equiv_classes, extensions,
                         true_rule_fingerprint=true_fp, grammar=None)

        # Score under weighted
        r_w = score_rule(rule_id, hands, equiv_classes, extensions,
                         true_rule_fingerprint=true_fp, grammar=g_weighted)

        # Extract metrics
        u_rank = r_u.get("true_rule_rank") or 0
        w_rank = r_w.get("true_rule_rank") or 0
        u_ent = r_u["difficulty"]["posterior_entropy"]
        w_ent = r_w["difficulty"]["posterior_entropy"]
        u_top1 = r_u["top_hypotheses"][0]["program"][:38] if r_u["top_hypotheses"] else "?"
        w_top1 = r_w["top_hypotheses"][0]["program"][:38] if r_w["top_hypotheses"] else "?"

        u_rank_str = str(u_rank) if u_rank else "N/A"
        w_rank_str = str(w_rank) if w_rank else "N/A"

        if _is_shallow_top1(u_top1):
            n_shallow_uniform += 1
        if _is_shallow_top1(w_top1):
            n_shallow_weighted += 1

        if u_rank and w_rank:
            if w_rank < u_rank:
                n_improved += 1
            elif w_rank > u_rank:
                n_worsened += 1
            else:
                n_unchanged += 1

        print(f"  {rule_id:<28} {rule_info['group']:>3} | "
              f"{u_rank_str:>6} {u_ent:5.2f} {u_top1:<40} | "
              f"{w_rank_str:>6} {w_ent:5.2f} {w_top1:<40}")

    print(f"\n{'-'*90}")
    print(f"SUMMARY:")
    print(f"  Shallow (has_suit/has_color) as top-1: {n_shallow_uniform} -> {n_shallow_weighted}")
    print(f"  True rule rank: {n_improved} improved, {n_worsened} worsened, {n_unchanged} unchanged")


if __name__ == "__main__":
    main()
