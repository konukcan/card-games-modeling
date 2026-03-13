"""
Depth-stratified posterior mass analysis.

For each rule, compute what fraction of posterior mass comes from hypotheses
at each AST depth level (1–6). This reveals whether the posterior is dominated
by shallow/simple hypotheses or whether deeper programs contribute meaningfully,
informing whether depth 7 enumeration is worth the cost.

Also produces detailed confusion profiles for selected rules.

Usage:
    python3 gallery_analysis/depth_mass_analysis.py \
        --results gallery_analysis/results/depth6_injected_v2.json \
        --extension-cache gallery_analysis/results/extension_cache_depth6.json \
        --verbose 2
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import (
    GALLERY_RULES,
    load_exemplars,
    estimate_extensions,
    build_hypothesis_pool,
    compute_log_likelihood_noisy,
    compute_log_likelihood_strict,
)
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.exemplars import generate_probe_set


def ast_depth(program_str: str) -> int:
    """Compute AST depth as max parenthesis nesting in an s-expression."""
    max_d = 0
    d = 0
    for c in program_str:
        if c == '(':
            d += 1
            max_d = max(max_d, d)
        elif c == ')':
            d -= 1
    return max_d


def compute_depth_mass_table(
    equivalence_classes: List[Dict],
    extensions: List[Tuple[int, float]],
    exemplars: Dict,
    epsilon: float = 0.01,
    prior_mode: str = "summed",
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    For each rule, compute posterior mass stratified by AST depth.

    Returns a dict keyed by rule_id, each containing:
    - depth_mass: {depth: fraction_of_posterior} for depths 1–6+
    - cumulative_mass: {depth: cumulative_fraction} (mass at depth <= d)
    - top_competitors: list of top 20 hypotheses with depth, mass, program
    - true_rule_rank, true_rule_mass
    - confusion_profile: for top competitors, per-exemplar agreement
    """
    # Precompute AST depth for each equivalence class's canonical program
    depths = [ast_depth(cls["canonical_program"]) for cls in equivalence_classes]

    # Build true-rule fingerprint lookup
    true_rule_fps = {}
    for cls in equivalence_classes:
        true_for_list = cls.get("true_for_rules", [])
        if not true_for_list:
            single = cls.get("true_for_rule")
            if single:
                true_for_list = [single]
        for rule_id in true_for_list:
            true_rule_fps[rule_id] = cls["fingerprint"]

    results = {}
    n_rules = len(GALLERY_RULES)

    for rule_idx, (rule_id, rule_info) in enumerate(GALLERY_RULES.items()):
        if rule_id not in exemplars:
            continue

        exemplar_hands = exemplars[rule_id]["hands_primary"]
        n_exemplars = len(exemplar_hands)

        # Score all hypotheses
        scored = []  # (log_posterior, depth, cls_idx, n_hits, hit_vector)
        for i, (cls, (ext_size, base_rate)) in enumerate(
            zip(equivalence_classes, extensions)
        ):
            pred = cls["predicate"]
            n_hits = 0
            hit_vector = []
            for hand in exemplar_hands:
                try:
                    result = pred(hand)
                    hit_vector.append(result)
                    if result:
                        n_hits += 1
                except Exception:
                    hit_vector.append(False)

            log_lik = compute_log_likelihood_noisy(
                n_hits, n_exemplars, ext_size, epsilon
            )

            if prior_mode == "canonical":
                log_prior = cls["canonical_prior"]
            else:
                log_prior = cls["summed_prior"]

            log_post = log_prior + log_lik
            scored.append((log_post, depths[i], i, n_hits, hit_vector))

        # Normalize posteriors
        scored.sort(key=lambda x: -x[0])
        max_lp = scored[0][0]
        log_norm = max_lp + math.log(
            sum(math.exp(s[0] - max_lp) for s in scored)
        )
        normalized = [
            (math.exp(s[0] - log_norm), s[1], s[2], s[3], s[4])
            for s in scored
        ]

        # Stratify by depth
        depth_mass = defaultdict(float)
        for prob, depth, _, _, _ in normalized:
            depth_mass[depth] += prob

        # Cumulative mass
        max_depth = max(depth_mass.keys()) if depth_mass else 6
        cumulative = {}
        running = 0.0
        for d in range(1, max_depth + 1):
            running += depth_mass.get(d, 0.0)
            cumulative[d] = running

        # Find true rule
        true_fp = true_rule_fps.get(rule_id)
        true_rank = None
        true_mass = None
        for rank, (prob, depth, cls_idx, n_hits, hit_vec) in enumerate(normalized):
            if true_fp and equivalence_classes[cls_idx]["fingerprint"] == true_fp:
                true_rank = rank + 1
                true_mass = prob
                break

        # Top 20 competitors with confusion profiles
        top_competitors = []
        for prob, depth, cls_idx, n_hits, hit_vec in normalized[:20]:
            cls = equivalence_classes[cls_idx]
            is_true = (true_fp and cls["fingerprint"] == true_fp)
            top_competitors.append({
                "program": cls["canonical_program"],
                "probability": prob,
                "depth": depth,
                "n_hits": n_hits,
                "n_exemplars": n_exemplars,
                "agrees_on_exemplars": hit_vec,
                "n_agreements": sum(hit_vec),
                "extension_size": extensions[cls_idx][0],
                "base_rate": extensions[cls_idx][1],
                "is_true_rule": is_true,
                "log_prior": (cls["summed_prior"] if prior_mode == "summed"
                              else cls["canonical_prior"]),
                "n_expressions": cls["n_expressions"],
                "source": cls.get("source", "enumerated"),
            })

        results[rule_id] = {
            "group": rule_info["group"],
            "depth_mass": dict(depth_mass),
            "cumulative_mass": cumulative,
            "true_rule_rank": true_rank,
            "true_rule_mass": true_mass,
            "top_competitors": top_competitors,
        }

        if verbose >= 2 and (rule_idx + 1) % 10 == 0:
            print(f"  {rule_idx+1}/{n_rules} rules scored...", flush=True)

    return results


def print_depth_mass_report(results: Dict[str, Any], verbose: int = 1):
    """Print the depth-stratified mass table and confusion profiles."""

    # Sort by group, then entropy-like ordering (use true_rule_rank as proxy)
    sorted_rules = sorted(
        results.items(),
        key=lambda x: (x[1]["group"], -(x[1]["true_rule_rank"] or 99999))
    )

    print("\n" + "=" * 100)
    print("DEPTH-STRATIFIED POSTERIOR MASS ANALYSIS")
    print("=" * 100)

    # Header for depth mass table
    print(f"\n{'Rule':<35} {'Grp':>3} {'d1':>6} {'d2':>6} {'d3':>6} "
          f"{'d4':>6} {'d5':>6} {'d6':>6} {'TrRk':>6} {'TrMass':>7}")
    print("─" * 100)

    # Group by difficulty group
    for group in [1, 2, 3]:
        group_rules = [(rid, r) for rid, r in sorted_rules if r["group"] == group]
        # Sort within group by true_rule_rank
        group_rules.sort(
            key=lambda x: x[1]["true_rule_rank"] if x[1]["true_rule_rank"] else 99999
        )

        labels = {1: "Easy", 2: "Medium", 3: "Hard"}
        print(f"\n  --- Group {group} ({labels[group]}) ---")

        for rule_id, r in group_rules:
            dm = r["depth_mass"]
            tr = r["true_rule_rank"]
            tm = r["true_rule_mass"]
            tr_str = str(tr) if tr else "N/A"
            tm_str = f"{tm*100:.1f}%" if tm else "N/A"

            depths_str = ""
            for d in range(1, 7):
                mass = dm.get(d, 0.0)
                if mass >= 0.005:
                    depths_str += f"{mass*100:5.1f}%"
                else:
                    depths_str += f"    - "

            print(f"  {rule_id:<33} {group:>3} {depths_str} {tr_str:>6} {tm_str:>7}")

        # Group summary
        group_depth_means = defaultdict(list)
        for _, r in group_rules:
            for d in range(1, 7):
                group_depth_means[d].append(r["depth_mass"].get(d, 0.0))

        print(f"  {'MEAN':<33} {'':>3}", end="")
        for d in range(1, 7):
            vals = group_depth_means[d]
            mean = sum(vals) / len(vals) if vals else 0
            print(f"{mean*100:5.1f}%", end="")
        print()

    # Now print detailed confusion profiles for interesting rules
    if verbose >= 2:
        print("\n" + "=" * 100)
        print("DETAILED CONFUSION PROFILES")
        print("=" * 100)

        for rule_id, r in sorted_rules:
            tr = r["true_rule_rank"]
            # Print profiles for rules with interesting confusion (rank > 1 but < 100,
            # OR specifically requested categories)
            if tr is not None and 1 < tr <= 100:
                _print_confusion_profile(rule_id, r)
            elif tr is not None and tr == 1 and r["true_rule_mass"] < 0.99:
                _print_confusion_profile(rule_id, r)

        # Also print a few "hard" cases (rank > 1000)
        print("\n  --- Selected Hard Cases ---")
        hard_cases = [
            (rid, r) for rid, r in sorted_rules
            if r["true_rule_rank"] and r["true_rule_rank"] > 1000
        ]
        # Pick a few representative ones
        for rule_id, r in hard_cases[:6]:
            _print_confusion_profile(rule_id, r)


def _print_confusion_profile(rule_id: str, r: Dict):
    """Print detailed confusion profile for one rule."""
    print(f"\n{'─'*70}")
    tr = r["true_rule_rank"]
    tm = r["true_rule_mass"]
    print(f"{rule_id}  (group={r['group']}, true_rank={tr}, "
          f"true_mass={tm*100:.2f}%)")

    for i, comp in enumerate(r["top_competitors"][:10]):
        agree_str = "".join(
            "Y" if a else "n" for a in comp["agrees_on_exemplars"]
        )
        true_marker = " ← TRUE" if comp["is_true_rule"] else ""
        source_tag = f"[{comp['source']}]" if comp['source'] != 'enumerated' else ""
        print(
            f"  {i+1:>2}. ({comp['probability']*100:5.1f}%) "
            f"d={comp['depth']}  "
            f"hits={comp['n_agreements']}/{comp['n_exemplars']}  "
            f"agree=[{agree_str}]  "
            f"ext={comp['extension_size']:>8,}  "
            f"prior={comp['log_prior']:>7.2f}  "
            f"n_expr={comp['n_expressions']:>4}  "
            f"{comp['program'][:55]}"
            f"{true_marker} {source_tag}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Depth-stratified posterior mass analysis"
    )
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-programs", type=int, default=500_000)
    parser.add_argument("--inject", type=str, default=None)
    parser.add_argument("--extension-cache", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--max-list-chain", type=int, default=2,
                        help="Max consecutive list→list transforms (default 2)")
    parser.add_argument("--no-list-chain-limit", action="store_true",
                        help="Disable list→list chain limit")
    args = parser.parse_args()

    max_list_chain = None if args.no_list_chain_limit else args.max_list_chain

    print("=" * 70)
    print("DEPTH-STRATIFIED POSTERIOR MASS ANALYSIS")
    print("=" * 70)

    # Steps 1-3: Build hypothesis pool (enumerate + filter + fingerprint)
    equiv_classes, pipeline_stats = build_hypothesis_pool(
        max_depth=args.depth,
        max_programs=args.max_programs,
        max_list_chain=max_list_chain,
        verbose=args.verbose,
    )
    print(f"  {len(equiv_classes):,} equivalence classes", flush=True)

    # Step 3b: Inject
    if args.inject:
        grammar = build_gallery_grammar()
        print(f"\nInjecting from {args.inject}...", flush=True)
        probes = generate_probe_set(500, seed=42)
        injected = load_and_validate_injections(args.inject, grammar=grammar)
        equiv_classes = merge_injected(equiv_classes, injected, probes)

    # Step 3: Extension sizes
    print(f"\nStep 3: Extension sizes...", flush=True)
    extensions = estimate_extensions(
        equiv_classes, verbose=args.verbose, cache_path=args.extension_cache,
    )

    # Step 4: Depth-mass analysis
    print(f"\nStep 4: Computing depth-stratified posteriors...", flush=True)
    exemplars = load_exemplars()
    results = compute_depth_mass_table(
        equiv_classes, extensions, exemplars,
        verbose=args.verbose,
    )

    # Print report
    print_depth_mass_report(results, verbose=args.verbose)

    # Save results
    if args.output:
        # Make JSON-serializable
        out = {}
        for rule_id, r in results.items():
            out[rule_id] = {
                "group": r["group"],
                "depth_mass": {str(k): v for k, v in r["depth_mass"].items()},
                "cumulative_mass": {str(k): v for k, v in r["cumulative_mass"].items()},
                "true_rule_rank": r["true_rule_rank"],
                "true_rule_mass": r["true_rule_mass"],
                "top_competitors": [
                    {k: v for k, v in comp.items() if k != "agrees_on_exemplars"}
                    | {"agrees_on_exemplars": comp["agrees_on_exemplars"]}
                    for comp in r["top_competitors"]
                ],
            }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
