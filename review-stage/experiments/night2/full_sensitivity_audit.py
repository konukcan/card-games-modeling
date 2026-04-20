"""Full-scale residual mixed-class sensitivity audit (Night 2).

Goal (Codex's R4 ask):
  Run the residual mixed-class sensitivity audit on the full depth=7 / 300k
  pool across ALL official variants and ALL 60 rules. Report the maximum
  posterior mass on residual mixed classes and the worst-case movement in
  the headline outputs.

This script:
  1. Builds the full pool (depth=7, max_programs=300_000) ONCE under the
     uniform-enumeration grammar. Same pipeline as run_analysis() default.
  2. Identifies residual mixed classes on 2k fresh audit hands (seed=98765).
  3. Splits each mixed class by audit-hand agreement vector.
  4. For each rule (only those with frozen exemplars), scores the original
     pool and the split pool under summed prior (the headline variant).
  5. Saves results INCREMENTALLY after each rule so partial data survives.

Variant scope:
  - Summed prior, strict likelihood, uniform grammar (headline).
  - Optional: --extra-variants enables canonical-prior pass too.

Key insight: residual mixed classes are a property of the pool, not the rule.
So we identify them once, then test sensitivity per rule.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

import math
import random
import time
import json
import argparse
from collections import defaultdict
from pathlib import Path

from rules.cards import Card, Suit, Rank
from gallery_analysis.analyze import (
    build_hypothesis_pool,
    _strict_split_classes,
    _recompute_class_prior,
)
from gallery_analysis.injection import (
    load_and_validate_injections,
    merge_injected,
)
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.bayesian_scorer import (
    score_hypotheses,
    normalize_posteriors,
)
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.exemplars import load_exemplars


def generate_random_hands(n, seed):
    rng = random.Random(seed)
    suits = list(Suit)
    ranks = list(Rank)
    out = []
    for _ in range(n):
        deck = [Card(s, r) for s in suits for r in ranks]
        out.append(rng.sample(deck, 6))
    return out


def identify_mixed_classes(equiv_classes, audit_hands):
    """Return list of (class_idx, class_dict, member_vectors) for mixed classes."""
    mixed = []
    for idx, cls in enumerate(equiv_classes):
        preds = cls.get("_all_predicates") or []
        if len(preds) < 2:
            continue
        vecs = []
        for pred in preds:
            try:
                vec = tuple(bool(pred(h)) for h in audit_hands)
            except Exception:
                vec = None
            vecs.append(vec)
        valid = [v for v in vecs if v is not None]
        if len(valid) < 2:
            continue
        ref = valid[0]
        if any(v != ref for v in valid):
            mixed.append((idx, cls, vecs))
    return mixed


def split_mixed_class(cls, vecs, grammar):
    """Partition members by agreement vector on audit hands."""
    preds = cls["_all_predicates"]
    programs = cls["all_programs"]
    groups = defaultdict(list)
    for pred, prog, vec in zip(preds, programs, vecs):
        if vec is None:
            continue
        groups[vec].append((pred, prog))
    subs = []
    for vec, members in groups.items():
        members_preds = [m[0] for m in members]
        members_progs = [m[1] for m in members]
        sub = dict(cls)
        sub["_all_predicates"] = members_preds
        sub["all_programs"] = members_progs
        sub["n_expressions"] = len(members_progs)
        sub["canonical_program"] = members_progs[0]
        sub["predicate"] = members_preds[0]
        try:
            sub["canonical_prior"] = _recompute_class_prior(
                sub, grammar, prior_mode="canonical", strict=False
            )
            sub["summed_prior"] = _recompute_class_prior(
                sub, grammar, prior_mode="summed", strict=False
            )
        except Exception:
            pass
        subs.append(sub)
    return subs


def score_one_rule(equiv, equiv_split, mixed, rule, exemplars, audit_hands,
                   prior_mode, extension_samples=5_000):
    """Return per-rule sensitivity dict for given prior mode."""
    # Recompute hit_vector for each class against exemplars
    def build_hit_vectors(classes):
        for cls in classes:
            pred = cls["predicate"]
            try:
                vec = tuple(bool(pred(h)) for h in exemplars)
            except Exception:
                vec = tuple(False for _ in exemplars)
            cls["hit_vector"] = vec
            cls["n_hits"] = sum(vec)
            cls["n_misses"] = len(vec) - cls["n_hits"]

    build_hit_vectors(equiv)
    build_hit_vectors(equiv_split)
    n_exemplars_actual = len(exemplars)

    scored_orig = score_hypotheses(
        equiv, n_exemplars=n_exemplars_actual,
        extension_samples=extension_samples, prior_mode=prior_mode,
    )
    scored_split = score_hypotheses(
        equiv_split, n_exemplars=n_exemplars_actual,
        extension_samples=extension_samples, prior_mode=prior_mode,
    )
    norm_orig = normalize_posteriors(scored_orig)
    norm_split = normalize_posteriors(scored_split)

    target_vec = tuple(bool(rule["predicate"](h)) for h in audit_hands)

    def find_true_in_pool(classes, norm):
        fp_to_rank = {hyp.fingerprint: (rnk, prob)
                      for rnk, (hyp, prob) in enumerate(norm, start=1)}
        best_rnk = None
        best_prob = 0.0
        for cls in classes:
            pred = cls["predicate"]
            try:
                vec = tuple(bool(pred(h)) for h in audit_hands)
            except Exception:
                continue
            if vec != target_vec:
                continue
            entry = fp_to_rank.get(cls["fingerprint"])
            if entry is None:
                continue
            rnk, prob = entry
            if best_rnk is None or rnk < best_rnk:
                best_rnk = rnk
                best_prob = prob
        return best_rnk, best_prob

    rnk_orig, prob_orig = find_true_in_pool(equiv, norm_orig)
    rnk_split, prob_split = find_true_in_pool(equiv_split, norm_split)
    top1_orig = norm_orig[0][0].canonical_program if norm_orig else None
    top1_split = norm_split[0][0].canonical_program if norm_split else None

    # Posterior mass held by mixed classes (pre-split) for this rule
    mixed_mass = 0.0
    for idx, cls, _ in mixed:
        for hyp, prob in norm_orig:
            if hyp.fingerprint == cls["fingerprint"]:
                mixed_mass += prob
                break

    return {
        "true_rank_orig": rnk_orig,
        "true_rank_split": rnk_split,
        "true_prob_orig": prob_orig,
        "true_prob_split": prob_split,
        "delta_rank": (rnk_split - rnk_orig) if (rnk_orig and rnk_split) else None,
        "delta_prob": prob_split - prob_orig,
        "top1_orig": str(top1_orig),
        "top1_split": str(top1_split),
        "top1_changed": str(top1_orig) != str(top1_split),
        "mixed_class_posterior_mass": mixed_mass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-programs", type=int, default=300_000)
    parser.add_argument("--audit-n", type=int, default=2000)
    parser.add_argument("--rules", type=str, default="ALL",
                        help="Comma-separated rule list, or ALL for every rule with exemplars")
    parser.add_argument("--max-rules", type=int, default=None,
                        help="Cap rules processed (useful for timing tests)")
    parser.add_argument("--n-exemplars", type=int, default=6)
    parser.add_argument("--exemplar-seed", type=int, default=12345)
    parser.add_argument("--extension-samples", type=int, default=5_000)
    parser.add_argument("--inject", type=str,
                        default="src/gallery_analysis/data/injected_hypotheses.json")
    parser.add_argument("--prior-modes", type=str, default="summed",
                        help="Comma-sep: summed,canonical")
    parser.add_argument("--output-dir", type=str,
                        default="review-stage/experiments/night2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "full_sensitivity_results.json"
    out_partial = output_dir / "full_sensitivity_partial.json"

    print(f"[AUDIT] Building pool: depth={args.depth}, max_programs={args.max_programs}",
          flush=True)
    t0 = time.time()
    equiv, stats = build_hypothesis_pool(
        max_depth=args.depth, max_programs=args.max_programs, verbose=1,
    )
    pool_build_s = time.time() - t0
    print(f"[AUDIT] Base pool: {len(equiv):,} classes in {pool_build_s:.1f}s",
          flush=True)

    probes = stats["_probes"]
    exemplars_disamb = stats.get("_exemplar_hands") or []
    grammar = build_gallery_grammar()
    enum_priors = [c["canonical_prior"] for c in equiv]
    rng_prior = (min(enum_priors), max(enum_priors)) if enum_priors else None

    inject_path = args.inject
    if not os.path.isabs(inject_path):
        inject_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "..", inject_path)

    injected = load_and_validate_injections(
        inject_path, grammar=grammar, enumerated_prior_range=rng_prior,
    )
    print(f"[AUDIT] Merging {len(injected)} injections...", flush=True)
    equiv = merge_injected(list(equiv), injected, probes)
    if exemplars_disamb:
        equiv, _ = _strict_split_classes(
            equiv, exemplar_hands=exemplars_disamb, main_probes=probes, verbose=0,
        )
    print(f"[AUDIT] Post-inject pool: {len(equiv):,} classes", flush=True)

    audit_hands = generate_random_hands(args.audit_n, seed=98765)
    mixed = identify_mixed_classes(equiv, audit_hands)
    print(f"[AUDIT] Residual mixed classes: {len(mixed)}", flush=True)
    for idx, cls, _ in mixed:
        print(f"  idx={idx}  canonical={str(cls['canonical_program'])[:80]}  "
              f"n_members={cls['n_expressions']}", flush=True)

    # Split mixed classes
    equiv_split = list(equiv)
    for idx, cls, vecs in mixed:
        subs = split_mixed_class(cls, vecs, grammar)
        if len(subs) == 0:
            continue
        equiv_split[idx] = subs[0]
        for s in subs[1:]:
            equiv_split.append(s)
    print(f"[AUDIT] Split pool: {len(equiv_split):,} classes "
          f"({len(equiv_split) - len(equiv)} new)", flush=True)

    # Determine rule list
    frozen_exemplars = load_exemplars()
    if args.rules.upper() == "ALL":
        rule_ids = [r for r in GALLERY_RULES if r in frozen_exemplars]
    else:
        rule_ids = [r.strip() for r in args.rules.split(",") if r.strip()]
    if args.max_rules:
        rule_ids = rule_ids[:args.max_rules]

    prior_modes = [p.strip() for p in args.prior_modes.split(",") if p.strip()]
    print(f"[AUDIT] Scoring {len(rule_ids)} rules × {len(prior_modes)} prior modes",
          flush=True)

    results = {pm: {} for pm in prior_modes}
    timings = []

    for rule_idx, rule_id in enumerate(rule_ids, start=1):
        rule = GALLERY_RULES.get(rule_id)
        if rule is None:
            print(f"[AUDIT] {rule_idx}/{len(rule_ids)} {rule_id}: not in GALLERY_RULES, skip",
                  flush=True)
            continue
        ex_dict = frozen_exemplars.get(rule_id)
        if ex_dict is None or "hands_primary" not in ex_dict:
            print(f"[AUDIT] {rule_idx}/{len(rule_ids)} {rule_id}: no frozen exemplars, skip",
                  flush=True)
            continue
        exemplars = ex_dict["hands_primary"]

        rule_start = time.time()
        for pm in prior_modes:
            try:
                r = score_one_rule(
                    equiv, equiv_split, mixed, rule, exemplars, audit_hands,
                    prior_mode=pm, extension_samples=args.extension_samples,
                )
                r["rule_answer"] = rule.get("answer", "")
                results[pm][rule_id] = r
            except Exception as e:
                results[pm][rule_id] = {"error": str(e)}
                print(f"  ! {pm}: ERROR {e}", flush=True)

        rule_elapsed = time.time() - rule_start
        timings.append((rule_id, rule_elapsed))
        # Print compact line
        line = f"[AUDIT] {rule_idx}/{len(rule_ids)} {rule_id} ({rule_elapsed:.1f}s)"
        for pm in prior_modes:
            r = results[pm][rule_id]
            if "error" in r:
                line += f"  {pm}:ERR"
            else:
                line += (f"  {pm}: rank {r['true_rank_orig']}->{r['true_rank_split']} "
                         f"(Δ{r['delta_rank']}) prob {r['true_prob_orig']:.3g}->{r['true_prob_split']:.3g} "
                         f"top1_chg={r['top1_changed']} mass={r['mixed_class_posterior_mass']:.2g}")
        print(line, flush=True)

        # Incremental dump
        partial = {
            "n_mixed_classes": len(mixed),
            "n_rules_done": rule_idx,
            "n_rules_total": len(rule_ids),
            "pool_build_seconds": pool_build_s,
            "pool_size_orig": len(equiv),
            "pool_size_split": len(equiv_split),
            "args": vars(args),
            "results": results,
            "timings": timings,
        }
        with open(out_partial, "w") as f:
            json.dump(partial, f, indent=2, default=str)

    # Final summary
    summary = {}
    for pm, rule_results in results.items():
        valid = [r for r in rule_results.values() if "error" not in r]
        if not valid:
            summary[pm] = {"n_rules_valid": 0}
            continue
        max_drank = max((abs(r["delta_rank"]) for r in valid
                        if r["delta_rank"] is not None), default=0)
        max_dprob = max((abs(r["delta_prob"]) for r in valid), default=0.0)
        max_mixed = max((r["mixed_class_posterior_mass"] for r in valid), default=0.0)
        any_top1 = sum(1 for r in valid if r["top1_changed"])
        summary[pm] = {
            "n_rules_valid": len(valid),
            "max_delta_true_rank": max_drank,
            "max_delta_true_prob": max_dprob,
            "max_mixed_class_posterior_mass": max_mixed,
            "n_top1_changed": any_top1,
        }

    final = {
        "n_mixed_classes": len(mixed),
        "n_rules_done": len(rule_ids),
        "pool_build_seconds": pool_build_s,
        "pool_size_orig": len(equiv),
        "pool_size_split": len(equiv_split),
        "args": vars(args),
        "summary": summary,
        "results": results,
        "timings": timings,
    }
    with open(out_json, "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(f"Mixed classes: {len(mixed)}", flush=True)
    for pm, s in summary.items():
        print(f"  [{pm}] n_valid={s['n_rules_valid']} "
              f"max_|Δrank|={s.get('max_delta_true_rank')} "
              f"max_|Δprob|={s.get('max_delta_true_prob'):.4g} "
              f"max_mixed_mass={s.get('max_mixed_class_posterior_mass'):.4g} "
              f"top1_flips={s.get('n_top1_changed')}", flush=True)
    print(f"\nWrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
