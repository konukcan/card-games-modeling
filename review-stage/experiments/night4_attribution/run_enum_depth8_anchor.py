"""
Night 4 — d8 exhaustive reference for the all_same_suit anchor rule.

Single-rule, large-budget enumeration that serves as a near-saturated "ground
truth" posterior against which the d6 and d7 pools can be checked. all_same_suit
was Night 3's tightest agreement (mass_mapped 0.9897), so a d8 reference answers:

  * Did d6/d7 already include every class that carries posterior mass for this
    rule? (If so: d8_retained_mass ≈ d7_retained_mass ≈ d6_retained_mass.)
  * Are there depth-7-only or depth-8-only classes that move the posterior?
    (If so: d8's direct-fp universe contains new fingerprints with non-trivial
    mass.)

Fork of run_enum_depth7.py, with two specializations:
  1. Reads config from CONFIG["enumeration_d8_anchor"] (single-rule).
  2. RULE_IDS is the single rule specified in that block.

Everything else — adaptive extension ladder, posterior computation, JSON
sidecars — matches the d6 and d7 runners exactly, so fingerprints are
directly comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import cloudpickle as pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from gallery_analysis.analyze import build_hypothesis_pool
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.hand_diagnosticity import compute_posteriors_for_rule
from gallery_analysis.hypothesis_table import estimate_extension_size


CONFIG = json.loads((HERE / "config.json").read_text())
ANCHOR_CFG = CONFIG["enumeration_d8_anchor"]
RULE_IDS = [ANCHOR_CFG["rule"]]
OUT_DIR = HERE / ANCHOR_CFG["output_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "posteriors").mkdir(parents=True, exist_ok=True)


def estimate_extensions_adaptive(
    equiv_classes: List[Dict[str, Any]],
    base_probes: int,
    escalated_probes: int,
    base_rate_threshold: float,
    seed: int = 123,
    verbose: bool = True,
) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    n_escalate = 0
    t0 = time.time()
    for i, cls in enumerate(equiv_classes):
        ext_size, base_rate = estimate_extension_size(
            cls["predicate"], n_samples=base_probes, seed=seed,
        )
        if base_rate < base_rate_threshold:
            ext_size2, base_rate2 = estimate_extension_size(
                cls["predicate"], n_samples=escalated_probes, seed=seed + 1,
            )
            out.append((ext_size2, base_rate2))
            n_escalate += 1
        else:
            out.append((ext_size, base_rate))
        if verbose and (i + 1) % 500 == 0:
            print(
                f"  [ext] {i + 1}/{len(equiv_classes)} "
                f"elapsed={time.time() - t0:.0f}s escalated={n_escalate}",
                flush=True,
            )
    if verbose:
        print(
            f"  [ext] Done. {n_escalate}/{len(equiv_classes)} escalated. "
            f"Total time={time.time() - t0:.0f}s",
            flush=True,
        )
    return out


def compute_probe_hash(probes) -> str:
    canon = [tuple(sorted((c.suit.name, c.rank.name) for c in h)) for h in probes]
    return hashlib.sha256(repr(canon).encode()).hexdigest()


def slim_pool_entry(cls: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "fingerprint", "canonical_program", "canonical_prior",
        "summed_prior", "n_expressions", "all_programs",
    )
    return {k: cls[k] for k in keep if k in cls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="If pool.pkl already exists, load and use it instead "
                         "of re-enumerating.")
    args = ap.parse_args()

    comp_cfg = CONFIG["comparison"]
    pool_pkl = OUT_DIR / "pool.pkl"
    extensions_pkl = OUT_DIR / "extensions.pkl"

    # -------- Step 1: enumeration + fingerprinting --------
    if args.skip_if_exists and pool_pkl.exists():
        print(f"[ENUM-d8] Loading existing pool from {pool_pkl}", flush=True)
        with open(pool_pkl, "rb") as f:
            payload = pickle.load(f)
        equiv = payload["equiv"]
        stats = payload["stats"]
    else:
        print(
            f"[ENUM-d8] Building pool: depth={ANCHOR_CFG['depth']} "
            f"max_programs={ANCHOR_CFG['max_programs']:,} "
            f"grammar={ANCHOR_CFG['grammar']} (NO injections)...",
            flush=True,
        )
        t0 = time.time()
        equiv, stats = build_hypothesis_pool(
            max_depth=ANCHOR_CFG["depth"],
            max_programs=ANCHOR_CFG["max_programs"],
            max_cost=ANCHOR_CFG["max_cost"],
            timeout=ANCHOR_CFG["timeout_seconds"],
            n_probes=ANCHOR_CFG["n_probes"],
            probe_seed=ANCHOR_CFG["probe_seed"],
            verbose=1,
            enumeration_grammar=ANCHOR_CFG["grammar"],
        )
        print(
            f"[ENUM-d8] Pool built: {len(equiv):,} classes in "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
        probe_hash = compute_probe_hash(stats["_probes"])
        with open(pool_pkl, "wb") as f:
            pickle.dump({"equiv": equiv, "stats": stats,
                         "probe_hash": probe_hash}, f)
        slim_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
        with open(OUT_DIR / "pool.json", "w") as f:
            json.dump(
                {
                    "n_classes": len(equiv),
                    "stats": slim_stats,
                    "probe_hash": probe_hash,
                    "classes": [slim_pool_entry(c) for c in equiv],
                },
                f, indent=2, default=str,
            )

    # -------- Step 2: adaptive-ladder extensions --------
    if args.skip_if_exists and extensions_pkl.exists():
        print(f"[ENUM-d8] Loading existing extensions from {extensions_pkl}",
              flush=True)
        with open(extensions_pkl, "rb") as f:
            extensions = pickle.load(f)
    else:
        print(
            f"[ENUM-d8] Estimating extensions: "
            f"{comp_cfg['shared_ext_base_probes']:,} base, "
            f"{comp_cfg['shared_ext_escalated_probes']:,} for "
            f"base_rate<{comp_cfg['escalation_threshold_base_rate']}...",
            flush=True,
        )
        extensions = estimate_extensions_adaptive(
            equiv,
            base_probes=comp_cfg["shared_ext_base_probes"],
            escalated_probes=comp_cfg["shared_ext_escalated_probes"],
            base_rate_threshold=comp_cfg["escalation_threshold_base_rate"],
        )
        with open(extensions_pkl, "wb") as f:
            pickle.dump(extensions, f)
        with open(OUT_DIR / "extensions.json", "w") as f:
            json.dump(
                [{"ext_size": e[0], "base_rate": e[1]} for e in extensions],
                f,
            )

    # -------- Step 3: per-rule posteriors (single rule) --------
    print(f"[ENUM-d8] Computing posteriors for {len(RULE_IDS)} rule(s)...",
          flush=True)
    frozen = load_exemplars()
    per_rule_summary: Dict[str, Any] = {}

    for rid in RULE_IDS:
        if rid not in GALLERY_RULES or rid not in frozen:
            print(f"  [skip] {rid}: missing rule or exemplars", flush=True)
            continue
        exemplars = frozen[rid]["hands_primary"]
        t0 = time.time()
        posteriors, retained = compute_posteriors_for_rule(
            equiv_classes=equiv,
            extensions=extensions,
            exemplar_hands=exemplars,
            mass_threshold=0.001,
            return_retained_mass=True,
        )
        elapsed = time.time() - t0
        top_progs = [
            {
                "rank": i + 1,
                "prob": float(prob),
                "cls_idx": int(cls_idx),
                "fingerprint": equiv[cls_idx]["fingerprint"],
                "canonical_program": equiv[cls_idx]["canonical_program"],
                "n_expressions": equiv[cls_idx]["n_expressions"],
            }
            for i, (prob, cls_idx, _) in enumerate(posteriors[:20])
        ]
        rule_out = {
            "rule_id": rid,
            "n_hyps_in_posterior": len(posteriors),
            "enum_retained_mass": float(retained),
            "elapsed_seconds": round(elapsed, 2),
            "top20": top_progs,
            "full_posterior": [
                {
                    "prob": float(prob),
                    "cls_idx": int(cls_idx),
                    "fingerprint": equiv[cls_idx]["fingerprint"],
                }
                for prob, cls_idx, _ in posteriors
            ],
        }
        with open(OUT_DIR / "posteriors" / f"{rid}.json", "w") as f:
            json.dump(rule_out, f, indent=2)
        per_rule_summary[rid] = {
            "n_hyps": rule_out["n_hyps_in_posterior"],
            "retained_mass": rule_out["enum_retained_mass"],
            "elapsed_s": rule_out["elapsed_seconds"],
            "top_fp": top_progs[0]["fingerprint"] if top_progs else None,
        }
        print(
            f"  [ok] {rid}: {len(posteriors):,} hyps, "
            f"retained={retained:.4f}, {elapsed:.1f}s",
            flush=True,
        )

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(
            {
                "depth": ANCHOR_CFG["depth"],
                "max_programs": ANCHOR_CFG["max_programs"],
                "n_classes": len(equiv),
                "n_rules": len(per_rule_summary),
                "rules": per_rule_summary,
            },
            f, indent=2,
        )
    print("[ENUM-d8] Done.", flush=True)


if __name__ == "__main__":
    main()
