"""
Night 3 — Enumeration runner (depth-6, 300k programs, NO injections, uniform grammar).

Builds the equivalence-class pool that MCMC visits will be aggregated against.
Computes per-class extensions via adaptive ladder (100k base, 1M if base_rate <
0.001). Computes per-rule posteriors for the 20 target rules.

Serializes a pickled "pool" file (predicates can't easily be JSON'd) plus
JSON sidecars for human inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import cloudpickle as pickle  # predicates are local closures; stdlib pickle fails
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
RULE_IDS = CONFIG["rules_night2"] + CONFIG["rules_new"]
OUT_DIR = HERE / "enum_depth6_300k"
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
    """Two-pass adaptive extension estimation.

    Pass 1: estimate with `base_probes` MC samples.
    Pass 2: for any class with base_rate < threshold, re-estimate with
            `escalated_probes` samples for higher resolution on rare rules.
    """
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
            f"  [ext] Done. {n_escalate}/{len(equiv_classes)} escalated to "
            f"{escalated_probes:,} probes. Total time={time.time() - t0:.0f}s",
            flush=True,
        )
    return out


def compute_probe_hash(probes) -> str:
    """Canonical hash of the probe set (so future runs can verify compat)."""
    canon = [tuple(sorted((c.suit.name, c.rank.name) for c in h)) for h in probes]
    return hashlib.sha256(repr(canon).encode()).hexdigest()


def slim_pool_entry(cls: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe class projection (predicates/callables dropped)."""
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

    enum_cfg = CONFIG["enumeration"]
    comp_cfg = CONFIG["comparison"]

    pool_pkl = OUT_DIR / "pool.pkl"
    extensions_pkl = OUT_DIR / "extensions.pkl"

    # -------- Step 1: enumeration + fingerprinting --------
    if args.skip_if_exists and pool_pkl.exists():
        print(f"[ENUM] Loading existing pool from {pool_pkl}", flush=True)
        with open(pool_pkl, "rb") as f:
            payload = pickle.load(f)
        equiv = payload["equiv"]
        stats = payload["stats"]
    else:
        print(
            f"[ENUM] Building pool: depth={enum_cfg['depth']} "
            f"max_programs={enum_cfg['max_programs']:,} "
            f"grammar={enum_cfg['grammar']} (NO injections)...",
            flush=True,
        )
        t0 = time.time()
        equiv, stats = build_hypothesis_pool(
            max_depth=enum_cfg["depth"],
            max_programs=enum_cfg["max_programs"],
            max_cost=enum_cfg["max_cost"],
            timeout=enum_cfg["timeout_seconds"],
            n_probes=enum_cfg["n_probes"],
            probe_seed=enum_cfg["probe_seed"],
            verbose=1,
            enumeration_grammar=enum_cfg["grammar"],
        )
        print(
            f"[ENUM] Pool built: {len(equiv):,} classes in "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
        probe_hash = compute_probe_hash(stats["_probes"])
        with open(pool_pkl, "wb") as f:
            pickle.dump({"equiv": equiv, "stats": stats,
                         "probe_hash": probe_hash}, f)
        # Also write a JSON sidecar (predicates dropped — not serializable).
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
        print(f"[ENUM] Loading existing extensions from {extensions_pkl}",
              flush=True)
        with open(extensions_pkl, "rb") as f:
            extensions = pickle.load(f)
    else:
        print(
            f"[ENUM] Estimating extensions: adaptive ladder "
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

    # -------- Step 3: per-rule posteriors --------
    print(f"[ENUM] Computing posteriors for {len(RULE_IDS)} rules...",
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
        top_idx = [cls_idx for _, cls_idx, _ in posteriors[:20]]
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
            # Full list (prob, fingerprint) for downstream comparison.
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
                "n_classes": len(equiv),
                "n_rules": len(per_rule_summary),
                "rules": per_rule_summary,
            },
            f, indent=2,
        )
    print("[ENUM] Done.", flush=True)


if __name__ == "__main__":
    main()
