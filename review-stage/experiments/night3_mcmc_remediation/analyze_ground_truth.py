"""Ground-truth correctness analysis for night-3 enum vs MCMC comparison.

Computes, for every rule:
  - The true-rule extension (from GALLERY_RULES predicate) on the 500 probes.
  - The "ground-truth class" = the class whose fingerprint on those 500 probes
    matches the true-rule fingerprint.
  - Per-class MCC between class extension and true extension (500-bit vectors).
  - Expected MCC under enum's full posterior and under MCMC's aggregated posterior
    (renormalised over mapped mass).
  - Whether each method's modal class is the ground-truth class, and how much
    posterior mass sits on that class.

Inputs (all already on disk from the overnight run):
  enum_depth6_300k/pool.pkl           — 2501 classes + 500 probes
  enum_depth6_300k/posteriors/*.json  — per-rule full enum posterior
  mcmc_50k_4chains/raw_visits/*.json  — merged visit counts across 4 chains
  comparison/summary.json             — existing TV / JS / mass_mapped per rule

Run from inside the night3_mcmc_remediation directory:
    python3 analyze_ground_truth.py
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Locate pieces of the project.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

POOL_PATH = HERE / "enum_depth6_300k" / "pool.pkl"
ENUM_POST_DIR = HERE / "enum_depth6_300k" / "posteriors"
MCMC_VISITS_DIR = HERE / "mcmc_50k_4chains" / "raw_visits"
SUMMARY_PATH = HERE / "comparison" / "summary.json"
CONFIG_PATH = HERE / "config.json"
OUT_DIR = HERE / "ground_truth"
PLOT_DIR = HERE / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

from gallery_analysis.gallery_rules import GALLERY_RULES  # noqa: E402
from gallery_analysis.enumerator import build_gallery_grammar, _make_evaluator  # noqa: E402
from gallery_analysis.exemplars import load_exemplars, generate_probe_set  # noqa: E402
from gallery_analysis.hypothesis_table import compute_fingerprint  # noqa: E402
from dreamcoder_core.program import parse_program, Primitive  # noqa: E402


# ---------------------------------------------------------------------------
# MCC helper.
# ---------------------------------------------------------------------------

def mcc(a: np.ndarray, b: np.ndarray) -> float:
    """Matthews correlation coefficient between two boolean vectors."""
    a = a.astype(bool)
    b = b.astype(bool)
    tp = int(np.sum(a & b))
    tn = int(np.sum(~a & ~b))
    fp = int(np.sum(~a & b))
    fn = int(np.sum(a & ~b))
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        if tp + tn == len(a):
            return 1.0
        if fp + fn == len(a):
            return -1.0
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denom_sq)


def _bits_for_predicate(pred, probes) -> np.ndarray:
    out = np.zeros(len(probes), dtype=bool)
    for i, h in enumerate(probes):
        try:
            out[i] = bool(pred(h))
        except Exception:
            out[i] = False
    return out


# ---------------------------------------------------------------------------
# MCMC aggregation — re-implements run_comparison._aggregate_mcmc_to_classes
# but stripped of extras we don't need.
# ---------------------------------------------------------------------------

def _compose_fp(parent_fp: str, sub_fp: str) -> str:
    return hashlib.sha256(f"{parent_fp}|{sub_fp}".encode()).hexdigest()


def _build_member_fp_fn(exemplar_hands, holdout_seed: int = 9999,
                        n_holdout: int = 1000):
    holdout = generate_probe_set(n_probes=n_holdout, seed=holdout_seed)
    check_hands = list(exemplar_hands) + list(holdout)

    def member_fp(pred):
        bits = []
        for h in check_hands:
            try:
                bits.append("1" if pred(h) else "0")
            except Exception:
                bits.append("E")
        return "".join(bits)
    return member_fp


def _build_indices(pool):
    fp_to_cls: Dict[str, int] = {}
    parent_to_subs: Dict[str, List[Tuple[str, int]]] = {}
    for i, cls in enumerate(pool["equiv"]):
        fp = cls["fingerprint"]
        parent = cls.get("parent_fingerprint")
        if parent is None:
            fp_to_cls[fp] = i
        else:
            parent_to_subs.setdefault(parent, []).append((fp, i))
    return fp_to_cls, parent_to_subs


def aggregate_mcmc_to_classes(
    visit_counts: Dict[str, int],
    prim_dict: Dict[str, Primitive],
    probes,
    fp_to_cls,
    parent_to_subs,
    member_fp_fn,
) -> Tuple[Dict[int, int], int, int]:
    """Returns (class_counts, mapped_total, total_visits)."""
    class_counts: Dict[int, int] = {}
    total_visits = 0
    mapped_total = 0
    for prog_str, count in visit_counts.items():
        total_visits += count
        try:
            program = parse_program(prog_str, prim_dict)
        except Exception:
            continue
        try:
            pred = _make_evaluator(program)
            fp = compute_fingerprint(pred, probes)
        except Exception:
            continue
        if fp in fp_to_cls:
            idx = fp_to_cls[fp]
            class_counts[idx] = class_counts.get(idx, 0) + count
            mapped_total += count
        elif fp in parent_to_subs:
            try:
                sub_fp = member_fp_fn(pred)
            except Exception:
                continue
            composite = _compose_fp(fp, sub_fp)
            hit_idx = None
            for stored_comp, cidx in parent_to_subs[fp]:
                if stored_comp == composite:
                    hit_idx = cidx
                    break
            if hit_idx is not None:
                class_counts[hit_idx] = class_counts.get(hit_idx, 0) + count
                mapped_total += count
    return class_counts, mapped_total, total_visits


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    print("Loading pool.pkl...")
    pool = pickle.load(open(POOL_PATH, "rb"))
    probes_500 = pool["stats"]["_probes"]
    classes = pool["equiv"]
    print(f"  {len(classes)} classes, {len(probes_500)} probes, hash {pool['probe_hash'][:12]}")

    cfg = json.load(open(CONFIG_PATH))
    rule_ids = cfg["rules_night2"] + cfg["rules_new"]

    summary = json.load(open(SUMMARY_PATH))
    per_rule_summary = summary["rules"]  # dict keyed by rule_id

    fp_to_cls, parent_to_subs = _build_indices(pool)

    grammar = build_gallery_grammar()
    prim_dict: Dict[str, Primitive] = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            prim_dict[prod.program.name] = prod.program

    # Load exemplars — we'll append them to the 500 probes for MCC computation
    # so that rare rules (true rate ≈ 0 on random probes) aren't degenerate.
    exemplars = load_exemplars()

    # Pool-wide member_fp: uses the flat list of ALL exemplar hands across
    # the 60 gallery rules (360 hands) + 1000 holdout. This matches what
    # analyze._strict_split_classes saw when it split, so composite
    # fingerprints will agree with those stored on pool sub-classes.
    pool_exemplar_hands = pool["stats"].get("_exemplar_hands")
    if pool_exemplar_hands is None:
        raise RuntimeError("pool.pkl has no _exemplar_hands in stats")
    print(f"Pool-wide member_fp: {len(pool_exemplar_hands)} exemplars + 1000 holdout")
    member_fp_fn = _build_member_fp_fn(pool_exemplar_hands)

    results = []
    for rule_id in rule_ids:
        if rule_id not in per_rule_summary:
            print(f"[skip] {rule_id}: not in comparison summary")
            continue
        print(f"\n=== {rule_id} ===")

        # Pre-registered check: rule must have complete exemplars.
        if rule_id not in exemplars:
            print(f"  [skip] no exemplars registered")
            continue
        exemplar_hands = exemplars[rule_id]["hands_primary"]

        # MCC probe set = 500 fingerprint probes + 6 positive exemplars.
        # This breaks the all-False degeneracy for rare rules: a class that
        # returns False on the exemplars can no longer trivially match a
        # rare true rule just by also being all-False on 500 random hands.
        mcc_probes = list(probes_500) + list(exemplar_hands)

        # Re-evaluate each class on the extended probe set for MCC only
        # (the fingerprint index is still keyed on the original 500 probes).
        print(f"  evaluating {len(classes)} classes on {len(mcc_probes)} "
              f"(500 probes + {len(exemplar_hands)} exemplars)...")
        cls_bits = np.zeros((len(classes), len(mcc_probes)), dtype=bool)
        for i, cls in enumerate(classes):
            cls_bits[i] = _bits_for_predicate(cls["predicate"], mcc_probes)

        # True-rule bit pattern on the same extended set.
        true_pred = GALLERY_RULES[rule_id]["predicate"]
        true_bits = _bits_for_predicate(true_pred, mcc_probes)
        p_true_500 = float(true_bits[:500].mean())
        print(f"  true-rule base rate on 500 random probes: {p_true_500:.3f} "
              f"(on 6 exemplars: {int(true_bits[500:].sum())}/6)")

        # Ground-truth class — classes that match the extended bit pattern
        # exactly are the true-rule class.
        match_ids = np.where(np.all(cls_bits == true_bits[None, :], axis=1))[0]
        per_class_mcc = np.array([mcc(cls_bits[i], true_bits) for i in range(len(classes))])
        if len(match_ids) == 1:
            gt_cls = int(match_ids[0])
            gt_mcc = 1.0
            gt_note = "unique exact match"
        elif len(match_ids) > 1:
            gt_cls = int(match_ids[0])
            gt_mcc = 1.0
            gt_note = f"{len(match_ids)} exact matches; using first"
        else:
            gt_cls = int(np.argmax(per_class_mcc))
            gt_mcc = float(per_class_mcc[gt_cls])
            gt_note = f"no exact match; argmax MCC cls {gt_cls} (MCC {gt_mcc:.3f})"
        print(f"  ground-truth class: {gt_cls} ({gt_note})")
        print(f"    canonical program: {classes[gt_cls]['canonical_program']}")

        # Enum full posterior → expected MCC.
        enum_post = json.load(open(ENUM_POST_DIR / f"{rule_id}.json"))
        enum_probs = np.zeros(len(classes))
        for entry in enum_post["full_posterior"]:
            enum_probs[entry["cls_idx"]] = entry["prob"]
        if enum_probs.sum() > 0:
            enum_probs /= enum_probs.sum()
        enum_emcc = float(np.dot(enum_probs, per_class_mcc))
        enum_mass_on_gt = float(enum_probs[gt_cls])
        enum_top1 = int(np.argmax(enum_probs))
        enum_top1_is_gt = (enum_top1 == gt_cls)

        # MCMC posterior — aggregate via same logic as comparison script.
        mcmc_blob = json.load(open(MCMC_VISITS_DIR / f"{rule_id}.json"))
        visit_counts = mcmc_blob["visit_counts"]
        class_counts, mapped_total, total_visits = aggregate_mcmc_to_classes(
            visit_counts, prim_dict, probes_500, fp_to_cls, parent_to_subs, member_fp_fn,
        )
        if mapped_total > 0:
            mcmc_probs = np.zeros(len(classes))
            for idx, c in class_counts.items():
                mcmc_probs[idx] = c / mapped_total
            mcmc_emcc = float(np.dot(mcmc_probs, per_class_mcc))
            mcmc_mass_on_gt = float(mcmc_probs[gt_cls])
            mcmc_top1 = int(np.argmax(mcmc_probs))
            mcmc_top1_is_gt = (mcmc_top1 == gt_cls)
        else:
            mcmc_emcc = float("nan")
            mcmc_mass_on_gt = float("nan")
            mcmc_top1 = None
            mcmc_top1_is_gt = False

        mcmc_raw_mass_on_gt = class_counts.get(gt_cls, 0) / max(total_visits, 1)

        result = {
            "rule_id": rule_id,
            "true_base_rate_on_500_probes": p_true_500,
            "n_exemplars_true": int(true_bits[500:].sum()),
            "gt_cls": gt_cls,
            "gt_note": gt_note,
            "gt_canonical_program": classes[gt_cls]["canonical_program"],

            "enum_top1": enum_top1,
            "enum_top1_is_gt": enum_top1_is_gt,
            "enum_mass_on_gt": enum_mass_on_gt,
            "enum_expected_mcc": enum_emcc,

            "mcmc_top1": mcmc_top1,
            "mcmc_top1_is_gt": mcmc_top1_is_gt,
            "mcmc_mass_on_gt": mcmc_mass_on_gt,            # renormalised over mapped mass
            "mcmc_raw_mass_on_gt": mcmc_raw_mass_on_gt,    # over all MCMC mass
            "mcmc_expected_mcc": mcmc_emcc,

            "tv": per_rule_summary[rule_id]["total_variation"],
            "js": per_rule_summary[rule_id]["jensen_shannon"],
            "mass_mapped": per_rule_summary[rule_id]["mass_mapped"],
            "comparison_valid": per_rule_summary[rule_id]["comparison_valid"],
        }
        results.append(result)
        print(f"  enum  E[MCC]={enum_emcc:+.3f}   P(gt)={enum_mass_on_gt:.3f}   top1=gt? {enum_top1_is_gt}")
        print(f"  MCMC  E[MCC]={mcmc_emcc:+.3f}   P(gt|mapped)={mcmc_mass_on_gt:.3f}   "
              f"raw={mcmc_raw_mass_on_gt:.3f}   top1=gt? {mcmc_top1_is_gt}")

    out_path = OUT_DIR / "ground_truth_summary.json"
    out_path.write_text(json.dumps({"rules": results}, indent=2))
    print(f"\nWrote {out_path}")

    make_plots(results)


def make_plots(results):
    if not results:
        return
    rule_ids = [r["rule_id"] for r in results]
    enum_emcc = np.array([r["enum_expected_mcc"] for r in results])
    mcmc_emcc = np.array([r["mcmc_expected_mcc"] for r in results])
    tvs = np.array([r["tv"] for r in results])
    mass_mapped = np.array([r["mass_mapped"] for r in results])

    # (a) Side-by-side E[MCC] per rule.
    order = np.argsort(enum_emcc)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))
    ax.barh(y - 0.2, enum_emcc[order], height=0.38, label="enum E[MCC]", color="#2a6bcc")
    ax.barh(y + 0.2, mcmc_emcc[order], height=0.38, label="MCMC E[MCC]", color="#cc5a2a")
    ax.set_yticks(y)
    ax.set_yticklabels([rule_ids[i] for i in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(1, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Expected MCC vs. true-rule extension on 500 probes")
    ax.set_title("Ground-truth correctness of posterior mass, per rule")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "expected_mcc_per_rule.png", dpi=150)
    plt.close(fig)

    # (b) Disagreement vs correctness gap.
    diff = enum_emcc - mcmc_emcc
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(tvs, diff, c=mass_mapped, cmap="viridis",
                    s=70, edgecolor="k", lw=0.5)
    for r, x, dy in zip(results, tvs, diff):
        ax.annotate(r["rule_id"], (x, dy), fontsize=6.8, xytext=(3, 3),
                    textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="mass_mapped (MCMC)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("TV(enum, MCMC)")
    ax.set_ylabel("E[MCC]$_{enum}$ − E[MCC]$_{MCMC}$")
    ax.set_title("When enum and MCMC disagree, is one correct?\n(positive y = enum puts mass closer to ground truth)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "disagreement_vs_correctness.png", dpi=150)
    plt.close(fig)

    # (c) TV vs best correctness.
    best_emcc = np.maximum(enum_emcc, mcmc_emcc)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(tvs, best_emcc,
               c=[1 if r["comparison_valid"] else 0 for r in results],
               cmap="RdYlGn", s=70, edgecolor="k", lw=0.5)
    for r, x, y in zip(results, tvs, best_emcc):
        ax.annotate(r["rule_id"], (x, y), fontsize=6.8, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("TV(enum, MCMC)")
    ax.set_ylabel("max(E[MCC]$_{enum}$, E[MCC]$_{MCMC}$)")
    ax.set_title("Do the rules where they agree also happen to be the rules where they're correct?")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "agreement_vs_best_correctness.png", dpi=150)
    plt.close(fig)

    # (d) Mass on ground-truth class.
    gt_mass_enum = np.array([r["enum_mass_on_gt"] for r in results])
    gt_mass_mcmc = np.array([r["mcmc_mass_on_gt"] for r in results])
    order = np.argsort(gt_mass_enum)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))
    ax.barh(y - 0.2, gt_mass_enum[order], height=0.38, label="P(gt class | enum)", color="#2a6bcc")
    ax.barh(y + 0.2, gt_mass_mcmc[order], height=0.38, label="P(gt class | MCMC, renorm)", color="#cc5a2a")
    ax.set_yticks(y)
    ax.set_yticklabels([rule_ids[i] for i in order], fontsize=8)
    ax.set_xlabel("Posterior mass on ground-truth class")
    ax.set_title("How much mass each method places on the true rule's equivalence class")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mass_on_ground_truth.png", dpi=150)
    plt.close(fig)

    print(f"Plots written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
