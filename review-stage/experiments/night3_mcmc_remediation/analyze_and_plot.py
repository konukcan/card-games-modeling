"""
Post-hoc analysis + plots for Night 3 MCMC remediation.

Generates:
  1. convergence_mass_mapped.png  — mass_mapped trajectory per rule
  2. convergence_tv.png           — total-variation trajectory per rule
  3. final_scatter.png            — TV vs mass_mapped, all rules
  4. mass_allocation.png          — stacked bar: mapped vs unmapped vs parse_fail per rule
  5. renormalized_tv.png          — raw TV vs renormalized TV (redistribute leaked mass)
  6. topk_best_vs_worst.png       — enum vs MCMC posterior bars for 3 best + 3 worst rules
  7. ground_truth_rank.png        — rank of enum's top-1 class in MCMC's posterior

Also prints a summary table with the renormalization diagnostic.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -------------------- paths --------------------
ROOT = Path(__file__).parent
COMP = ROOT / "comparison"
CONV = COMP / "convergence_diagnostics"
QA = COMP / "question_a"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# -------------------- load data --------------------

def load_all() -> Tuple[List[str], Dict[str, dict], Dict[str, dict], dict]:
    """Load per-rule convergence, question-A details, and the overall summary.

    Returns:
        rules: sorted rule IDs
        conv:  rule_id -> convergence diagnostic dict
        qa:    rule_id -> question-A detail dict
        summary: top-level comparison/summary.json
    """
    summary = json.loads((COMP / "summary.json").read_text())
    rules = sorted(summary["rules"].keys())
    conv = {r: json.loads((CONV / f"{r}.json").read_text()) for r in rules}
    qa = {r: json.loads((QA / f"{r}.json").read_text()) for r in rules}
    return rules, conv, qa, summary


# -------------------- diagnostics --------------------

def renormalized_tv(qa_rule: dict) -> float:
    """Recompute TV after throwing away MCMC's unmapped mass and renormalising
    over only the classes enum also knows about.

    If this is close to the raw TV, MCMC genuinely misallocates among known
    classes. If it's near zero, the divergence was mostly mass leakage.

    Implementation: we use the class-level top-20 lists from question-A.
    Both enum and MCMC top-20s are sub-sets of the common class index; we
    take the union, pull (prob_enum, prob_mcmc) for each, then renormalise
    MCMC to drop unmapped mass proportionally across the mapped classes
    before computing TV.
    """
    enum_top = qa_rule.get("enum_top20", [])
    mcmc_top = qa_rule.get("mcmc_top20", [])
    mapped_classes = {e["cls_idx"] for e in enum_top} | {m["cls_idx"] for m in mcmc_top}

    enum_probs = {e["cls_idx"]: e["prob"] for e in enum_top}
    mcmc_probs = {m["cls_idx"]: m["prob"] for m in mcmc_top}

    # Scale MCMC by 1 / mass_mapped — i.e., redistribute leaked mass
    # proportionally across the classes MCMC *did* hit that enum knows.
    mass_mapped = qa_rule["parse_audit"]["mass_mapped_frac"]
    if mass_mapped <= 0:
        return float("nan")
    renorm = {c: p / mass_mapped for c, p in mcmc_probs.items()
              if c in mapped_classes and c in {e["cls_idx"] for e in enum_top}}
    # Restrict enum to the intersection for a fair comparison
    intersect = set(renorm.keys()) & set(enum_probs.keys())
    if not intersect:
        return float("nan")
    # Renormalise enum over the intersection too (since top-20 truncates)
    e_sum = sum(enum_probs[c] for c in intersect)
    m_sum = sum(renorm[c] for c in intersect)
    if e_sum == 0 or m_sum == 0:
        return float("nan")
    tv = 0.5 * sum(abs(enum_probs[c] / e_sum - renorm[c] / m_sum) for c in intersect)
    return tv


def rank_of_enum_top1_in_mcmc(qa_rule: dict) -> int | None:
    """Return 1-based rank of enum's top-1 class in MCMC's top-20, or None if
    MCMC didn't rank it in the top 20."""
    enum_top1 = qa_rule["top_k"].get("enum_top1_cls")
    if enum_top1 is None:
        return None
    for m in qa_rule.get("mcmc_top20", []):
        if m["cls_idx"] == enum_top1:
            return m["rank"]
    return None


def mcmc_prob_on_enum_top1(qa_rule: dict) -> float:
    """Return the probability MCMC assigns to enum's top-1 class (0 if not in top-20)."""
    enum_top1 = qa_rule["top_k"].get("enum_top1_cls")
    if enum_top1 is None:
        return float("nan")
    for m in qa_rule.get("mcmc_top20", []):
        if m["cls_idx"] == enum_top1:
            return m["prob"]
    return 0.0


# -------------------- plots --------------------

def plot_convergence(rules: List[str], conv: Dict[str, dict], metric: str,
                     ylabel: str, threshold: float | None, out_path: Path,
                     title: str) -> None:
    """Small-multiple line plot: one panel per rule, metric over checkpoint steps.

    `metric` selects the field inside each trajectory entry ("mass_mapped" or
    "total_variation"). `threshold` draws a horizontal reference line (e.g. 0.9
    for the validity gate)."""
    n = len(rules)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.2 * rows), sharex=True)
    axes = axes.flatten()
    for i, r in enumerate(rules):
        ax = axes[i]
        traj = conv[r]["trajectory"]
        steps = [t["step"] for t in traj]
        vals = [t[metric] for t in traj]
        passed = traj[-1]["comparison_valid"]
        colour = "#2a9d8f" if passed else "#e76f51"
        ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.3, color=colour)
        if threshold is not None:
            ax.axhline(threshold, linestyle="--", linewidth=0.8, color="#888")
        ax.set_title(r, fontsize=8)
        ax.tick_params(labelsize=7)
        if metric == "mass_mapped":
            ax.set_ylim(-0.02, 1.02)
        if i % cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if i >= n - cols:
            ax.set_xlabel("MCMC step", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_final_scatter(summary: dict, qa: Dict[str, dict], out_path: Path) -> None:
    """Scatter: TV on y-axis, mass_mapped on x-axis. Each point is a rule.
    Colour encodes whether enum and MCMC agree on the modal class."""
    rules = sorted(summary["rules"].keys())
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rules:
        s = summary["rules"][r]
        top_match = qa[r]["top_k"].get("enum_top1_cls") == qa[r]["top_k"].get("mcmc_top1_cls")
        colour = "#2a9d8f" if top_match else "#e76f51"
        ax.scatter(s["mass_mapped"], s["total_variation"], color=colour, s=60,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(r, (s["mass_mapped"], s["total_variation"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axvline(0.90, linestyle="--", color="#888", linewidth=0.8,
               label="validity gate (mass_mapped ≥ 0.9)")
    ax.axhline(0.20, linestyle=":", color="#888", linewidth=0.8,
               label="agreement bar (TV < 0.2)")
    ax.set_xlabel("mass_mapped  (fraction of MCMC posterior on enum-known classes)")
    ax.set_ylabel("Total variation between enum and MCMC posteriors")
    ax.set_title("Final metrics per rule (colour = top-1 class agrees)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0.6, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_mass_allocation(rules: List[str], summary: dict, out_path: Path) -> None:
    """Horizontal stacked bar per rule: mapped / unmapped / parse_fail."""
    order = sorted(rules, key=lambda r: summary["rules"][r]["mass_mapped"])
    mapped = [summary["rules"][r]["mass_mapped"] for r in order]
    unmapped = [summary["rules"][r]["mass_unmapped"] for r in order]
    parse_fail = [summary["rules"][r]["mass_parse_fail"] for r in order]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))
    ax.barh(y, mapped, color="#2a9d8f", label="mass_mapped (enum knows this class)")
    ax.barh(y, unmapped, left=mapped, color="#e76f51", label="mass_unmapped (enum never enumerated)")
    ax.barh(y, parse_fail, left=[a + b for a, b in zip(mapped, unmapped)],
            color="#888", label="mass_parse_fail")
    ax.axvline(0.90, linestyle="--", color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Fraction of MCMC posterior mass")
    ax.set_xlim(0, 1.01)
    ax.set_title("Where MCMC's posterior mass lands (sorted by mass_mapped)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_renormalized_tv(rules: List[str], summary: dict, qa: Dict[str, dict],
                         out_path: Path) -> Dict[str, float]:
    """Compare raw TV to TV after redistributing MCMC's leaked mass.

    This is the key failure-mode diagnostic. Close columns = mass leakage
    was the problem (and renormalising fixes it). Far-apart columns =
    MCMC genuinely misallocates among enum-known classes."""
    renorm = {r: renormalized_tv(qa[r]) for r in rules}
    order = sorted(rules, key=lambda r: summary["rules"][r]["total_variation"])
    raw = [summary["rules"][r]["total_variation"] for r in order]
    rn = [renorm[r] if not math.isnan(renorm[r]) else 0 for r in order]
    x = np.arange(len(order))
    width = 0.42
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, raw, width, color="#e76f51", label="raw TV")
    ax.bar(x + width / 2, rn, width, color="#2a9d8f",
           label="TV after redistributing leaked mass")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total variation")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.20, linestyle=":", color="#888", linewidth=0.8)
    ax.set_title("Does redistributing MCMC's leaked mass rescue the comparison?")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return renorm


def plot_topk_bars(rules: List[str], qa: Dict[str, dict], summary: dict,
                   out_path: Path, k: int = 10) -> None:
    """Side-by-side posterior bars for the top-k classes of a few illustrative rules.

    Picks: 2 best (tight agreement), 2 middle (passes gate, loose), 2 worst (fail)."""
    ranked = sorted(rules, key=lambda r: summary["rules"][r]["total_variation"])
    picks = [ranked[0], ranked[1], ranked[len(ranked) // 2 - 1],
             ranked[len(ranked) // 2 + 1], ranked[-2], ranked[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, r in enumerate(picks):
        ax = axes[i]
        enum_top = qa[r].get("enum_top20", [])[:k]
        mcmc_top = qa[r].get("mcmc_top20", [])[:k]
        # Build union of top-k class indices, with enum first (sorted by enum rank
        # then mcmc rank) for interpretability.
        enum_by_cls = {e["cls_idx"]: e["prob"] for e in enum_top}
        mcmc_by_cls = {m["cls_idx"]: m["prob"] for m in mcmc_top}
        union = list(enum_by_cls.keys())
        for c in mcmc_by_cls:
            if c not in enum_by_cls:
                union.append(c)
        union = union[:12]  # cap at 12 bars to stay readable
        x = np.arange(len(union))
        width = 0.4
        e_probs = [enum_by_cls.get(c, 0) for c in union]
        m_probs = [mcmc_by_cls.get(c, 0) for c in union]
        ax.bar(x - width / 2, e_probs, width, color="#264653", label="enum")
        ax.bar(x + width / 2, m_probs, width, color="#e9c46a", label="mcmc")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in union], rotation=0, fontsize=7)
        ax.set_title(f"{r}  (TV={summary['rules'][r]['total_variation']:.2f})",
                     fontsize=9)
        ax.set_ylabel("posterior prob", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Top-k posterior comparison  (x-axis = enum class index)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_ground_truth_rank(rules: List[str], qa: Dict[str, dict], summary: dict,
                           out_path: Path) -> Dict[str, dict]:
    """For each rule, how concentrated is each algorithm on what we treat as the
    ground-truth class (= enum's top-1)?

    Two bars per rule:
      - enum prob on its top-1 (self-confidence; should be near 1)
      - MCMC prob on enum's top-1 (cross-concentration)
    """
    stats = {}
    for r in rules:
        enum_p = qa[r]["enum_top20"][0]["prob"] if qa[r]["enum_top20"] else 0.0
        mcmc_p = mcmc_prob_on_enum_top1(qa[r])
        rank = rank_of_enum_top1_in_mcmc(qa[r])
        stats[r] = {"enum_p": enum_p, "mcmc_p": mcmc_p, "rank_in_mcmc": rank}
    order = sorted(rules, key=lambda r: stats[r]["mcmc_p"], reverse=True)
    x = np.arange(len(order))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, [stats[r]["enum_p"] for r in order], width,
           color="#264653", label="enum on its own top-1")
    ax.bar(x + width / 2, [stats[r]["mcmc_p"] for r in order], width,
           color="#e9c46a", label="MCMC on enum's top-1")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Posterior probability on ground-truth class")
    ax.set_ylim(0, 1.05)
    ax.set_title("How much mass does each algorithm put on the 'true' class?")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return stats


# -------------------- summary printout --------------------

def print_summary(rules: List[str], summary: dict, qa: Dict[str, dict],
                  renorm: Dict[str, float], gt: Dict[str, dict]) -> None:
    print("=" * 100)
    print(f"{'rule':<26} {'raw_TV':>8} {'renorm_TV':>10} {'mass_mapped':>12} "
          f"{'enum_top1_p':>12} {'mcmc_on_gt':>11} {'rank_gt_mcmc':>13}")
    print("-" * 100)
    for r in sorted(rules, key=lambda r: summary["rules"][r]["total_variation"]):
        s = summary["rules"][r]
        rn = renorm[r]
        rn_str = f"{rn:>10.3f}" if not math.isnan(rn) else f"{'nan':>10}"
        g = gt[r]
        rank = g["rank_in_mcmc"]
        rank_str = f"{rank:>13}" if rank is not None else f"{'>20':>13}"
        print(f"{r:<26} {s['total_variation']:>8.3f} {rn_str} "
              f"{s['mass_mapped']:>12.3f} {g['enum_p']:>12.3f} "
              f"{g['mcmc_p']:>11.3f} {rank_str}")
    print("=" * 100)


# -------------------- driver --------------------

def main() -> None:
    rules, conv, qa, summary = load_all()
    print(f"Loaded {len(rules)} rules")

    plot_convergence(rules, conv, metric="mass_mapped",
                     ylabel="mass_mapped", threshold=0.90,
                     out_path=OUT / "convergence_mass_mapped.png",
                     title="MCMC mass landing on enum-known classes, over checkpoints")
    plot_convergence(rules, conv, metric="total_variation",
                     ylabel="TV(enum, mcmc)", threshold=0.20,
                     out_path=OUT / "convergence_tv.png",
                     title="Total variation between enum and MCMC posteriors, over checkpoints")
    plot_final_scatter(summary, qa, OUT / "final_scatter.png")
    plot_mass_allocation(rules, summary, OUT / "mass_allocation.png")
    renorm = plot_renormalized_tv(rules, summary, qa, OUT / "renormalized_tv.png")
    plot_topk_bars(rules, qa, summary, OUT / "topk_best_vs_worst.png")
    gt = plot_ground_truth_rank(rules, qa, summary, OUT / "ground_truth_rank.png")

    print_summary(rules, summary, qa, renorm, gt)

    print("\nPlots written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name, f"({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
