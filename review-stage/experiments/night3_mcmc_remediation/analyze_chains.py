"""Chain-level analysis for night-3 MCMC runs.

Addresses the two questions:
  Q1. What did running 4 independent chains per rule actually buy us?
  Q2. Does agreement between MCMC and enum scale with the number of chains?

Per-chain diagnostics saved during the overnight run:
  - n_unique_per_chain  — # unique programs each chain visited
  - acceptance_rate_per_chain — MH acceptance rate per chain

Per-chain VISIT COUNTS were NOT persisted (only the merged dict across
chains). That means we cannot directly recompute a "1-chain MCMC posterior"
from the saved data; doing so would require rerunning with per-chain
persistence. We do what we can with the summaries we have.

Computes:
  - Dispersion of chain statistics per rule (mean, std, CV of n_unique, etc.)
  - Whether chain heterogeneity correlates with TV-vs-enum.
  - A simple "mode coverage" diagnostic: total_unique_merged /
    mean(n_unique_per_chain) — if close to 1, chains visited largely
    overlapping regions; if ≫ 1, chains explored disjoint subsets.

Plots:
  - chain_diagnostics_overview.png
  - chain_heterogeneity_vs_tv.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "mcmc_50k_4chains" / "raw_visits"
SUMMARY = json.load(open(HERE / "comparison" / "summary.json"))
GT = json.load(open(HERE / "ground_truth" / "ground_truth_summary.json"))
PLOT_DIR = HERE / "plots"
OUT_PATH = HERE / "ground_truth" / "chain_analysis.json"


def main():
    per_rule_summary = SUMMARY["rules"]
    gt_by_rule = {r["rule_id"]: r for r in GT["rules"]}

    records = []
    for f in sorted(RAW_DIR.glob("*.json")):
        rule = f.stem
        blob = json.loads(f.read_text())
        n_unique_chains = blob["n_unique_per_chain"]            # list of 4 ints
        acc_chains = blob["acceptance_rate_per_chain"]          # list of 4 floats
        n_unique_merged = blob["n_unique_merged"]
        total_steps = blob["total_steps"]
        # "Mode diversity": if all chains found the same programs, merged ≈ mean(per chain).
        # If chains are disjoint, merged ≈ sum(per chain).
        mean_per_chain = float(np.mean(n_unique_chains))
        coverage_ratio = n_unique_merged / max(1.0, mean_per_chain)  # in [1, n_chains]

        rec = {
            "rule_id": rule,
            "n_chains": blob["n_chains"],
            "total_steps": total_steps,
            "steps_per_chain": total_steps // blob["n_chains"],
            "n_unique_per_chain": n_unique_chains,
            "mean_n_unique_per_chain": mean_per_chain,
            "std_n_unique_per_chain": float(np.std(n_unique_chains)),
            "cv_n_unique_per_chain": (
                float(np.std(n_unique_chains) / mean_per_chain)
                if mean_per_chain > 0 else float("nan")
            ),
            "n_unique_merged": n_unique_merged,
            "coverage_ratio": coverage_ratio,  # 1 = full overlap, 4 = disjoint
            "acceptance_rate_per_chain": acc_chains,
            "mean_acceptance": float(np.mean(acc_chains)),
            "std_acceptance": float(np.std(acc_chains)),
            "tv_vs_enum": per_rule_summary[rule]["total_variation"],
            "mass_mapped": per_rule_summary[rule]["mass_mapped"],
            "enum_expected_mcc": gt_by_rule[rule]["enum_expected_mcc"] if rule in gt_by_rule else None,
            "mcmc_expected_mcc": gt_by_rule[rule]["mcmc_expected_mcc"] if rule in gt_by_rule else None,
        }
        records.append(rec)

    # Print table
    print(f"{'rule':<28} {'n_uniq/chain':>14} {'cov_ratio':>10} {'mean_acc':>9} {'TV':>6} {'E[MCC]_m':>9}")
    for r in records:
        mean_s = f"{r['mean_n_unique_per_chain']:.0f}±{r['std_n_unique_per_chain']:.0f}"
        print(f"{r['rule_id']:<28} {mean_s:>14} {r['coverage_ratio']:>10.2f} "
              f"{r['mean_acceptance']:>9.3f} {r['tv_vs_enum']:>6.3f} "
              f"{r['mcmc_expected_mcc']:>+9.3f}")

    # Aggregate
    coverage = np.array([r["coverage_ratio"] for r in records])
    tvs = np.array([r["tv_vs_enum"] for r in records])
    cvs = np.array([r["cv_n_unique_per_chain"] for r in records])
    accs = np.array([r["mean_acceptance"] for r in records])
    mcmc_emcc = np.array([r["mcmc_expected_mcc"] for r in records])

    # Pearson-style correlations (simple, no scipy dependency)
    def pearson(x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        mx, my = x.mean(), y.mean()
        sx = np.sqrt(((x - mx) ** 2).sum())
        sy = np.sqrt(((y - my) ** 2).sum())
        if sx == 0 or sy == 0:
            return float("nan")
        return float(((x - mx) * (y - my)).sum() / (sx * sy))

    print("\nCross-rule correlations (n=18):")
    print(f"  coverage_ratio   vs  TV:        r = {pearson(coverage, tvs):+.3f}")
    print(f"  chain-CV(n_uniq) vs  TV:        r = {pearson(cvs, tvs):+.3f}")
    print(f"  mean_acceptance  vs  TV:        r = {pearson(accs, tvs):+.3f}")
    print(f"  coverage_ratio   vs  MCMC E[MCC]: r = {pearson(coverage, mcmc_emcc):+.3f}")
    print(f"  mean_acceptance  vs  MCMC E[MCC]: r = {pearson(accs, mcmc_emcc):+.3f}")

    aggregate = {
        "n_rules": len(records),
        "mean_coverage_ratio": float(coverage.mean()),
        "median_coverage_ratio": float(np.median(coverage)),
        "mean_chain_cv": float(cvs.mean()),
        "mean_acceptance_across_rules": float(accs.mean()),
        "correlations": {
            "coverage_ratio_vs_tv": pearson(coverage, tvs),
            "chain_cv_vs_tv": pearson(cvs, tvs),
            "acceptance_vs_tv": pearson(accs, tvs),
            "coverage_ratio_vs_mcmc_emcc": pearson(coverage, mcmc_emcc),
            "acceptance_vs_mcmc_emcc": pearson(accs, mcmc_emcc),
        },
    }

    out = {"per_rule": records, "aggregate": aggregate}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")

    # Plots
    make_plots(records)


def make_plots(records):
    order = sorted(range(len(records)), key=lambda i: records[i]["tv_vs_enum"])
    rule_ids = [records[i]["rule_id"] for i in order]
    coverage = np.array([records[i]["coverage_ratio"] for i in order])
    mean_u = np.array([records[i]["mean_n_unique_per_chain"] for i in order])
    std_u = np.array([records[i]["std_n_unique_per_chain"] for i in order])
    merged_u = np.array([records[i]["n_unique_merged"] for i in order])
    tvs = np.array([records[i]["tv_vs_enum"] for i in order])
    accs = np.array([records[i]["mean_acceptance"] for i in order])

    # (a) Mean ± std unique-per-chain, plus merged, per rule.
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))
    ax.errorbar(mean_u, y - 0.15, xerr=std_u, fmt="o",
                color="#2a6bcc", label="mean ± std unique/chain", capsize=2)
    ax.scatter(merged_u, y + 0.15, marker="s", color="#cc5a2a",
               label="unique merged (all chains)")
    ax.set_yticks(y)
    ax.set_yticklabels(rule_ids, fontsize=8)
    ax.set_xlabel("Number of unique programs visited")
    ax.set_title("Per-chain vs. merged unique programs\n"
                 "(large gap ⇒ chains explored largely disjoint regions)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "chain_diagnostics_overview.png", dpi=150)
    plt.close(fig)

    # (b) Coverage ratio vs TV, colored by acceptance rate.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(coverage, tvs, c=accs, cmap="viridis",
                    s=70, edgecolor="k", lw=0.5)
    for i, rid in enumerate(rule_ids):
        ax.annotate(rid, (coverage[i], tvs[i]), fontsize=6.8,
                    xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="mean MH acceptance rate")
    ax.set_xlabel("Coverage ratio   (merged_unique / mean per-chain_unique)")
    ax.set_ylabel("TV(enum, MCMC)")
    ax.set_title("Does chain heterogeneity predict disagreement with enum?\n"
                 "1 = chains fully overlap, 4 = chains entirely disjoint")
    ax.axvline(1, color="k", lw=0.5, ls=":")
    ax.axvline(4, color="k", lw=0.5, ls=":")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "chain_heterogeneity_vs_tv.png", dpi=150)
    plt.close(fig)
    print(f"Plots written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
