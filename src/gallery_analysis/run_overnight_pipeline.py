"""
Overnight pipeline: re-run full Bayesian analysis + diagnosticity for all variants.

Applies all fixes:
- Config I targeted + near-miss probes (1,580 probes)
- Injection deduplication (401 → ~260 unique)
- Corrected TOTAL_HANDS (P(52,6) = 14.66B)
- Corrected suit_brackets_no_cross predicate
- Two-pass adaptive extension estimation
- 1M max_programs budget

Usage:
    cd src
    nohup caffeinate -d -i -s python gallery_analysis/run_overnight_pipeline.py > overnight.log 2>&1 &

Expected runtime: ~2-3 hours total (enumeration + 10 scoring variants + 10 diagnosticity runs)
"""

import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("gallery_analysis/results")
INJECT_PATH = "gallery_analysis/data/injected_hypotheses.json"
# v3 cache (separate from v2 to allow A/B comparison if needed)
EXT_CACHE = str(RESULTS_DIR / "extension_cache_depth6_v3.json")
PREFIX = "v3"

# Common flags for enumeration
COMMON = [
    "--depth", "6",
    "--max-programs", "1000000",
    "--inject", INJECT_PATH,
    "--extension-cache", EXT_CACHE,
    "--targeted-probes",
    "--verbose", "1",
]

# All 10 scoring variants
# inject: "all" = 401 hypotheses, "true_only" = 42 true rules only, False = no injection
VARIANTS = [
    ("weighted_canonical_inject",    "canonical", "weighted", "all",       "noisy"),
    ("weighted_summed_inject",       "summed",    "weighted", "all",       "noisy"),
    ("weighted_canonical_trueonly",  "canonical", "weighted", "true_only", "noisy"),
    ("weighted_summed_trueonly",     "summed",    "weighted", "true_only", "noisy"),
    ("uniform_canonical_inject",     "canonical", "uniform",  "all",       "noisy"),
    ("uniform_summed_inject",        "summed",    "uniform",  "all",       "noisy"),
    ("uniform_canonical_trueonly",   "canonical", "uniform",  "true_only", "noisy"),
    ("uniform_summed_trueonly",      "summed",    "uniform",  "true_only", "noisy"),
    ("weighted_canonical_strict",    "canonical", "weighted", "all",       "strict"),
    ("weighted_summed_strict",       "summed",    "weighted", "all",       "strict"),
]


def run_variant(name, prior_mode, scoring_grammar, inject, likelihood_mode):
    """Run analyze.py for one scoring variant."""
    output = str(RESULTS_DIR / f"{PREFIX}_{name}.json")
    cmd = [
        sys.executable, "gallery_analysis/analyze.py",
        "--depth", "6",
        "--max-programs", "1000000",
        "--prior", prior_mode,
        "--grammar", scoring_grammar,
        "--extension-cache", EXT_CACHE,
        "--targeted-probes",
        "--verbose", "1",
        "--output", output,
    ]
    if inject in ("all", "true_only"):
        cmd += ["--inject", INJECT_PATH]
    if inject == "true_only":
        cmd += ["--inject-true-only"]
    if likelihood_mode == "strict":
        cmd += ["--likelihood-mode", "strict"]

    print(f"\n{'='*60}")
    print(f"VARIANT: {name}")
    print(f"  prior={prior_mode}, grammar={scoring_grammar}, inject={inject}, lik={likelihood_mode}")
    print(f"  output: {output}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return None

    print(f"  Done in {elapsed:.1f}s", flush=True)
    return output


def run_diagnosticity(results_file, name):
    """Run diagnosticity analysis for one variant."""
    output = str(RESULTS_DIR / f"{PREFIX}_diagnosticity.json")
    cmd = [
        sys.executable, "gallery_analysis/run_diagnosticity.py",
        "--all-rules",
        "--n-candidates", "10000",
        "--balanced", "500",
        "--extension-cache", EXT_CACHE,
        "--inject", INJECT_PATH,
        "--targeted-probes",
        "--output", output,
        "--verbose", "1",
    ]

    print(f"\n{'='*60}")
    print(f"DIAGNOSTICITY: {name}")
    print(f"  output: {output}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return None

    print(f"  Done in {elapsed:.1f}s", flush=True)
    return output


def main():
    t_total = time.time()

    print("=" * 60)
    print("OVERNIGHT PIPELINE — Full Re-run")
    print("=" * 60)
    print(f"Variants: {len(VARIANTS)}")
    print(f"Max programs: 1,000,000")
    print(f"Probes: Config I (1,580 targeted + near-miss)")
    print(f"Injection dedup: enabled")
    print(f"Adaptive extension: enabled")
    print(flush=True)

    # Phase 1: Run all scoring variants
    print("\n" + "=" * 60)
    print("PHASE 1: Scoring Variants")
    print("=" * 60, flush=True)

    results = {}
    for name, prior, grammar, inject, lik in VARIANTS:
        out = run_variant(name, prior, grammar, inject, lik)
        if out:
            results[name] = out

    print(f"\nPhase 1 complete: {len(results)}/{len(VARIANTS)} variants succeeded")

    # Phase 2: Run diagnosticity for all variants
    # Note: diagnosticity uses the same hypothesis pool but scores differ per variant.
    # For now, run diagnosticity once (it uses its own scoring internally).
    # TODO: make diagnosticity variant-aware
    print("\n" + "=" * 60)
    print("PHASE 2: Diagnosticity (shared pool, all rules)")
    print("=" * 60, flush=True)

    run_diagnosticity(None, "all_rules")

    # Phase 3: Depth decomposition
    print("\n" + "=" * 60)
    print("PHASE 3: Depth Decomposition")
    print("=" * 60, flush=True)

    cmd = [
        sys.executable, "gallery_analysis/depth_mass_analysis.py",
        "--inject", INJECT_PATH,
        "--extension-cache", EXT_CACHE,
        "--output", str(RESULTS_DIR / f"{PREFIX}_depth_decomposition.json"),
        "--verbose", "1",
    ]
    subprocess.run(cmd, capture_output=False)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
