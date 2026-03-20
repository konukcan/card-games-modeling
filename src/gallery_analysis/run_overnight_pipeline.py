"""
Overnight pipeline: re-run full Bayesian analysis for all variants.

Usage:
    cd src
    nohup caffeinate -d -i -s python gallery_analysis/run_overnight_pipeline.py > overnight.log 2>&1 &
"""
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("gallery_analysis/results")
INJECT_PATH = "gallery_analysis/data/injected_hypotheses.json"
EXT_CACHE = str(RESULTS_DIR / "extension_cache_depth6_v2.json")

# inject: "all" = true rules + LLM foils, "true_only" = true rules only, "none" = nothing
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
    output = str(RESULTS_DIR / f"v2_{name}.json")
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

    print(f"\n{'='*60}", flush=True)
    print(f"VARIANT: {name}", flush=True)
    print(f"  prior={prior_mode}, grammar={scoring_grammar}, inject={inject}, lik={likelihood_mode}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  {status} in {elapsed:.1f}s", flush=True)
    return output if result.returncode == 0 else None

def main():
    t_total = time.time()
    print("=" * 60, flush=True)
    print("OVERNIGHT PIPELINE", flush=True)
    print("=" * 60, flush=True)

    successes = 0
    for name, prior, grammar, inject, lik in VARIANTS:
        out = run_variant(name, prior, grammar, inject, lik)
        if out:
            successes += 1

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"DONE: {successes}/{len(VARIANTS)} variants in {elapsed/3600:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
