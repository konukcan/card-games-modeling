"""Smoke test: run 1 rule, 2 chains, 2000 steps to validate the runner."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from run_mcmc_night3 import _run_one_chain  # noqa: E402
import pickle
from gallery_analysis.exemplars import load_exemplars, generate_probe_set  # noqa

frozen = load_exemplars()
probes = generate_probe_set(n_probes=5000, seed=42)
exemplars = frozen["all_red"]["hands_primary"]

task = {
    "rule_id": "all_red",
    "chain_idx": 0,
    "seed": 123,
    "n_steps": 2000,
    "max_depth": 6,
    "noise_epsilon": 0.01,
    "max_nodes": 25,
    "init_max_depth": 3,
    "beta_start": 0.3,
    "beta_end": 1.0,
    "exemplar_hands": pickle.dumps(exemplars),
    "ext_probes": pickle.dumps(probes),
    "checkpoint_steps": [199, 499, 999, 1999],
}
t0 = time.time()
res = _run_one_chain(task)
print(f"OK: rule={res['rule_id']} chain={res['chain_idx']} "
      f"steps={res['n_steps']} unique={res['n_unique']} "
      f"accept={res['acceptance_rate']:.3f} "
      f"best_lp={res['best_log_posterior']:.2f} "
      f"elapsed={time.time()-t0:.1f}s")
print("Best program:", res["best_program"])
print("Checkpoint unique counts:",
      {cp: len(v) for cp, v in res["checkpoints"].items()})
top5 = sorted(res["visit_counts"].items(), key=lambda kv: -kv[1])[:5]
print("Top-5 by visit count:")
for p, c in top5:
    print(f"  {c:>6}  {p[:100]}")
