"""Smoke test with reduced probes to measure speedup."""
from __future__ import annotations
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from run_mcmc_night3 import _run_one_chain  # noqa: E402
from gallery_analysis.exemplars import load_exemplars, generate_probe_set  # noqa

frozen = load_exemplars()
exemplars = frozen["all_red"]["hands_primary"]

for n_probes in (500, 1000, 2000):
    probes = generate_probe_set(n_probes=n_probes, seed=42)
    task = {
        "rule_id": "all_red",
        "chain_idx": 0,
        "seed": 123,
        "n_steps": 1000,
        "max_depth": 6,
        "noise_epsilon": 0.01,
        "max_nodes": 25,
        "init_max_depth": 3,
        "beta_start": 0.3,
        "beta_end": 1.0,
        "exemplar_hands": pickle.dumps(exemplars),
        "ext_probes": pickle.dumps(probes),
        "checkpoint_steps": [999],
    }
    t0 = time.time()
    res = _run_one_chain(task)
    dt = time.time() - t0
    print(f"n_probes={n_probes:5d}  1000 steps in {dt:.1f}s  ({dt:.1f} ms/step)  "
          f"accept={res['acceptance_rate']:.3f}  unique={res['n_unique']}")
