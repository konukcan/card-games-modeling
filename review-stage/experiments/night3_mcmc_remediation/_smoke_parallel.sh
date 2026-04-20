#!/bin/bash
# Tiny parallel smoke: 2 rules, 4 chains, 500 steps, 4 workers.
# Verifies ProcessPool + checkpoint serialization + output file layout.
set -e
cd "$(dirname "$0")"
~/miniforge3/bin/python -c "
import json, sys
from pathlib import Path
cfg = json.loads(Path('config.json').read_text())
cfg['mcmc']['n_steps'] = 500
cfg['mcmc']['n_checkpoints'] = 3
Path('config_smoke.json').write_text(json.dumps(cfg, indent=2))
"
# Run with a temporary config swapped in
cp config.json config.backup.json
cp config_smoke.json config.json
~/miniforge3/bin/python run_mcmc_night3.py --n-workers 4 --only-rules all_red,all_even 2>&1 | tail -30
cp config.backup.json config.json
rm config.backup.json config_smoke.json
