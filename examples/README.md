# Examples

Simple demonstrations of the DreamCoder card game rule learning system.

## Scripts

| Script | Purpose | Run Time |
|--------|---------|----------|
| `main_demo.py` | Complete pipeline demo (rules → features → visualization) | ~1 min |

## Running

All examples must be run from the `src/` directory to ensure imports work correctly:

```bash
cd src
python ../examples/main_demo.py
```

Or add the src directory to your PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/card-games-modelling/src"
python examples/main_demo.py
```

## What the Demo Shows

The `main_demo.py` script demonstrates:

1. **Loading Rules**: Loads the 45 rules from the catalogue
2. **Task Generation**: Creates labeled examples (hand, True/False) for each rule
3. **Feature Extraction**: Extracts 104-dimensional features from task examples
4. **Primitive Analysis**: Identifies which primitives each rule uses
5. **Visualization**: Creates heatmaps and plots of primitive usage

## For Canonical Experiments

For serious experiments (overnight runs, ablations, etc.), see:

```bash
src/experiments/run_reference_wakesleep.py  # Primary entry point
```

See `src/experiments/README.md` for details on all canonical experiments.

## Output

Running the demo creates visualizations in `results/`:
- `primitive_usage_heatmap.png` - Which primitives each rule uses
- `feature_statistics.png` - Feature mean/std across tasks
- `primitive_cooccurrence.png` - Which primitives appear together
- `demo_report.json` - Summary statistics
