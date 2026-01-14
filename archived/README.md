# Archived Code

This directory contains historical implementations and exploratory scripts preserved for reference.

**These files are NOT part of the active codebase** and should not be imported by production code.

## Directory Index

| Directory | Contents | Why Archived |
|-----------|----------|--------------|
| `legacy_runners/` | Old runner scripts (v3, progressive, incremental) | Superseded by `experiments/run_reference_wakesleep.py` |
| `legacy_recognition/` | GRU and Set Transformer models | Superseded by contrastive recognition model |
| `legacy_compression/` | Old monolithic compression module | Refactored into `dreamcoder_core/compression/` package |
| `deprecated_experiments/` | Early experiments (warmstart, softmax) | One-off experiments, findings documented elsewhere |
| `parameter_tuning/` | Ablation and comparison scripts | Parameter exploration, not user-facing |
| `model_development/` | Recognition model training variants | Experimental training approaches |
| `analysis/` | Evaluation and diagnostic scripts | Debugging tools, not production |
| `special_purpose/` | Task-specific scripts (16 rules focus) | Narrow scope experiments |

## When to Look Here

- **Historical reference**: Understanding design decisions and why certain approaches were abandoned
- **Reproducing specific results**: If a paper references a specific ablation study
- **Learning from failures**: Each subdirectory documents what didn't work and why

## File Counts

| Directory | Python Files |
|-----------|-------------|
| `parameter_tuning/` | 20 |
| `model_development/` | 4 |
| `analysis/` | 7 |
| `special_purpose/` | 2 |
| `legacy_runners/` | 6 |
| `legacy_recognition/` | 2 |
| `legacy_compression/` | 1 |
| `deprecated_experiments/` | 6 |
| **Total** | **48** |

## Important Notes

1. These files may have broken imports (they reference old module paths)
2. Some experiments require specific configurations that are no longer documented
3. For current experiments, use scripts in `src/experiments/` instead

## See Also

- `src/experiments/run_reference_wakesleep.py` - The canonical reference implementation
- `src/experiments/ARCHITECTURE.md` - Current architecture documentation
- `docs/KNOWN_ISSUES.md` - Bug documentation and lessons learned
