# Unused Modules

These modules were developed but are not currently used by any active code.

## Files

| File | Purpose | Why Unused |
|------|---------|------------|
| `recognition_variants.py` | Architectural variants of recognition model (attention, embedding heads, etc.) | Experiments showed standard ContrastiveRecognitionModel outperformed all variants |
| `html_report.py` | Generate HTML reports of experiment results | Never integrated into experiment pipeline |
| `visualization.py` | Text-based visualization utilities | Never integrated into experiment pipeline |

## Notes

- `recognition_variants.py` contains classes explicitly marked "TODO: DELETE" with experiment results showing they performed worse than the baseline
- These files are kept for reference in case the functionality is needed in future
- All imports of these modules are in `archived/` experiment scripts
