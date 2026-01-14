# LAPS Research (Archived)

This directory contains documentation from the **Language-Assisted Program Synthesis (LAPS)** subproject.

## Status: PAUSED

The LAPS subproject was paused in January 2025. The core (non-language) DreamCoder model takes priority.

## Key Findings

From the LAPS experiments:

1. **LLM descriptions had 50.5% error rate** initially (hallucinated poker features)
2. **Improved prompting reduced errors to 7.5%** but lost discrimination between rules
3. **LLMs focus on surface features** (colors 63%, ranks 62%) not structural patterns (8%)

## Documents

| File | Contents |
|------|----------|
| `LAPS_deep_dive.md` | Detailed analysis of LAPS approach |
| `LAPS_integration_analysis.md` | Integration with DreamCoder architecture |

## To Resume LAPS

If continuing this research direction:

1. Read the full subproject report at `../laps-subproject-PAUSED/LAPS_SUBPROJECT_REPORT.md`
2. Review the description generator code at `../laps-subproject-PAUSED/description_generator/`
3. See experiment results at `../laps-subproject-PAUSED/results_llm_annotation/`

## Related Active Docs

For current (non-LAPS) architecture, see:
- `docs/ARCHITECTURE.md` - Current system architecture
- `docs/KNOWN_ISSUES.md` - Bug documentation
