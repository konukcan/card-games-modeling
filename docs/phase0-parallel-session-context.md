# Phase 0 Translation Calibration — Parallel Session Context

> **For Claude:** Use this document to pick up Phase 0 work without interrupting the Phase 1b prompt design session.

## What was running

A 120-hypothesis Python-freeform translation run using Qwen (via Ollama) was launched.
It was at ~89/120 when last checked and should now be complete or nearly so.

## Key commands to check status

```bash
# Check if the process is still running
ps aux | grep run_phase0_v2 | grep -v grep

# Check progress (count result files)
ls llm/results/phase0_v2/*.json 2>/dev/null | wc -l

# View recent output
tail -50 llm/phase0_v2_run.out
```

## What to do next

### If the run completed:
1. **Generate a report** — look at pass rates, judge verdicts, common failure modes
2. **Check DSL and WebPPL results** — 3 DSL results were re-run after Option B fix, should show PASS with judge
3. **Consider launching Gemini Flash comparison** on the same 120 hypotheses:
   ```bash
   cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
   nohup caffeinate -d -i -s python -m llm.modeling.run_phase0_v2 \
     --translator gemini-flash --format python-freeform -v \
     > llm/phase0_v2_gemini_run.out 2>&1 &
   ```

### If the run is still going:
- Let it finish. Check `tail -f llm/phase0_v2_run.out` for live progress.

## Key files

| Purpose | File |
|---------|------|
| Phase 0 v2 runner | `llm/modeling/run_phase0_v2.py` |
| Correction loop (Option B) | `llm/modeling/correction_loop.py` |
| Format prompts | `llm/modeling/format_prompts.py` |
| Results directory | `llm/results/phase0_v2/` |
| Run output log | `llm/phase0_v2_run.out` |
| Probe set | `llm/results/probe_set_200.json` |
| Input hypotheses | `card-games/results_rule_induction_v2/` (120 JSONs) |

## Architecture summary

Phase 0 v2 pipeline: NL hypothesis → format prompt → translator (Qwen/Gemini) → code extraction → syntax check → rule-check on gallery hands → LLM judge (gemma2:9b) → retry with correction prompts (up to 2 retries).

Option B (non-executable formats): DSL-constrained and WebPPL skip syntax/rule-check, use judge-only validation.

## Branch

All work is on `feat/phase0-translation-calibration`.
