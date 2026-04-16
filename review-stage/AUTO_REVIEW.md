# Auto-Review Loop — MCMC Program Search (Night 2)

**Session started:** 2026-04-16T08:45:40Z
**Branch:** `aris/mcmc-review-20260411` (continuation of Night 1 branch)
**Loop topic:** Internal-coherence / theoretical-correctness review of MCMC program search — specifically the open issues Night 1 left (C3-tier2, C1-gallery, C2-cap, H-methodology, H-autocorr, H-seeds, H-n_sites-invariance, C5, H5).
**Difficulty:** hard (reviewer memory + debate protocol; 4 rounds max; aim 4 rounds even if score ≥6/10 is reached; positive threshold requires score ≥6 AND ≥3 rounds completed)
**Reviewer backend:** Codex MCP (`mcp__codex__codex`) with `model_reasoning_effort: xhigh` (`✓ Connected` confirmed via `claude mcp list`).
**Compact mode:** true (findings.md appended each round).

Night 1 artifacts are at `night1-archive/`. The branch baseline tests ran 47/47 green in 1732s (~28.9 min) at session start.

**Framing (from ARIS_LAUNCH_PROMPT.md):** Tonight is internal coherence, NOT benchmark chasing. Do NOT try to validate against the rule catalogue. Do NOT read `src/rules/`. Forbidden scope changes must be logged in `NOTES_FOR_HUMAN.md`.

---

## Round 0: Bootstrap

- Baseline tests: **47/47 green in 1732s** (28:52) at the start of the run. No regression from Night 1.
- `review-stage/` created fresh; Night 1 artifacts archived to `night1-archive/`.
- Codex MCP confirmed connected before launching Round 1.
- `REVIEWER_MEMORY.md` initialized empty (Night 1 reviewer was a Claude sub-agent; GPT-5.4 is fresh tonight).
