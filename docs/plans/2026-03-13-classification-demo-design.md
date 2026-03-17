# Design: Classification Demo Extension for Rule Gallery

## Overview

A self-contained demo page extending the rule-gallery paradigm with a **classification phase**: participants see the 6 gallery hands (positive exemplars of a hidden rule) and must classify new test hands as following the rule or not. Test hands are selected using the Bayesian induction model's confusability scores, loaded from a pre-computed JSON file that updates whenever model parameters change.

**Purpose**: Demo for collaborator discussion — explore variants of this extension before committing to a new experimental paradigm.

## Architecture

### No server required

The demo loads a static `diagnosticity_results.json` pre-computed by the existing `run_diagnosticity.py` pipeline in `card-games-modelling/`. When cost function or priors change:

```bash
cd card-games-modelling/src
python -m gallery_analysis.run_diagnosticity \
    --all-rules \
    --n-candidates 5000 \
    --output ../../card-games/rule-gallery/diagnosticity_results.json
```

Refresh the demo page to pick up new scores.

### Self-contained — no modifications to existing gallery

All new files live alongside the existing gallery without touching it:

```
card-games/rule-gallery/
├── (existing files — UNTOUCHED)
├── classification-demo.html        # NEW
├── classification-demo.js          # NEW
└── diagnosticity_results.json      # NEW (generated from model)
```

Only `card-games/index.html` (dev portal) is edited to add a link and parameter builder.

### Shared read-only dependencies

The demo reads from existing assets without modifying them:
- `js/cardex.js` — card rendering
- `rule-gallery/gallery-rules.js` — rule definitions and eval functions
- `rule-gallery/frozen-exemplars.json` — the 6 gallery hands per rule
- `stim/*.png` — card images

## URL Parameters

| Param | Values | Default | Effect |
|-------|--------|---------|--------|
| `nTest` | 1, 3, 6, ... | 3 | Number of test hands per rule |
| `mode` | `binary`, `forced` | `binary` | Binary Yes/No per hand vs. "select all that apply" |
| `feedback` | `0`, `1` | `0` | Show correct/incorrect after each judgment |
| `accumulate` | `0`, `1` | `0` | Keep classified hands visible with results |
| `group` | A, B, C | (none) | Which frozen rule set to load |

## Layout

### Without accumulation (`accumulate=0`)

Gallery hands in a single vertical column (full width). Test hand appears below with Yes/No buttons (binary) or all test hands with checkboxes (forced-choice).

```
┌──────────────────────────────────────────────┐
│  Header: Rule 1/20                            │
├──────────────────────────────────────────────┤
│  GALLERY (full width, vertical column)        │
│  "These 6 hands all follow the same rule:"    │
│  hand 1: [cards]                              │
│  hand 2: [cards]                              │
│  hand 3: [cards]                              │
│  hand 4: [cards]                              │
│  hand 5: [cards]                              │
│  hand 6: [cards]                              │
│                                               │
│  ── TEST ──────────────────────────────────   │
│  "Does this hand follow the same rule?"       │
│  test hand: [cards]                           │
│              [Yes]  [No]                      │
├──────────────────────────────────────────────┤
│  [Next Rule →]                                │
└──────────────────────────────────────────────┘
```

### With accumulation (`accumulate=1`)

Gallery occupies left half. Classified test hands stack on the right half with color-coded borders.

```
┌───────────────────────────────────────────────────────┐
│  Header: Rule 1/20                                     │
├────────────────────────┬──────────────────────────────┤
│  GALLERY (left half)   │  CLASSIFIED (right half)     │
│  vertical column       │  "Your classifications:"    │
│                        │                              │
│  hand 1: [cards]       │  ┌─green border──────────┐  │
│  hand 2: [cards]       │  │ [cards]    You: Yes  ✓│  │
│  hand 3: [cards]       │  └────────────────────────┘  │
│  hand 4: [cards]       │  ┌─red border────────────┐  │
│  hand 5: [cards]       │  │ [cards]    You: Yes  ✗│  │
│  hand 6: [cards]       │  └────────────────────────┘  │
│  hand 7: [cards]       │  ┌─green border──────────┐  │
│  hand 8: [cards]       │  │ [cards]    You: No   ✓│  │
│                        │  └────────────────────────┘  │
├────────────────────────┴──────────────────────────────┤
│  TEST: "Does this hand follow the same rule?"          │
│  test hand: [cards]                                    │
│              [Yes]  [No]                               │
├───────────────────────────────────────────────────────┤
│  [Next Rule →]                                         │
└───────────────────────────────────────────────────────┘
```

- **Green border**: participant's answer was correct (feedback=1 only)
- **Red border**: participant's answer was incorrect (feedback=1 only)
- **Neutral border**: when feedback=0, classified hands still accumulate but without correctness signal

## Test Hand Selection

The `diagnosticity_results.json` contains per-rule arrays of scored hands with `p_accept` (posterior predictive probability) and `confidence` values. On page load, for each rule, select `nTest` hands:

- ~50% ground-truth positives (hands that actually satisfy the rule)
- ~50% ground-truth negatives
- Spread across confidence levels: some easy (high confidence), some ambiguous (low confidence, near p_accept ≈ 0.5)

## Interaction Flow

### Binary mode (`mode=binary`)

1. Show gallery (6 hands) + first test hand
2. Participant clicks Yes or No
3. If `feedback=1`: flash green/red border on the test hand
4. If `accumulate=1`: move judged hand to classified column (right half)
5. Show next test hand, or enable "Next Rule →" when all `nTest` done

### Forced-choice mode (`mode=forced`)

1. Show gallery (6 hands) + all `nTest` test hands with checkboxes
2. Participant checks all they believe follow the rule
3. Click Submit
4. If `feedback=1`: reveal correct answers with green/red highlights
5. If `accumulate=1`: all hands move to classified column with results
6. Enable "Next Rule →"

## Data Recorded

Each trial logs:
- `ruleId`, `testHandIndex`, `hand` (6 cards)
- `groundTruth` (does it actually follow the rule)
- `modelPAccept`, `modelConfidence` (from diagnosticity JSON)
- `participantResponse` (yes/no or selected/not-selected)
- `correct` (boolean)
- `responseTime_ms`

Output as downloadable JSON via the same pattern as `gallery-save.js`.

## Diagnosticity JSON Schema

```json
{
  "metadata": {
    "generated": "2026-03-13T...",
    "config": { "epsilon": 0.01, "prior_mode": "summed", ... }
  },
  "rules": {
    "all_red": {
      "difficulty": { "posterior_entropy": 0.34, "n_effective_hypotheses": 1.4 },
      "test_hands": [
        {
          "hand": [{"suit":"HEARTS","rank":"5"}, ...],
          "ground_truth": true,
          "p_accept": 0.92,
          "confidence": 0.84
        },
        {
          "hand": [{"suit":"SPADES","rank":"K"}, ...],
          "ground_truth": false,
          "p_accept": 0.12,
          "confidence": 0.76
        },
        ...
      ]
    },
    ...
  }
}
```
