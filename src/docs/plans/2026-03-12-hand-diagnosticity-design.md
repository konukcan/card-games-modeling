# Hand Diagnosticity & Classification Difficulty — Design Sheet

## Goal

Build a tool that uses the existing Bayesian rule induction machinery to rate how **diagnostic** (informative / easy-to-classify) a candidate hand is with respect to a given rule. Given a rule and its 6 exemplar hands, how well can a Bayesian ideal learner determine whether a new 7th hand obeys the same rule?

This serves the behavioral experiment design: we want to select test hands that vary in classification difficulty, from "obviously follows the rule" to "genuinely ambiguous."

## Conceptual Framework

### What makes a hand diagnostic?

A hand `h_new` is **diagnostic** for rule `r` if showing it to the ideal learner (who has seen the 6 exemplars) produces a confident classification — either high P(h_new ∈ r | data) or high P(h_new ∉ r | data).

A hand is **ambiguous** if the posterior is split: many high-posterior hypotheses disagree about whether h_new satisfies them.

### Formal definition

After observing 6 exemplars D for rule r, the learner has a posterior over hypotheses:

```
P(hypothesis_j | D) ∝ P(D | hypothesis_j) × P(hypothesis_j)
```

For a new hand h_new, the **posterior predictive probability** that h_new satisfies the true rule is:

```
P(h_new ∈ rule | D) = Σ_j  P(hypothesis_j | D) × I[h_new ∈ ext(hypothesis_j)]
```

where `I[h_new ∈ ext(hypothesis_j)]` is 1 if hypothesis_j accepts h_new, 0 otherwise.

**Diagnosticity metrics**:

1. **Posterior predictive P(accept)**: The weighted vote across hypotheses. Values near 0 or 1 = diagnostic; near 0.5 = ambiguous.

2. **Classification confidence**: `|P(accept) - 0.5| × 2`, scaled to [0, 1]. Higher = more diagnostic.

3. **Agreement entropy**: Among hypotheses that carry ≥1% posterior mass, what fraction agree on h_new? High agreement = diagnostic. Measured as binary entropy of the accept/reject proportions weighted by posterior mass.

4. **Ground truth alignment**: Does the posterior prediction match the actual truth? P(accept) > 0.5 when h_new truly satisfies r = correct prediction. This gives a "calibration" view.

## Existing Pipeline Architecture

All the infrastructure you need already exists. Here's how the pieces fit.

### File Dependency Graph

```
gallery_analysis/analyze.py          ← Top-level orchestrator
├── gallery_analysis/enumerator.py   ← Program enumeration (shared)
│   └── dreamcoder_core/{grammar, enumeration, primitives, type_system}
├── gallery_analysis/exemplars.py    ← Load frozen exemplar hands
│   └── rules/cards.py              ← Card, Hand, Suit, Rank types
├── gallery_analysis/hypothesis_table.py  ← Trivial filter, fingerprint, extension estimation
├── gallery_analysis/bayesian_scorer.py   ← Likelihood + posterior computation
├── gallery_analysis/gallery_rules.py     ← 60 ground-truth rules with predicates
└── gallery_analysis/injection.py    ← Load & merge injected true-rule hypotheses
```

### Key Entry Points

#### 1. Build hypothesis pool (rule-independent, ~10-15 min)

```python
from gallery_analysis.analyze import build_hypothesis_pool, estimate_extensions

# Step 1-4: Enumerate → filter → fingerprint → deduplicate
equiv_classes, pipeline_stats = build_hypothesis_pool(
    max_depth=6,           # AST depth limit
    max_programs=500_000,  # programs to enumerate
    verbose=2,
)
# Returns ~4,135 equivalence classes

# Step 5: Extension sizes (or load from cache)
extensions = estimate_extensions(
    equiv_classes,
    cache_path="gallery_analysis/results/extension_cache_depth6.json",
    verbose=2,
)
# Returns [(ext_size, base_rate), ...] aligned with equiv_classes
```

#### 2. Score hypotheses for a specific rule (rule-specific, ~1 sec per rule)

```python
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.bayesian_scorer import (
    compute_log_likelihood_noisy,
    TOTAL_HANDS,
)

exemplars = load_exemplars()

# For each rule:
rule_id = "all_red"
exemplar_hands = exemplars[rule_id]["hands_primary"]  # 6 hands
n_exemplars = len(exemplar_hands)

# Score each equivalence class
scored = []  # (log_posterior, cls_idx, hit_vector)
for i, (cls, (ext_size, base_rate)) in enumerate(zip(equiv_classes, extensions)):
    pred = cls["predicate"]

    # Compute hit vector: which exemplars does this hypothesis accept?
    hit_vector = []
    for hand in exemplar_hands:
        try:
            hit_vector.append(pred(hand))
        except Exception:
            hit_vector.append(False)
    n_hits = sum(hit_vector)

    # Likelihood (noisy size principle)
    log_lik = compute_log_likelihood_noisy(n_hits, n_exemplars, ext_size, epsilon=0.01)

    # Prior (summed over equivalence class members)
    log_prior = cls["summed_prior"]

    log_post = log_prior + log_lik
    scored.append((log_post, i, hit_vector))

# Normalize to get P(h_j | D)
import math
scored.sort(key=lambda x: -x[0])
max_lp = scored[0][0]
log_norm = max_lp + math.log(sum(math.exp(s[0] - max_lp) for s in scored))
posteriors = [(math.exp(s[0] - log_norm), s[1], s[2]) for s in scored]
# posteriors[j] = (probability, cls_idx, hit_vector)
```

#### 3. Classify a new hand (the new part you build)

```python
def classify_hand(new_hand, posteriors, equiv_classes):
    """
    Posterior predictive: P(new_hand ∈ rule | data).

    Args:
        new_hand: A Hand (list of 6 Card objects)
        posteriors: [(probability, cls_idx, hit_vector), ...] from scoring
        equiv_classes: The shared equivalence classes

    Returns:
        p_accept: float in [0, 1]
    """
    p_accept = 0.0
    for prob, cls_idx, _ in posteriors:
        pred = equiv_classes[cls_idx]["predicate"]
        try:
            if pred(new_hand):
                p_accept += prob
        except Exception:
            pass  # hypothesis crashes on this hand → treat as reject
    return p_accept
```

### Key Data Structures

**Equivalence class** (from `build_hypothesis_pool()`):
```python
{
    "canonical_program": str,       # e.g., "(λ not (has_color $0 BLACK))"
    "canonical_prior": float,       # log P(shortest program)
    "summed_prior": float,          # log Σ P(all programs in class)
    "n_expressions": int,           # number of distinct programs
    "all_programs": [str, ...],     # all program strings
    "fingerprint": str,             # SHA256 hash of behavior on probes
    "predicate": Callable[[Hand], bool],  # the actual function
}
```

**Hand** (from `rules/cards.py`):
```python
Hand = List[Card]  # always 6 cards

@dataclass(frozen=True)
class Card:
    suit: Suit    # CLUBS, DIAMONDS, HEARTS, SPADES
    rank: Rank    # TWO through ACE

class Suit(Enum): CLUBS, DIAMONDS, HEARTS, SPADES
class Rank(Enum): TWO, THREE, FOUR, ..., KING, ACE
```

**Gallery rule** (from `gallery_analysis/gallery_rules.py`):
```python
GALLERY_RULES: Dict[str, Dict] = {
    "all_red": {
        "id": "all_red",
        "group": 1,           # 1=Easy, 2=Medium, 3=Hard
        "answer": "All red",  # human-readable description
        "predicate": Callable[[Hand], bool],
    },
    ...  # 60 rules total
}
```

**Exemplars** (from `gallery_analysis/exemplars.py`):
```python
exemplars = load_exemplars()
# {
#     "all_red": {
#         "hands_primary": [Hand, Hand, ...],  # 6 hands (shown to humans)
#         "hands_reserve": [Hand, Hand, ...],  # 6 hands (for LLM)
#         "group": 1,
#         "answer": "All red",
#     },
#     ...
# }
```

### Constants

- `TOTAL_HANDS = 20_358_520` — C(52, 6), total possible 6-card hands
- Default `epsilon = 0.01` — noise parameter for noisy likelihood
- Default `n_probes = 500` — random hands for fingerprinting
- Default `prior_mode = "summed"` — credit all expressions in equivalence class

## What to Build

### Core Module: `gallery_analysis/hand_diagnosticity.py`

A module with three capabilities:

#### 1. `rate_hand(rule_id, new_hand, posteriors, equiv_classes) → DiagnosticityReport`

For a single hand, compute:
- `p_accept`: posterior predictive probability
- `confidence`: `|p_accept - 0.5| × 2`
- `ground_truth`: does the hand actually satisfy the rule? (from `GALLERY_RULES[rule_id]["predicate"]`)
- `correct_prediction`: does the posterior get it right? (p_accept > 0.5 matches ground_truth)
- `top_5_votes`: for the top 5 posterior hypotheses, do they accept or reject this hand?

#### 2. `rate_hand_set(rule_id, hands, posteriors, equiv_classes) → List[DiagnosticityReport]`

Batch version. Rate many candidate hands at once.

#### 3. `generate_diagnostic_spectrum(rule_id, posteriors, equiv_classes, n_candidates=10000) → DiagnosticSpectrum`

Sample random hands and rate them all, then bin by confidence level to produce a "spectrum" of diagnosticity for this rule:
- How many hands are easy to classify (confidence > 0.8)?
- How many are ambiguous (confidence < 0.2)?
- What's the distribution of P(accept) across random hands?
- Select representative hands at each confidence level (e.g., 5 easy-accept, 5 easy-reject, 5 ambiguous)

### Output Format

```python
@dataclass
class DiagnosticityReport:
    hand: Hand
    rule_id: str
    p_accept: float                    # posterior predictive P(hand ∈ rule | data)
    confidence: float                  # |p_accept - 0.5| × 2, in [0, 1]
    ground_truth: bool                 # does hand actually satisfy the rule?
    correct_prediction: bool           # posterior agrees with ground truth?
    top_hypotheses_votes: List[Dict]   # top 5 hypotheses: {program, prob, accepts_hand}

@dataclass
class DiagnosticSpectrum:
    rule_id: str
    group: int
    n_candidates: int
    # Distribution statistics
    mean_p_accept: float
    std_p_accept: float
    mean_confidence: float
    fraction_high_confidence: float    # confidence > 0.8
    fraction_ambiguous: float          # confidence < 0.2
    accuracy: float                    # fraction of correct predictions
    # Binned distribution
    p_accept_histogram: Dict[str, int] # bins like "0.0-0.1", "0.1-0.2", etc.
    # Selected representative hands
    easy_accept_hands: List[DiagnosticityReport]   # high confidence, accept
    easy_reject_hands: List[DiagnosticityReport]    # high confidence, reject
    ambiguous_hands: List[DiagnosticityReport]      # low confidence
```

### CLI Script: `gallery_analysis/run_diagnosticity.py`

```
Usage:
    cd src
    python gallery_analysis/run_diagnosticity.py \
        --rule all_red \
        --n-candidates 10000 \
        --extension-cache gallery_analysis/results/extension_cache_depth6.json \
        --output gallery_analysis/results/diagnosticity_all_red.json \
        --verbose 2

    # Or for all 60 rules:
    python gallery_analysis/run_diagnosticity.py \
        --all-rules \
        --n-candidates 5000 \
        --extension-cache gallery_analysis/results/extension_cache_depth6.json \
        --output gallery_analysis/results/diagnosticity_all_rules.json
```

### Integration with Existing Pipeline

The key insight is that **all of the expensive computation is already done and cached**:

1. **Hypothesis pool**: `build_hypothesis_pool()` produces ~4,135 equivalence classes. This takes ~10 minutes but only needs to be done once.
2. **Extension sizes**: Cached at `gallery_analysis/results/extension_cache_depth6.json`. Loading is instant.
3. **Per-rule scoring**: Computing posteriors for one rule takes ~1 second (iterate over 4,135 classes × 6 exemplars).
4. **Hand rating**: Evaluating one hand against the posterior takes ~0.5 seconds (iterate over 4,135 predicates).

So rating 10,000 candidate hands for one rule ≈ 10 min setup + 1 sec scoring + ~80 min rating. For all 60 rules × 5,000 candidates each ≈ 10 min setup + 60 sec scoring + ~4,000 min (67 hours) rating.

**Optimization opportunity**: Only evaluate hypotheses with >0.1% posterior mass. Typically ~50-200 hypotheses. This cuts rating time by ~20×, making 60 rules × 5,000 hands ≈ 3-4 hours.

### Extension: Injection Support

The existing pipeline supports **injecting** hand-crafted hypotheses (the true rules) alongside enumerated ones:

```python
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import generate_probe_set

grammar = build_gallery_grammar()
probes = generate_probe_set(500, seed=42)
injected = load_and_validate_injections("gallery_analysis/injected_hypotheses.json", grammar=grammar)
equiv_classes = merge_injected(equiv_classes, injected, probes)
```

The diagnosticity tool should support injection via `--inject` flag, matching the existing `depth_mass_analysis.py` pattern. This ensures the true rule is always in the hypothesis pool.

## Cached Results Available

These files already exist and should be reused:

| File | Contents | Size |
|------|----------|------|
| `gallery_analysis/results/extension_cache_depth6.json` | fingerprint → (ext_size, base_rate) for 4,135 classes | ~200 KB |
| `gallery_analysis/results/depth6_injected_v2.json` | Full analysis results with injected true rules | ~5 MB |
| `gallery_analysis/results/depth_decomposition_data.json` | Per-rule per-depth posterior decomposition | ~1 MB |

## Research Questions This Tool Answers

1. **For experiment design**: "Which new hands should we show participants as test items?" → Select hands spanning the confidence spectrum.

2. **For model validation**: "Does the Bayesian learner's classification match human intuition about easy vs. hard test items?"

3. **For difficulty calibration**: "Are there rules where even unambiguously rule-following hands get low confidence?" → These rules have high posterior entropy, meaning the learner hasn't narrowed down the hypothesis space.

4. **For the size principle**: "Does increasing the number of exemplars (6 → 12) improve classification confidence?" → Run the analysis with `hands_primary + hands_reserve` (12 exemplars) and compare confidence distributions.

## Style Notes

- Follow existing patterns in `gallery_analysis/` — same import structure, verbose levels, CLI argument style
- Use `GALLERY_RULES` as the source of truth for rule predicates
- Reuse `build_hypothesis_pool()` and `estimate_extensions()` from `analyze.py`
- All logging via `print(..., flush=True)` with verbose level checks
- JSON output should be human-readable (`indent=2`)
