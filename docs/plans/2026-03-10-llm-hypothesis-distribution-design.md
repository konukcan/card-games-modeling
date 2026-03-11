# LLM-Based Hypothesis Distribution Modeling

## Motivation

The rule-gallery experiment presents participants with six hands of cards satisfying a hidden rule and asks them to infer that rule. Sixty rules span three difficulty levels. In parallel with the Bayesian program-induction model (documented separately), we want a complementary computational model that uses large language models to recover the landscape of plausible alternative hypotheses for each gallery.

This project serves three purposes. First, it provides complementary validation: LLM-based difficulty predictions can be compared against the Bayesian PCFG model to see which better predicts human behavior. Second, it enables hypothesis discovery: LLMs may surface plausible rules that the grammar-based DSL cannot express. Third, it functions as a standalone cognitive model, using LLM hypothesis distributions as a proxy for human priors, producing independent difficulty scores for test hands.

## Project Structure

The existing `card-games/llm_experiment/` module moves to `card-games-modelling/llm/`, placing all modeling work (Bayesian and LLM-based) in a single project. The new hypothesis modeling code lives alongside the existing experiment code.

```
card-games-modelling/
├── src/
│   ├── dreamcoder_core/
│   ├── gallery_analysis/         # Bayesian model
│   ├── rules/
│   └── experiments/
├── llm/
│   ├── experiment/               # Moved from card-games/llm_experiment/
│   │   ├── gemini_client.py
│   │   ├── rule_induction.py
│   │   ├── rule_induction_prompts.py
│   │   ├── two_category.py
│   │   ├── conditions.py
│   │   ├── judge.py
│   │   └── pipeline/
│   ├── modeling/                  # NEW
│   │   ├── elicitation.py        # Enumeration prompts (5 variants)
│   │   ├── sampler.py            # Repeated sampling with temperature
│   │   ├── translator.py         # Hypothesis → Python lambda
│   │   ├── validator.py          # Syntax + semantic + cross-gen checks
│   │   ├── clustering.py         # Group by extensional equivalence
│   │   ├── distribution.py       # Build probability distribution
│   │   └── difficulty.py         # Test-hand difficulty scoring
│   ├── results/                   # All results (moved + new)
│   │   ├── rule_induction_v2/
│   │   ├── two_category_v2/
│   │   └── hypothesis_modeling/
│   ├── analysis/
│   └── shared/
│       ├── gemini_client.py
│       └── stimuli.py            # Loads from ../../card-games/rule-gallery/
```

Stimuli JSON files remain in `card-games/rule-gallery/` as the source of truth. All internal path references are updated during the move.

## System Prompt

The system prompt stays minimal and unchanged from the existing v2 experiment:

```
You will be presented with hands of six cards sampled from a standard
52-card deck.

Card notation:
- Each card is written as its rank followed by a suit symbol.
- Ranks: 2, 3, 4, 5, 6, 7, 8, 9, 10, J (Jack), Q (Queen), K (King), A (Ace)
- Suits: ♠ (Spades), ♥ (Hearts), ♦ (Diamonds), ♣ (Clubs)
- Each hand has 6 cards shown in a fixed order from left to right.
```

The rationale is that feature activation (whether the model attends to colors, positions, sums, etc.) is an experimental manipulation tested via prompt variants, not something baked into the baseline. A human participant does not arrive as a blank slate, but the system prompt's card notation already implicitly activates suit and rank awareness, which is sufficient for the baseline.

## Prompt Designs

### Core Enumeration Prompt (Baseline / Variant 1)

```
Below are {N} hands of six cards. All of these hands satisfy the same
hidden rule.

{hands}

The rule could involve the ranks, suits, or positions of the cards in
the hand, or any combination of these features.

List up to 5 hypotheses for what the hidden rule might be, from most to
least likely. Tag each as [HIGH], [MEDIUM], or [LOW] confidence.

Format each hypothesis like this:
<hypothesis rank="1" confidence="HIGH">your hypothesis here</hypothesis>
```

If Phase 0 translation calibration reveals that hypotheses are too ambiguous to translate reliably without additional context, a one-sentence evidence field is added:

```
<hypothesis rank="1" confidence="HIGH">
Rule: your hypothesis here
Evidence: one sentence explaining why
</hypothesis>
```

The decision to include or exclude evidence is made empirically based on Phase 0 metrics (see below).

### Sampling Prompt

The existing v2 rule induction prompt is reused unchanged. It requests a single best-guess hypothesis wrapped in `<hypothesis>` tags. This provides backward compatibility with the 120 existing results (60 rules x 2 models at temperature=0.0), which become the first data points in the sampling distribution.

### Prompt Variants for Sensitivity Pilot

Five variants are tested on a representative 15-rule subset (5 per difficulty group) to measure how prompt framing affects the hypothesis landscape. All variants modify only the user prompt; the system prompt stays minimal throughout.

**Variant 1 (Baseline):** The core enumeration prompt defined above.

**Variant 2 (Feature-primed):** Replaces the single hint line with an explicit feature inventory:

```
Consider properties such as: individual card ranks and suits, colors
(red/black), card positions (left-to-right order, first half vs second
half), numerical relationships (sums, differences, consecutive ranks),
and structural patterns (pairs, runs, alternations, matching across
positions).
```

This tests whether listing feature categories broadens or biases the hypothesis space.

**Variant 3 (Minimal):** Removes all hints about what features to consider. The prompt goes directly from presenting the hands to requesting hypotheses.

```
Below are {N} hands of six cards. All of these hands satisfy the same
hidden rule.

{hands}

List up to 5 hypotheses for what the hidden rule might be, from most to
least likely. Tag each as [HIGH], [MEDIUM], or [LOW] confidence.

<hypothesis rank="1" confidence="HIGH">your hypothesis here</hypothesis>
```

This tests whether the baseline's feature hints do meaningful work.

**Variant 4 (Cognitive framing):** Frames the task as human-like guessing:

```
Below are {N} hands of six cards. All of these hands satisfy the same
hidden rule.

{hands}

Imagine you are a participant in a psychology experiment. You have no
expert knowledge of card games or mathematics — just everyday
familiarity with playing cards and common sense. What rules would you
guess these hands follow?

List up to 5 hypotheses from most to least likely, focusing on rules
that an ordinary person would plausibly consider. Tag each as [HIGH],
[MEDIUM], or [LOW] confidence.

<hypothesis rank="1" confidence="HIGH">your hypothesis here</hypothesis>
```

This tests whether framing the task as "what would a human guess" shifts the distribution toward more cognitively plausible hypotheses.

**Variant 5 (Numeric probabilities):** Replaces qualitative confidence tags with numeric probability assignments:

```
Below are {N} hands of six cards. All of these hands satisfy the same
hidden rule.

{hands}

The rule could involve the ranks, suits, or positions of the cards in
the hand, or any combination of these features.

List up to 5 hypotheses for what the hidden rule might be, from most to
least likely. For each, assign an approximate probability (they should
sum to at most 1.0, leaving room for unlisted possibilities).

<hypothesis rank="1" probability="0.45">your hypothesis here</hypothesis>
```

This serves double duty: it is both a prompt sensitivity test and a calibration experiment for numeric probability estimates. If the numbers are well-rank-ordered and correlate with sampling frequencies on the pilot subset, they could be used more broadly. If not, the qualitative tags are confirmed as the right design choice.

## Experimental Phases

### Phase 0: Translation Calibration

Phase 0 determines whether LLM-generated hypotheses can be reliably translated to Python lambda functions, and if so, which translator model and prompting strategy to use. It runs on the 120 existing single-hypothesis results from the rule induction experiment (60 rules x 2 models), which span all three difficulty groups and include hypotheses ranging from simple ("all cards are red") to complex ("the ranks alternate between ascending and descending").

#### Translation Grid

Four translator models, three generations per hypothesis, two visibility conditions (with and without gallery hands shown to the translator):

| Translator | Cost | Notes |
|---|---|---|
| Claude Opus 4.6 | Free (subscription, via `claude -p`) | Quality ceiling |
| Claude Sonnet | Free (subscription, via `claude -p`) | Mid-tier, different behavior |
| Gemini Flash | ~$0.50 | Same ecosystem as hypothesis generator |
| Qwen2.5-Coder-14B | Free (local, via Ollama) | Code-specialized local model |

Total translations: 120 hypotheses x 4 translators x 3 generations x 2 conditions = 2,880.

Each translation produces a Python lambda with signature `hand -> bool`, where a hand is a list of 6 Card objects with `.rank` (int, 2-14 where J=11, Q=12, K=13, A=14) and `.suit` (str: 'S', 'H', 'D', 'C'). This matches the card representation in `card-games-modelling/src/rules/cards.py`.

The with-hands condition shows the translator the 6 gallery hands alongside the hypothesis text, allowing it to disambiguate vague phrasing. The without-hands condition shows only the hypothesis text. Comparing the two reveals whether hands improve translation accuracy or cause overfitting to the examples.

#### Validation Pipeline

**Syntactic validation:** Does the lambda parse as valid Python? Does it accept a hand and return a bool? Does it run without errors on a sample hand? Fully automated via `exec()`.

**Semantic validation:** Does the lambda return True for 6/6 (or at least 5/6) of the rule's gallery hands? Automated evaluation against frozen exemplars.

**Cross-generation consistency:** Run all 3 generated lambdas on a shared probe set of 200 random hands (the same probe set used for the Bayesian model's observational equivalence fingerprinting). If all 3 produce identical 200-bit boolean vectors, the hypothesis is unambiguous and translation is reliable. If 2/3 agree, majority vote gives the right predicate. If all 3 disagree, the hypothesis is genuinely vague.

#### Decision Metrics

| Metric | Threshold | Action if below threshold |
|---|---|---|
| Syntactic success rate | >95% | Fix the code-generation prompt format |
| Semantic match rate (6/6 on gallery hands) | >80% | Add reasoning/evidence to enumeration prompt |
| Cross-generation agreement (3/3 on probes) | >70% | Add reasoning/evidence to enumeration prompt |

#### Additional Outputs

**Translator leaderboard:** Which model produces the highest semantic match rate and cross-generation agreement, at what cost? This determines the translator for production runs.

**With/without hands comparison:** If without-hands translations still pass at high rates, use the without-hands condition for production (cleaner, no overfitting risk). If they drop substantially, either use hands as context or add reasoning to the enumeration prompt.

**Error taxonomy:** For hypotheses that fail translation, categorize why: vague language ("the cards form a pattern"), domain ambiguity ("the hand has pairs" — how many?), conceptual complexity ("the suits form a bracket structure"), or translator limitation (clear hypothesis, code model can't express it). This informs not just whether to add reasoning, but what kind of disambiguation to request.

### Phase 1: Pilot Experiments

Three pilots run before committing to the full 60-rule experiment.

**Prompt sensitivity pilot (75 calls, ~$2.50):** 15 rules (5 per difficulty group) x 5 prompt variants x 6 hands x Flash. Measures whether prompt framing significantly shifts the hypothesis landscape. If a variant produces notably richer or more diverse results, it is promoted to Phase 2.

**Order permutation pilot (45 calls, ~$1.50):** 15 rules x 3 random shuffles of hand order x 6 hands x Flash. Measures whether the order in which the 6 hands are presented affects which hypotheses the model generates. If order matters significantly, Phase 2 averages over shuffles; if not, single-order results are trusted.

**Temperature pilot (150 calls, ~$2):** 5 rules x 3 temperatures (0.5, 0.7, 1.0) x 10 samples x Flash. Measures how temperature affects the diversity-noise tradeoff in repeated sampling. The temperature that produces the most diverse hypotheses without excessive noise is used for Phase 2.

### Phase 2: Core Experiments

**Enumeration (480 calls, ~$38):** 60 rules x 4 hand counts (3, 6, 9, 12) x 2 models (Flash, Pro). Uses the baseline enumeration prompt (or the best variant from Phase 1 if one clearly dominates). The four hand counts exploit the nested design of the stimuli (hands[0:3] subset of hands[0:6] subset of hands[0:9] subset of hands[0:12]) to measure how the hypothesis distribution narrows with additional evidence, directly paralleling the Bayesian model's size principle.

**Repeated sampling (1,200 calls, ~$15):** 60 rules x 6 hands x 20 samples x temperature=0.7 x Flash. Uses the existing v2 single-hypothesis prompt. Produces frequency-based distributions to cross-validate against the enumerated rankings, and captures long-tail hypotheses that would not appear in a top-5 list.

### Budget Summary

| Phase | Calls | Cost |
|---|---|---|
| Phase 0 (translation calibration) | 2,880 translations | ~$1 (mostly free via subscription/local) |
| Phase 1 (pilots) | 270 | ~$6 |
| Phase 2 (core experiments) | 1,680 | ~$53 |
| **Total** | **~4,830** | **~$60** |

Remaining budget of $40-140 is reserved for follow-up experiments informed by results (expanding to additional models, testing specific rules where the Bayesian model makes surprising predictions, etc.).

## Analysis Pipeline

### Stage 1: Hypothesis Collection

Parse all enumeration and sampling responses. Extract structured hypothesis objects containing: text, rank, confidence tag, rule_id, model, prompt variant, num_hands, temperature, and timestamp.

### Stage 2: Translation to Code

Translate each unique hypothesis text to a Python lambda using the best translator identified in Phase 0. Three generations per hypothesis; majority vote on a 200-hand probe set determines the canonical lambda. Each lambda follows the card representation from `cards.py` (card.rank as int 2-14, card.suit as str S/H/D/C).

### Stage 3: Extensional Clustering

Evaluate every validated lambda on the shared 200-hand probe set (the same set used for the Bayesian model's observational equivalence fingerprinting). Hash the 200-bit boolean vector. Group hypotheses with identical hashes into equivalence classes. Each class gets a canonical lambda, a count of contributing hypotheses, and the list of source phrasings. This mirrors exactly what the Bayesian model does with its fingerprint-based equivalence pruning, enabling direct comparison of hypothesis spaces.

### Stage 4: Distribution Construction

For each rule, build two complementary probability distributions over equivalence classes.

The enumeration distribution uses rank-weighted scoring: a hypothesis at rank 1 with HIGH confidence contributes more mass than rank 5 with LOW. Scores are aggregated across hand counts and models, with the weighting scheme documented and adjustable.

The sampling distribution uses frequency: each equivalence class's probability is proportional to how often it was sampled across the 20 temperature-0.7 runs. This is the more principled estimate but captures only hypotheses the model generates as its single best guess.

Both distributions are reported. Their correlation is itself a measure of robustness: if they agree, confidence in the distribution is high; if they diverge, the enumeration and sampling prompts are activating different parts of the model's hypothesis space, which is scientifically interesting in its own right.

### Stage 5: Test-Hand Difficulty Scoring

Given the posterior distribution over equivalence classes for a rule, the predicted probability that a rational agent classifies a test hand as positive is:

P(hand is positive) = sum over all classes of P(class) times class_predicate(hand)

If this probability is close to 1 or 0, most high-probability hypotheses agree, and the hand is easy to classify correctly even without knowing the true rule. If the probability is near 0.5, the hypotheses are split, and the hand is ambiguous.

A complementary measure is the agreement rate: the fraction of posterior mass that agrees with the true rule's classification of the test hand. High agreement means the hand is easy because even wrong hypotheses make the right prediction. Low agreement means the hand discriminates between the true rule and its competitors, making it useful as a diagnostic test item.

Both metrics parallel the Bayesian model's test-hand scoring, enabling direct comparison of difficulty predictions.

## Outputs

For each of the 60 rules, the pipeline produces:

**Hypothesis landscape:** The top-K equivalence classes with canonical lambda, list of contributing natural-language phrasings, rank/frequency scores, and extension size (number of valid hands from a 52-card deck that satisfy the predicate).

**Difficulty prediction:** Posterior entropy in bits, MAP (maximum a posteriori) hypothesis, rank of the true rule in the posterior, and posterior mass on the true rule.

**Test-hand scores:** For any candidate test hand, the predicted classification probability and agreement rate, enabling batch scoring of candidate test items for experimental design.

**Cross-model comparison:** How Flash versus Pro distributions differ in terms of hypothesis diversity, true-rule recovery, and difficulty predictions.

**Hand-count analysis:** How the distribution narrows from 3 to 6 to 9 to 12 hands, paralleling the Bayesian size principle. This reveals which hypotheses are ruled out by additional evidence and which persist as strong alternatives.

**Cross-framework comparison:** Correlation of difficulty rankings across 60 rules between the LLM distribution and the Bayesian PCFG distribution. Once pilot behavioral data is available, both models' predictions can be evaluated against human classification accuracy.

## Execution Order

```
Phase 0: Install Qwen2.5-Coder-14B (ollama pull qwen2.5-coder:14b)
         Run translation calibration on 120 existing hypotheses
         → Decide: reasoning in enumeration prompt? Best translator?
           Hands visible to translator?
              │
Phase 1: Run prompt sensitivity pilot (15 rules × 5 variants)
         Run order permutation pilot (15 rules × 3 shuffles)
         Run temperature pilot (5 rules × 3 temperatures)
         → Decide: best variant for Phase 2? Order averaging needed?
           Optimal temperature?
              │
Phase 2: Move llm_experiment/ to card-games-modelling/llm/
         Update all internal path references
         Run core enumeration (60 rules × 4 hand counts × 2 models)
         Run repeated sampling (60 rules × 20 samples)
              │
Phase 3: Translate all hypotheses to Python lambdas
         Cluster by extensional equivalence
         Build enumeration and sampling distributions
         Score test hands
              │
Phase 4: Compare with Bayesian model predictions
         Generate cross-framework difficulty rankings
         Prepare difficulty-scored test items for behavioral pilot
```

## Dependencies

**API access:** Google Gemini API key (existing), Anthropic subscription for Claude Opus 4.6 and Sonnet (existing).

**Local models:** Qwen2.5-Coder-14B via Ollama (to be installed; requires ~9GB RAM).

**Stimuli:** `card-games/rule-gallery/stimuli_llm.json` (60 rules, 12 hands each, frozen).

**Existing data:** `card-games/results_rule_induction_v2/` (120 result JSONs from v2 experiment).

**Code dependencies:** `google-genai` SDK (existing), `anthropic` SDK or `claude -p` CLI (existing), `ollama` Python client (to be added).

**Cross-project:** The shared 200-hand probe set should be coordinated with the Bayesian model's fingerprint set in `gallery_analysis/` so that extensional equivalence is measured identically across both frameworks.
