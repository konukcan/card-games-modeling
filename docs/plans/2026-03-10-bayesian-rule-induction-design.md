# Bayesian Rule Induction Engine for the Card Gallery Experiment

## Motivation

The rule-gallery experiment presents participants with six hands of cards, all of which satisfy some hidden rule, and asks them to infer what that rule is. Sixty rules span three difficulty levels, and participants vary widely in their ability to identify them. We want a principled computational model of what a rational agent would infer from those same six hands, so that we can predict which rules are easy or hard for humans, understand what alternative hypotheses compete with the true rule, evaluate whether our chosen exemplar hands are diagnostic enough, and score the difficulty of new test hands.

The natural framework is Bayesian concept learning, in the tradition of Tenenbaum's "number game." The posterior probability of a hypothesis given the observed hands is proportional to the product of a prior over hypotheses and a likelihood capturing how well the hypothesis explains the data. The prior comes from a probabilistic context-free grammar over a domain-specific language of card predicates, and the likelihood comes from the size principle: tighter hypotheses that match fewer hands assign higher probability to seeing exactly these six.

## Hypothesis Space

The hypothesis space is defined by enumerating all well-typed programs of type `hand -> bool` from a domain-specific language of 57 cognitively realistic primitives. These primitives cover card accessors (suit, rank, colour), positional operations (head, last, slicing into halves), counting and aggregation (count by suit, sum of ranks, number of unique values), comparisons, boolean connectives, higher-order functions (map, filter, all, any), and basic arithmetic. They were designed to reflect how people naturally talk and think about card patterns, avoiding abstract combinators like compose or flip that lack cognitive plausibility.

The enumeration proceeds top-down, filling typed holes in partial programs using the existing priority-queue enumerator in the modelling project. Rather than a hard depth cutoff, the initial target is depth 7-8, which empirical analysis of the gallery rules suggests is sufficient to capture most rules across all three difficulty groups. The two deepest rules in the gallery are the bracket-matching patterns (suit_brackets_nested, suit_brackets_no_cross), which require stack-like computation and reach depth 8 or beyond in the DSL. Most Group 1 and 2 rules fall within depth 4-6, and the majority of Group 3 rules within depth 5-7.

This is a "hybrid" approach: the enumeration is open-ended (discovering any expressible hypothesis, not just the 60 gallery rules), but reporting is anchored to the 60 gallery rules as ground truth. This lets us discover whether a rational agent's best guess is actually a rule outside the gallery set, which would reveal that the exemplar hands are ambiguous in a way the experimenters did not anticipate.

## Prior Over Hypotheses

The grammar assigns a probability to each program based on its derivation. At each step in the top-down construction, the enumerator chooses which production (primitive or bound variable) to place in a typed hole, and the grammar assigns a normalised probability to that choice among all type-compatible candidates. The total prior probability of a program is the product of these choices along its derivation.

A key design decision concerns how to handle hypotheses that can be expressed in multiple ways. The DSL is expressive enough that many rules have several syntactically distinct but extensionally equivalent formulations. For example, "all cards are red" can be expressed as checking that the number of unique colours is one, or that every card's colour equals red, or that the hand does not contain a black card, among other routes. Under a Solomonoff-style prior, the probability of a concept is the sum over all programs that compute it, so concepts with many short expressions receive a higher prior. This has a cognitive interpretation: if a learner can arrive at a hypothesis via multiple reasoning paths, they are more likely to consider it. However, this multiplicity is partly an artefact of DSL design, and a redundant primitive would inflate the prior of any hypothesis that uses it.

To avoid committing to one interpretation prematurely, the engine reports dual priors for every equivalence class. The canonical prior uses only the single shortest (most probable) expression, treating all other formulations as redundant. The summed prior aggregates across all expressions, in the Solomonoff spirit. Both priors are carried through to posterior computation, producing two parallel rankings. When human behavioural data becomes available, the comparison between these rankings will reveal which prior better predicts human difficulty, turning a modelling assumption into an empirical question.

The initial grammar uses uniform weights (every primitive equally probable). The architecture is designed so that alternative weight schemes, whether hand-tuned, complexity-based, or learned via inside-outside updates from solved tasks, can be swapped in later without re-enumerating. The enumeration produces a weight-agnostic hypothesis table, and the Bayesian scoring layer takes the grammar as a parameter.

## Likelihood and the Size Principle

The likelihood of observing six specific positive-example hands given a hypothesis is computed via the size principle. If a hypothesis matches a large number of possible hands, then seeing exactly these six is not very informative, because the same hypothesis could have produced many other sets. If it matches only a handful of hands, then seeing these six is strong evidence for it. Formally, the likelihood is one over the size of the hypothesis's extension (the number of distinct six-card hands from the deck that satisfy it), raised to the power of the number of observations.

Computing the extension size requires knowing how many of the 20,358,520 possible six-card hands from a standard 52-card deck satisfy a given predicate. For the bulk of hypotheses, this is estimated by Monte Carlo sampling: draw 100,000 random hands, evaluate the predicate on each, and scale the hit rate to the full combinatorial space. With 100,000 samples, the standard error on a base rate of a few percent is around 0.1 percentage points, which is more than sufficient for ranking hypotheses. For the final top-K hypotheses in each rule's posterior, the engine switches to exhaustive evaluation over all 20 million hands to ensure precise extension sizes where they matter most.

## Near-Miss Tolerance

The strict likelihood assigns zero probability to any hypothesis that fails on even one of the six exemplar hands. This is unrealistic for human learners, who may not scrutinise every card in every hand, may tolerate exceptions, or may adopt a rule that captures "almost everything." To model this, the engine supports a noisy likelihood with an error parameter epsilon. Each exemplar is independently treated as a clean observation (drawn from the hypothesis's extension) with probability one minus epsilon, or a noisy observation (drawn uniformly from outside the extension) with probability epsilon. A hypothesis that covers five of six hands receives a small but nonzero likelihood, and if it is simple enough, it can outrank a more complex hypothesis that covers all six.

During enumeration, the likelihood pruning criterion is relaxed accordingly: partial programs are pruned only if they fail on two or more exemplars, since a single miss is still viable under the noise model. This modestly increases the number of surviving hypotheses without dramatically expanding the search space, because most hypotheses that miss one exemplar also miss several.

The per-exemplar hit/miss vector is stored for every hypothesis in the table. At analysis time, posteriors can be recomputed under any value of epsilon without re-enumerating, including a sweep over epsilon to see how tolerance for exceptions changes the rankings.

## Efficiency Techniques

Four techniques make the enumeration tractable at depth 7-8 with 57 primitives.

### Observational Equivalence Pruning

Many syntactically distinct programs compute the same boolean function over hands. Rather than carrying all of them forward, the engine maintains a fingerprint table. A fixed probe set of 200 random hands is generated once. Every completed program is evaluated on these 200 hands, producing a 200-bit boolean vector that is hashed. If two programs produce the same hash, they are extensionally equivalent (with overwhelming probability), and only the shorter one is retained as the canonical representative of the equivalence class.

The risk of a false collision, where two genuinely different functions happen to agree on all 200 probes, is negligible. Two functions whose extensions differ by even one percent of the hand space have less than a 13% chance of agreeing on all 200 probes, and this drops below 1% with 500 probes. For the final top-K, exhaustive evaluation serves as a safety net. This technique typically eliminates 50-80% of redundant hypotheses, which translates directly into less work for the downstream scoring and analysis layers.

### Likelihood Pruning During Enumeration

The six exemplar hands for each gallery rule provide a powerful constraint on which hypotheses are worth exploring. As the enumerator builds a partial program top-down, it can evaluate completed subexpressions on the exemplar hands before the full program is finished. If the structure of the partial program guarantees that it will fail on two or more exemplars regardless of how the remaining holes are filled, the entire subtree is pruned.

The most common case involves conjunctions. If the enumerator is building a program whose outer structure is "A and something", and A already evaluates to false on two exemplars, then no matter what fills the second conjunct, the full program will fail on those two hands. Since the noise model allows at most one miss, this subtree is dead and can be skipped.

This technique is especially powerful for Group 1 rules with distinctive features. When analysing "all cards are red," any hypothesis branch involving a specific black suit is immediately pruned on the first exemplar evaluation. The effective search space narrows dramatically to hypotheses in the neighbourhood of the true rule. The pruning is run separately for each gallery rule's exemplar set, so the engine performs 60 focused enumeration passes rather than one monolithic enumeration.

### Constant Folding and Dead Code Elimination

During enumeration, some subexpressions have no free variables and can be evaluated immediately. If the enumerator fills a hole with an expression like "2 + 3", it replaces it with the constant 5 before continuing. This occasionally reveals that a partial program has collapsed to a trivially true or false function, which can be pruned outright. Similarly, boolean structures like "and false X" or "or true X" are recognised as dead or trivially reducible. The speedup from this technique is modest, around 20-30%, but it is essentially free to implement since the evaluation infrastructure is already in place.

### Memoized Enumeration

The existing enumerator supports memoization of subproblems keyed by type, environment, and cost budget. When filling a hole of a given type, if the same subproblem has been solved before at the same or higher budget, the cached solutions are reused. This provides 1000-8000x speedup at depth 7-8 and is already implemented in the modelling project's enumeration infrastructure.

## Output Format

For each of the 60 gallery rules, the engine produces a structured report containing three layers of analysis.

The difficulty prediction layer reports which hypothesis has the highest posterior probability (the MAP hypothesis), whether it matches the true rule, where the true rule ranks in the posterior, how much posterior mass it receives, and the posterior entropy in bits. Higher entropy means the posterior is spread across many competing hypotheses, predicting that the rule will be harder for participants to identify.

The competing hypotheses layer reports the top 10 equivalence classes by posterior probability. Each entry includes the canonical program (shortest expression), the number of alternative expressions in the DSL, the canonical and summed priors, the extension size, the number of exemplar hits (out of 6), and the posterior under both the strict and noisy likelihood. Equivalence classes are grouped by their observational fingerprint, so entries that are syntactically different but functionally identical appear as a single cluster with combined statistics.

The stimuli calibration layer reports the diagnosticity of the current six frozen exemplar hands: how much posterior mass they place on the true rule. It also provides a baseline for comparison, which could be computed by searching over alternative exemplar sets to find ones that maximise posterior concentration on the true rule.

## Test Hand Difficulty Scoring

Given the posterior distribution induced by the six gallery hands, the engine can evaluate the classification difficulty of any new test hand. The predicted probability that a rational agent classifies the test hand as positive is the sum over all hypotheses of each hypothesis's posterior probability times its prediction for that hand. If this probability is close to one or zero, most high-posterior hypotheses agree, and the hand is easy to classify correctly even without knowing the true rule. If the probability is near 0.5, the hypotheses are split, and the hand is ambiguous.

A complementary measure is the agreement rate: what fraction of the top-K posterior mass agrees with the true rule's classification of the test hand. A hand where agreement is high is one that participants will get right even if they have the wrong hypothesis, because the most plausible wrong hypotheses happen to make the same prediction. A hand where agreement is low is one that discriminates between the true rule and its competitors, making it useful as a diagnostic test item.

This scoring can be run in batch mode over any set of candidate test hands, producing a difficulty ranking that informs experimental design.

## Architecture

The new code lives in a `gallery_analysis/` module within the modelling project, structured as follows:

- `gallery_rules.py` contains all 60 gallery rules ported from JavaScript as Python predicate functions, independent of the existing catalogue.
- `exemplars.py` handles loading the frozen exemplar hands from JSON and generating the random probe set used for fingerprinting.
- `enumerator.py` wraps the core enumeration engine with the gallery-specific efficiency techniques: likelihood pruning against exemplar hands, constant folding, and the relaxed one-miss tolerance.
- `hypothesis_table.py` manages the fingerprint hash table, equivalence class construction, per-exemplar hit vectors, and Monte Carlo extension size estimation.
- `bayesian_scorer.py` computes posteriors under both strict and noisy likelihoods, with both canonical and summed priors, and supports test hand difficulty scoring.
- `analyze.py` is the main entry point that runs the full pipeline and writes per-rule JSON results.
- `results/` is the output directory.

The module imports from `dreamcoder_core/` (grammar, enumeration, type system, primitives) but does not modify it. The core synthesis engine remains clean and general-purpose; all gallery-specific logic is confined to the new module.

## Implementation Phases

The first phase builds a minimum viable pipeline that produces end-to-end results on a handful of rules. This involves porting all 60 gallery rules as Python predicates, loading the frozen exemplar hands, enumerating hypotheses at depth 5-6 with a uniform grammar, applying observational equivalence deduplication, computing strict likelihoods with Monte Carlo extension sizes, and producing the per-rule JSON output with top-10 equivalence classes and dual priors. The phase concludes with a sanity check on 3-5 rules spanning the three difficulty groups, verifying that the MAP hypothesis is sensible and the true rule appears in the top 10.

The second phase adds the remaining efficiency techniques and pushes the enumeration deeper. Likelihood pruning during enumeration (with one-miss tolerance), constant folding, and dead code elimination are integrated into the enumerator wrapper. The depth target increases to 7-8 with memoization. The pipeline runs on all 60 rules, and noisy likelihood scoring with an epsilon sweep is added to the output.

The third phase builds the analysis outputs that the research actually needs. This includes a difficulty prediction ranking across all 60 rules (ordered by posterior entropy or true-rule rank), a competing hypotheses report identifying which confusions are rational, a stimuli calibration analysis comparing the diagnosticity of the current frozen exemplars against what would be achievable with better-chosen hands, and the test hand difficulty scoring system for evaluating new classification items against the posterior distribution.
