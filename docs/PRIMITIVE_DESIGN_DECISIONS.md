# Primitive Library Design Decisions

## Context

The Bayesian rule induction model computes posterior probabilities over
hypotheses drawn from a DSL of hand→bool programs. For this to work, the
true rule for each of the 60 gallery rules must be **expressible** in the
DSL — otherwise the model has a hard ceiling: no amount of evidence can
identify a rule that isn't in the hypothesis space.

At depth 5 with the original 57 primitives, only 6 of 60 true rules were
found in the hypothesis space. Analysis revealed that 15 rules were
**structurally inexpressible** regardless of enumeration depth — not because
the programs were too deep, but because the DSL lacked the necessary
computational building blocks.

This document records the design decisions made when extending the primitive
library, including the alternatives considered and the biases introduced.

---

## Decision 1: Sort primitive

### The problem
Six rules require finding rank patterns regardless of card position order:
- `ap_len3_step1_anywhere` (Group 2): 3 cards with consecutive ranks anywhere
- `ap_step1_len3_adj` (Group 3): 3 adjacent cards forming AP in any rank order
- `ap_step2_len4_adj` (Group 3): 4 adjacent cards forming AP in any rank order
- `straight5` (Group 3): 5 cards with consecutive ranks
- `straight5_same_suit` (Group 3): 5-card straight flush
- `straight5_same_color` (Group 3): 5-card straight same color

### Options considered

**A. General `sort_by : (a → int) → list(a) → list(a)`**
- Most general: sort any list by any key function
- Downside: adds branching at every depth (the key function is a lambda argument), significantly expanding the search space
- The generality is not needed — in practice, we only ever sort cards by rank

**B. Targeted `sort_by_rank : list(card) → list(card)`**
- Sorts cards by rank value (ascending)
- Cognitively plausible: "arrange the cards by number" is a natural operation that humans perform when examining a hand
- Minimal search space impact: one extra primitive at type `list(card) → list(card)`, no lambda arguments
- Directly unlocks all 6 rules via compositions like `sort_by_rank $0 |> take 5 |> adjacent_pairs |> all (diff = 1)`

### Decision
**Option B: `sort_by_rank`** — targeted sort by rank value.

### Bias introduced
The model now has a built-in operation for ordering cards by rank. This is a
structural assumption that rank-ordering is a "free" cognitive operation for
the learner. If some rules are harder *because* humans find it difficult to
mentally sort cards, this primitive will mask that difficulty — the model will
treat the sorted representation as equally accessible as the positional one.

This bias is conservative for our purposes: we are interested in which rules
are *structurally* hard (many competing hypotheses) rather than
*procedurally* hard (requiring effortful mental operations). By giving the
model free access to sorting, we isolate structural difficulty from
procedural difficulty.

---

## Decision 2: Maximum suit count

### The problem
Two rules need to determine the most frequent suit without specifying which:
- `three_or_more_same_suit` (Group 1): max suit count ≥ 3
- `four_any_suit_anywhere` (Group 2): max suit count ≥ 4

The DSL has `count_suit` which counts a specific suit, but no way to compute
the maximum across all four suits.

### Options considered

**A. General `maximum : list(int) → int`**
- Requires also enumerating a list of suit counts: `maximum (map (λs. count_suit $0 s) [H,D,S,C])`
- Problem: we don't have a list-of-suits constant, and building one at enumeration time is expensive
- Would also need to solve the "iterate over suits" problem more generally

**B. Four-way disjunction (no new primitive)**
- Express as `or (≥3 count_suit H) (or (≥3 count_suit D) (or (≥3 count_suit S) (≥3 count_suit C)))`
- Depth ~8-9, but theoretically expressible
- Problem: extremely deep, unlikely to be enumerated within budget; also semantically ugly

**C. Targeted `max_suit_count : list(card) → int`**
- Returns the count of the most frequent suit in the hand
- Cognitively plausible: "what's the most common suit?" is a natural summary statistic
- Minimal search space impact: one extra primitive at type `list(card) → int`

### Decision
**Option C: `max_suit_count`** — returns the count of the most common suit.

### Bias introduced
Like `n_unique_suits` (which already exists), this primitive gives the model
"free" access to a suit-frequency summary. The alternative (four-way
disjunction) would be expressible but extremely deep, making these rules
appear artificially hard. By adding the primitive, we're asserting that
"checking if any suit dominates" is a basic cognitive operation, not a
complex derived computation. This seems psychologically reasonable —
noticing a dominant suit is perceptually salient.

---

## Decision 3: Count-where (conditional counting)

### The problem
Two rules need second-order counting — counting how many values meet a threshold:
- `two_pairs_ranks` (Group 2): ≥2 ranks each appearing ≥2 times
- `two_pairs_suits` (Group 2): ≥2 suits appearing exactly twice (+ additional constraints)

The DSL has `length` and `filter` separately but not a direct way to count
elements satisfying a predicate, and more importantly no way to count *how many
distinct values* meet a frequency threshold.

### Options considered

**A. Targeted `n_pairs_rank : list(card) → int` and `n_pairs_suit`**
- Very specific: counts how many ranks/suits appear ≥2 times
- Feels like baking the answer into the vocabulary
- Doesn't generalize to other counting patterns

**B. General `count_where : (a → bool) → list(a) → int`**
- Equivalent to `length (filter pred list)` but as a single primitive
- Semi-general: useful for many counting patterns
- Problem: still need to express "rank R appears ≥2 times" as the predicate argument, which requires nested lambdas and is very deep

**C. Targeted `n_repeated_ranks : list(card) → int`**
- Returns the number of ranks that appear more than once
- `two_pairs_ranks` = `≥2 n_repeated_ranks hand`
- More specific than general `count_where` but less ad hoc than `n_pairs_rank`
- Cognitively plausible: "how many ranks show up more than once?" is a natural question when examining a hand

### Decision
**Option C: `n_repeated_ranks`** — counts ranks appearing more than once.

For `two_pairs_suits`, this primitive alone is not sufficient (it needs
suit-based counting + per-half constraints). We add a parallel
`n_repeated_suits : list(card) → int` for symmetry.

### Bias introduced
These primitives give the model direct access to "repeated value" detection.
The alternative would require nested counting constructions at depth 9+,
making pair-based rules appear extremely hard. The bias is that we're treating
"noticing duplicates" as a basic cognitive operation. This aligns with
psychological evidence that humans are good at detecting repetitions and
matches — it's one of the most basic pattern recognition operations.

Note: `two_pairs_suits` has additional constraints (3 suits per half) that
may still require deep composition even with these primitives. The primitive
makes the *core* of the rule expressible but the full conjunction may still
require depth 7-8.

---

## Decision 4: Bracket-matching rules and sequential state

### The problem
Three rules require sequential/stateful processing:
- `suit_brackets_no_cross` (Group 3): suits form non-crossing nested brackets
- `suit_brackets_nested` (Group 3): suits form properly nested brackets (Dyck word)
- `suit_brackets_interleaved` (Group 3): two bracket types balanced independently

These rules fundamentally require *memory of what came before*: whether a
bracket at position *i* is valid depends on the sequence of brackets from
positions 1 through *i*-1.

### Options considered

**A. General `foldl : (b → a → b) → b → list(a) → b`**
- The standard functional programming solution for sequential accumulation
- Would express bracket matching as: fold with a stack/counter accumulator,
  check final state
- Problem: `foldl` is **Turing-complete** — with appropriate types, it can
  express any computable function on lists. This would cause a combinatorial
  explosion in the search space, as the enumerator must consider all possible
  fold bodies (which themselves are programs) at every depth level.
- The fact that `foldl` unlocks a fundamentally different computational class
  is itself theoretically significant: it suggests bracket-matching rules
  require a different *kind* of cognitive operation (sequential accumulation)
  than the other rules (parallel map/filter/reduce). This distinction does
  not need to be proved empirically within our model — it follows from
  computational complexity theory (regular languages vs context-free
  languages; bracket languages require a pushdown automaton, while all our
  other rules can be verified by finite automata or simple counting).

**B. Constrained `running_sum : (card → int) → list(card) → list(int)`**
- Computes cumulative sums of a card→int mapping applied to each card
- running_sum(f, [c1, c2, c3, ...]) = [f(c1), f(c1)+f(c2), f(c1)+f(c2)+f(c3), ...]
- Can express bracket matching: map each suit to +1 or -1, check that
  running sum never goes negative and ends at 0
- Can also express monotonicity checking (running differences)
- Much more constrained than `foldl`: the accumulator is always an integer,
  the fold body is always addition, only the per-element mapping varies
- Search space impact: moderate — adds one higher-order primitive with a
  card→int lambda argument, producing list(int) output

**C. Declare bracket rules out of scope**
- Accept that the DSL cannot express bracket matching
- Scientifically defensible: the model predicts these rules are "infinitely
  hard" (not in the hypothesis space), which is a genuine theoretical claim
- Downside: we lose 3 of 60 rules from the analysis, and the difficulty
  ranking cannot include them. The model's prediction for these 3 rules
  is a degenerate "undefined" rather than a graded difficulty.

### Decision
**Option B: `running_sum`** — constrained sequential accumulation.

We choose `running_sum` over `foldl` because it provides just enough
sequential power to express bracket matching without making the search space
intractable. The key constraint is that the accumulator is always additive
(integer sum), so the enumerator only needs to explore card→int mappings,
not arbitrary state transitions.

We choose it over Option C (out of scope) because having predictions for
all 60 rules is important for the empirical validity of the model. Even if
the bracket rules are the hardest, we want to see *how* hard the model
considers them, not just declare them missing.

### Bias introduced
`running_sum` is strictly weaker than `foldl` — it can only compute linear
accumulations, not arbitrary sequential functions. This means:
1. The model treats bracket matching as requiring sequential processing
   (correct), but assumes additive accumulation is "cheap" (one primitive
   application). A human verifying brackets does not have O(1) access to
   running sums — they must mentally track state.
2. By making `running_sum` available, we're asserting that the *difficulty*
   of bracket rules comes from the need to compose the right card→int
   mapping and check the right properties of the result, not from the
   sequential processing itself. In reality, the sequential processing
   is likely a major source of difficulty for humans.
3. The alternative (`foldl`) would make bracket rules accessible only at
   enormous search depth, potentially making them appear harder — but for
   the wrong reason (search space size rather than conceptual complexity).

This is the most significant bias in our primitive design. The model's
difficulty predictions for bracket rules should be interpreted with caution:
if they appear relatively "easy" (low entropy), it may be because
`running_sum` makes the sequential aspect too cheap.

---

## Decision 5: Suit-to-integer mapping

### The problem
One rule requires treating suits as ordered values:
- `suits_nonincreasing` (Group 3): suit values (♠=4,♥=3,♦=2,♣=1) are
  non-increasing, and all four suits are present

### Options considered

**A. `suit_to_int : suit → int` with gallery ordering**
- Maps ♦→4, ♠→3, ♣→2, ♥→1 (matching the gallery rule's convention D≥S≥C≥H)
- Cognitively plausible: card players often have a conventional suit ordering

**B. Express via `running_sum` + suit-specific mappings**
- Could check monotonicity using running differences: map each adjacent pair
  to (suit_val(second) - suit_val(first)), check all ≤ 0
- But still needs the suit→int mapping somewhere

### Decision
**Option A: `suit_to_int`** — gallery experiment suit ordering.

### Bias introduced
The ordering ♦ > ♠ > ♣ > ♥ is the convention used in the gallery experiment's
`suits_nonincreasing` rule (D≥S≥C≥H). This is arbitrary — other orderings
exist (e.g., bridge convention ♠ > ♥ > ♦ > ♣). By baking in this specific
ordering, the model can express `suits_nonincreasing` but would struggle
with rules using a different suit ordering convention. Since the gallery
rules use exactly this convention, this is a mild bias — the model is tuned
to the experiment's design.

---

## Decision 6: Zigzag detection

### The problem
One rule requires detecting alternating local extrema:
- `zigzag_ranks` (Group 3): ranks alternate between local maxima and minima

### Options considered

**A. Dedicated `is_zigzag : list(int) → bool`**
- Very specific, bakes the answer in entirely

**B. Express via existing primitives at depth 9+**
- Using nested `adjacent_pairs`: compute differences of rank values between
  adjacent cards, then check that consecutive differences alternate in sign
- Expressible as:
  `all (λ dp → lt (head dp * last dp) 0) (adjacent_pairs (map (λp → sub (rank_val (last p)) (rank_val (head p))) (adjacent_pairs (map rank_val hand))))`
- Very deep (~depth 9) but uses only existing primitives plus the sign check

**C. Add a `signum : int → int` primitive**
- Returns -1, 0, or +1 for negative, zero, positive integers
- Combined with existing adjacent_pairs + map, enables zigzag checking
- General purpose: useful for any sign-related pattern

### Decision
**Option C: `signum`** — general-purpose sign function.

Zigzag detection can then be expressed at depth ~8-9 using:
```
all (λ pair → lt (+ (signum (head pair)) (signum (last pair))) 1)
    (adjacent_pairs (map (λ p → - (rank_val (last p)) (rank_val (head p)))
                        (adjacent_pairs (map rank_val hand))))
```
This checks that consecutive rank differences have opposite signs (their
signums sum to 0, i.e., one is +1 and one is -1, so the sum < 1).

### Bias introduced
`signum` is general-purpose and introduces minimal bias — it's a standard
mathematical operation. The main concern is depth: zigzag requires depth
~8-9 even with `signum`, making it one of the hardest rules to enumerate.
The model will correctly predict zigzag as very difficult.

---

## Summary of new primitives

| Primitive | Type | Rules unlocked | Search space impact |
|-----------|------|---------------|-------------------|
| `sort_by_rank` | `list(card) → list(card)` | 6 | Minimal |
| `max_suit_count` | `list(card) → int` | 2 | Minimal |
| `n_repeated_ranks` | `list(card) → int` | 1-2 | Minimal |
| `n_repeated_suits` | `list(card) → int` | 1-2 | Minimal |
| `running_sum` | `(card → int) → list(card) → list(int)` | 3 | Moderate |
| `suit_to_int` | `suit → int` | 1 | Minimal |
| `signum` | `int → int` | 1 | Minimal |

Total: 7 new primitives (signum in arithmetic + 6 gallery extensions),
bringing the library from 57 to 64.

## Potential biases — summary for presentation

When presenting this model, the following caveats should be noted:

1. **Primitive selection bias**: The primitive library was designed to ensure
   all 60 gallery rules are expressible. This means the model's difficulty
   predictions are *conditional* on having the right vocabulary. If the
   primitive library were different, difficulty rankings would change.

2. **"Free operation" bias**: Each primitive is treated as a unit-cost
   operation in the grammar. In reality, some operations (like sorting,
   running sums, or counting duplicates) may be more cognitively effortful
   than others (like checking a suit). The model treats all primitives as
   equally accessible.

3. **Sequential processing bias**: `running_sum` makes sequential
   accumulation "cheap" (one primitive), which may underestimate the
   difficulty of bracket-matching rules. The decision not to use `foldl`
   (which would make them properly expensive) was made for tractability,
   not cognitive realism.

4. **Sort-as-free bias**: `sort_by_rank` makes rank-ordering a free
   operation, potentially underestimating the difficulty of rules that
   require mental sorting.

5. **Arbitrary suit ordering**: `suit_to_int` bakes in the gallery experiment's
   convention (♦ > ♠ > ♣ > ♥), which matches the `suits_nonincreasing` rule.

These biases are systematic and predictable — they all point in the same
direction (making rules appear easier than they might be for humans). The
model's difficulty predictions should be interpreted as a *lower bound* on
structural complexity: if the model says a rule is hard, it is genuinely
hard even with the full primitive vocabulary.
