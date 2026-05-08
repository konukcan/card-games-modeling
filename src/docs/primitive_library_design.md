# Primitive Library Design Document

> **⚠️ STALE — December 2024.** This document describes v3.1 of the primitive library (59 base + 4 parallel = 63). The current authoritative library is v5 in `src/dreamcoder_core/primitives.py` (**64 primitives**; MCMC subset = 62 minus `true`/`false`). The category labels and counts in this doc no longer match the code. See `src/docs/library-explanations.tex` (also pending regen) for narrative context. For the live count, run: `python -c "import sys; sys.path.insert(0,'src'); from dreamcoder_core.primitives import build_primitives; print(len(build_primitives()))"`.

**Version:** 3.1 (with Parallel Extensions)
**Date:** December 2024
**Status:** Active Development

---

## Executive Summary

This document describes the design of the domain-specific language (DSL) used in our DreamCoder-inspired program synthesis system for modeling how humans learn card game rules. The library currently contains **59 base primitives** organized into 10 categories, plus **4 experimental parallel primitives** for structural pattern matching.

The key design principle is **cognitive realism**: primitives should reflect how humans naturally think and talk about card games, rather than providing mathematically minimal but cognitively opaque abstractions.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Complete Primitive Reference](#complete-primitive-reference)
3. [Primitive Decomposition Map](#primitive-decomposition-map)
4. [Higher-Order Functions in Depth](#higher-order-functions-in-depth)
5. [Parallel Primitives (Experimental)](#parallel-primitives-experimental)
6. [Automata-Theoretic Expressiveness](#automata-theoretic-expressiveness)
7. [Design Decisions and Rationale](#design-decisions-and-rationale)
8. [Evolution History](#evolution-history)

---

## Design Philosophy

### Core Principles

1. **Cognitive Realism over Mathematical Minimality**
   - Primitives should be "directly nameable"—expressible in short natural language phrases
   - Example: `all_same_suit` is more cognitively real than `eq 1 (length (unique (map get_suit hand)))`
   - Humans think "are they all hearts?" not "is the length of the unique suits exactly one?"

2. **Domain-Specific Operations First**
   - Include operations humans use naturally when discussing cards
   - `has_suit hand SPADES` vs `any (λc. eq SPADES (get_suit c)) hand`
   - The direct query version matches the mental operation

3. **Search Tractability**
   - Shorter programs are found exponentially faster
   - A depth-3 program requires ~1000x less search than depth-6
   - "Compiled" primitives like `all_same_color` reduce search dramatically

4. **Avoid Abstract Combinators**
   - Removed: `compose`, `flip`, `const`, `id`
   - These have low cognitive reality—people don't think in terms of function composition
   - They also expand the search space without proportional benefit

5. **Minimal Numeric Constants**
   - Only include 0-5 for basic counting
   - Removed: rank thresholds (10, 11, 12, 13, 14), game scores (17, 21)
   - Rules should use relative comparisons, not hardcoded thresholds

---

## Complete Primitive Reference

### Category 1: Constants (14 primitives)

| Primitive | Type | Description |
|-----------|------|-------------|
| `CLUBS` | `Suit` | Suit constant |
| `DIAMONDS` | `Suit` | Suit constant |
| `HEARTS` | `Suit` | Suit constant |
| `SPADES` | `Suit` | Suit constant |
| `RED` | `Color` | Color constant |
| `BLACK` | `Color` | Color constant |
| `0`, `1`, `2`, `3`, `4`, `5` | `Int` | Counting constants |
| `true` | `Bool` | Boolean constant |
| `false` | `Bool` | Boolean constant |

### Category 2: Card Accessors (4 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `get_suit` | `Card → Suit` | Extract suit from card | **Yes** |
| `get_rank` | `Card → Rank` | Extract rank from card | **Yes** |
| `rank_val` | `Card → Int` | Get numeric value (2-14) | **Yes** |
| `get_color` | `Card → Color` | Get color (red/black) | Derived: `λc. if (or (eq (get_suit c) HEARTS) (eq (get_suit c) DIAMONDS)) RED BLACK` |

### Category 3: Position Access (5 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `head` | `[a] → a` | First element | **Yes** |
| `last` | `[a] → a` | Last element | Derived: `λxs. at xs (- (length xs) 1)` |
| `at` | `[a] → Int → a` | Element at index | **Yes** |
| `length` | `[a] → Int` | List length | **Yes** |
| `reverse` | `[a] → [a]` | Reverse list | **Yes** (not decomposable with our primitives) |

### Category 4: List Slicing (7 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `take` | `Int → [a] → [a]` | First n elements | **Yes** |
| `drop` | `Int → [a] → [a]` | Skip first n elements | **Yes** |
| `zip_with` | `(a → b → c) → [a] → [b] → [c]` | Combine lists with function | **Yes** |
| `adjacent_pairs` | `[a] → [[a]]` | Consecutive pairs | Derived: `λxs. zip_with (λa b. [a, b]) xs (drop 1 xs)` |
| `half_len` | `[a] → Int` | Half the length | Derived: `λxs. div (length xs) 2` |
| `first_half` | `[a] → [a]` | First half of list | Derived: `λxs. take (half_len xs) xs` |
| `second_half` | `[a] → [a]` | Second half of list | Derived: `λxs. drop (half_len xs) xs` |

### Category 5: Direct Queries (9 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `has_suit` | `Hand → Suit → Bool` | Does hand contain suit? | Derived: `λh s. any (λc. eq (get_suit c) s) h` |
| `has_color` | `Hand → Color → Bool` | Does hand contain color? | Derived: `λh c. any (λcard. eq (get_color card) c) h` |
| `count_suit` | `Hand → Suit → Int` | Count cards of suit | Derived: `λh s. length (filter (λc. eq (get_suit c) s) h)` |
| `count_color` | `Hand → Color → Int` | Count cards of color | Derived: `λh c. length (filter (λcard. eq (get_color card) c) h)` |
| `all_same_suit` | `Hand → Bool` | All cards same suit? | Derived: `λh. eq 1 (n_unique_suits h)` |
| `all_same_color` | `Hand → Bool` | All cards same color? | Derived: `λh. eq 1 (n_unique_colors h)` |
| `n_unique_suits` | `Hand → Int` | Count distinct suits | Derived: `λh. length (unique (map get_suit h))` |
| `n_unique_ranks` | `Hand → Int` | Count distinct ranks | Derived: `λh. length (unique (map get_rank h))` |
| `n_unique_colors` | `Hand → Int` | Count distinct colors | Derived: `λh. length (unique (map get_color h))` |

### Category 6: Aggregates (3 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `sum_ranks` | `Hand → Int` | Sum of rank values | Derived: requires `fold` (not in library) |
| `max_rank` | `Hand → Int` | Maximum rank value | Derived: requires `fold` |
| `min_rank` | `Hand → Int` | Minimum rank value | Derived: requires `fold` |

**Note:** These are "pseudo-fundamental"—they could be expressed with `fold`, but `fold` was intentionally removed for cognitive reasons.

### Category 7: Comparisons (5 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `eq` | `a → a → Bool` | Equality | **Yes** |
| `lt` | `Int → Int → Bool` | Less than | **Yes** |
| `le` | `Int → Int → Bool` | Less or equal | Derived: `λx y. or (lt x y) (eq x y)` |
| `gt` | `Int → Int → Bool` | Greater than | Derived: `λx y. lt y x` |
| `ge` | `Int → Int → Bool` | Greater or equal | Derived: `λx y. or (gt x y) (eq x y)` |

### Category 8: Boolean Operations (4 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `and` | `Bool → Bool → Bool` | Logical AND | **Yes** |
| `or` | `Bool → Bool → Bool` | Logical OR | **Yes** |
| `not` | `Bool → Bool` | Logical NOT | **Yes** |
| `if` | `Bool → a → a → a` | Conditional | **Yes** |

### Category 9: Higher-Order Functions (5 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `map` | `(a → b) → [a] → [b]` | Transform each element | **Yes** |
| `filter` | `(a → Bool) → [a] → [a]` | Keep matching elements | **Yes** |
| `all` | `(a → Bool) → [a] → Bool` | All satisfy predicate | Derived: `λf xs. not (any (λx. not (f x)) xs)` |
| `any` | `(a → Bool) → [a] → Bool` | Some satisfies predicate | **Yes** |
| `unique` | `[a] → [a]` | Remove duplicates | **Yes** (not efficiently expressible) |

### Category 10: Arithmetic (3 primitives)

| Primitive | Type | Description | Fundamental? |
|-----------|------|-------------|--------------|
| `+` | `Int → Int → Int` | Addition | **Yes** |
| `-` | `Int → Int → Int` | Subtraction | **Yes** |
| `mod` | `Int → Int → Int` | Modulo (remainder) | **Yes** |

---

## Primitive Decomposition Map

This section shows which primitives are truly **fundamental** (cannot be expressed using other primitives) versus **derived** (included for cognitive/search efficiency).

### Truly Fundamental Primitives (27)

These form the irreducible core of the language:

**Data Access:**
- `get_suit`, `get_rank`, `rank_val` (card projectors)
- `head`, `at`, `length` (list access)
- `take`, `drop`, `reverse` (list manipulation)

**Higher-Order:**
- `map`, `filter`, `any`, `unique`, `zip_with`

**Comparisons & Logic:**
- `eq`, `lt`, `and`, `or`, `not`, `if`

**Arithmetic:**
- `+`, `-`, `mod`

**Constants:**
- All suit, color, number, and boolean constants

### Derived but Included (32)

These could be expressed using fundamentals but are included for:
- **Cognitive realism**: Matches how humans think
- **Search efficiency**: Dramatically reduces program depth

| Derived Primitive | Decomposition | Included Because |
|-------------------|---------------|------------------|
| `get_color` | `λc. if (or (eq (get_suit c) HEARTS) ...)` | Cognitive: "what color is it?" |
| `last` | `λxs. at xs (- (length xs) 1)` | Cognitive: "the last card" |
| `first_half` | `λxs. take (div (length xs) 2) xs` | Cognitive: "first three cards" |
| `second_half` | `λxs. drop (div (length xs) 2) xs` | Cognitive: "second three cards" |
| `has_suit` | `λh s. any (λc. eq (get_suit c) s) h` | Cognitive: "do we have a spade?" |
| `all_same_suit` | `λh. eq 1 (n_unique_suits h)` | Cognitive: "is it a flush?" |
| `n_unique_suits` | `λh. length (unique (map get_suit h))` | Cognitive: "how many suits?" |
| `all` | `λf xs. not (any (λx. not (f x)) xs)` | Cognitive: "do all cards...?" |
| `le`, `gt`, `ge` | Various using `lt`, `eq`, `or` | Cognitive: natural comparisons |
| `sum_ranks` | Would need `fold` | Cognitive: "what's the total?" |

### The Fold Question

Note that `fold` (reduce) is **NOT** included, despite being the most powerful combinator. Rationale:

1. **Low cognitive reality**: People don't think "fold plus zero over the hand"
2. **Search explosion**: `fold` dramatically increases the search space
3. **Covered by aggregates**: `sum_ranks`, `max_rank`, `min_rank` cover common cases

This is a deliberate trade-off: we sacrifice expressiveness for tractability.

---

## Higher-Order Functions in Depth

### `map`: Transformation

```
map : (a → b) → [a] → [b]
```

**Semantics:** Apply a function to each element, returning transformed list.

**Examples:**
```
map get_suit [2♥, K♠, 5♦] → [HEARTS, SPADES, DIAMONDS]
map rank_val [2♥, K♠, A♦] → [2, 13, 14]
map get_color [2♥, K♠] → [RED, BLACK]
```

**Use cases:**
- Extract property lists: `map get_suit hand` → list of suits
- Transform for comparison: `map rank_val hand` for numeric operations

### `filter`: Selection

```
filter : (a → Bool) → [a] → [a]
```

**Semantics:** Keep only elements that satisfy the predicate.

**Examples:**
```
filter (λc. eq (get_suit c) HEARTS) [2♥, K♠, 5♥] → [2♥, 5♥]
filter (λc. gt (rank_val c) 10) [2♥, K♠, 5♦] → [K♠]
```

**Use cases:**
- Select subset: "only the hearts"
- Count via composition: `length (filter pred hand)`

### `all`: Universal Quantification

```
all : (a → Bool) → [a] → Bool
```

**Semantics:** Returns `true` iff every element satisfies the predicate.

**Examples:**
```
all (λc. eq (get_color c) RED) [2♥, 5♦, K♥] → true
all (λc. gt (rank_val c) 5) [2♥, K♠, 5♦] → false (2 fails)
```

**Use cases:**
- Global properties: "all cards are red"
- Constraints: "every card is above 5"

### `any`: Existential Quantification

```
any : (a → Bool) → [a] → Bool
```

**Semantics:** Returns `true` iff at least one element satisfies the predicate.

**Examples:**
```
any (λc. eq (get_suit c) SPADES) [2♥, K♠, 5♦] → true
any (λc. gt (rank_val c) 14) [2♥, K♠, 5♦] → false
```

**Use cases:**
- Existence: "is there an ace?"
- At least one: "does the hand have a heart?"

### `unique`: Deduplication

```
unique : [a] → [a]
```

**Semantics:** Remove duplicates, preserving first occurrence order.

**Examples:**
```
unique [HEARTS, SPADES, HEARTS, DIAMONDS] → [HEARTS, SPADES, DIAMONDS]
unique (map get_suit hand) → distinct suits in order of appearance
```

**Use cases:**
- Count distinct: `length (unique (map get_suit hand))`
- Set-like operations without explicit sets

### `zip_with`: Parallel Combination

```
zip_with : (a → b → c) → [a] → [b] → [c]
```

**Semantics:** Apply binary function to corresponding pairs from two lists.

**Examples:**
```
zip_with eq [A, B, C] [A, X, C] → [true, false, true]
zip_with eq (map get_suit hand) (reverse (map get_suit hand)) → palindrome check
```

**Use cases:**
- Palindrome detection: compare forward and reversed
- Halves comparison: compare first_half with second_half
- Sorted check: compare adjacent pairs

---

## Parallel Primitives (Experimental)

These are **not** in the base library but are used in experiments to test whether structural patterns are the main barrier to rule discovery.

### `halves_equal_by`

```
halves_equal_by : (Hand → a) → Hand → Bool
```

**Semantics:** Check if a function gives the same result on both halves.

**Equivalent to:** `λf h. eq (f (first_half h)) (f (second_half h))`

**Example:**
```
halves_equal_by all_same_color [R, R, R, B, B, B]
= eq (all_same_color [R, R, R]) (all_same_color [B, B, B])
= eq true true
= true
```

### `ends_equal_by`

```
ends_equal_by : (Card → a) → Hand → Bool
```

**Semantics:** Check if a function gives the same result on first and last card.

**Equivalent to:** `λf h. eq (f (head h)) (f (last h))`

**Example:**
```
ends_equal_by get_suit [2♥, K♠, 5♦, 3♥]
= eq (get_suit 2♥) (get_suit 3♥)
= eq HEARTS HEARTS
= true
```

### `is_palindrome_by`

```
is_palindrome_by : (Card → a) → Hand → Bool
```

**Semantics:** Check if transformed list equals its reverse.

**Equivalent to:** `λf h. eq (map f h) (reverse (map f h))`

**Example:**
```
is_palindrome_by get_color [R, B, B, R]
= eq [R, B, B, R] [R, B, B, R]
= true
```

### `all_adjacent_satisfy`

```
all_adjacent_satisfy : (Card → Card → Bool) → Hand → Bool
```

**Semantics:** Check if every consecutive pair satisfies a relation.

**Equivalent to:** `λp h. all (λpair. p (head pair) (last pair)) (adjacent_pairs h)`

**Example:**
```
all_adjacent_satisfy (λa b. le (rank_val a) (rank_val b)) [2♥, 5♠, K♦]
= checks: 2≤5 ∧ 5≤13
= true (sorted ascending)
```

### Experimental Results

With parallel primitives, task solve rate increased from **8/45** to **11/45** (37.5% improvement). New solutions found:
- `Halves_uniform_color_equal`: `halves_equal_by all_same_color`
- `Ends_same_suit`: `ends_equal_by get_suit`
- `Suits_palindrome`: `is_palindrome_by get_suit`

---

## Automata-Theoretic Expressiveness

### What the DSL Can Express

The DSL is equivalent to a restricted form of **counter automata** and can express:

1. **Regular Patterns**
   - "All hearts" → finite automaton over suit alphabet
   - "Alternating colors" → 2-state automaton

2. **Bounded Counting**
   - "Exactly 3 hearts" → counter bounded by hand length
   - "At most 2 suits" → bounded uniqueness check

3. **Linear Scans with Memory**
   - `all_same_suit` → single pass with one memory cell (first suit)
   - `is_sorted` → single pass comparing adjacent elements

4. **Parallel Structure Comparison**
   - Compare halves: bounded by fixed hand size
   - Palindrome check: requires reversing (O(n) space)

### What the DSL Cannot Express

The DSL **cannot** express:

1. **Pushdown Patterns (Stack Operations)**
   - Balanced parentheses (e.g., bracket-by-suit rules)
   - Nested structures that require unbounded stack
   - This is fundamental: no primitive provides stack semantics

2. **Turing-Complete Computation**
   - Arbitrary recursion
   - Unbounded iteration
   - General while-loops

3. **Cross-Element Dependencies Beyond Adjacent**
   - "Card i relates to card i+k for variable k"
   - Would require explicit recursion or fold-like operations

### The Bracket Rule Problem

The rule `Well_formed_brackets_by_suit` requires matching opening/closing brackets:
- HEARTS/DIAMONDS = open
- SPADES/CLUBS = close

This is a **context-free language** (pushdown automaton), which is fundamentally beyond our DSL's expressiveness. No amount of primitive addition short of adding explicit stack operations would solve this.

### Implication for Search

The DSL's limited expressiveness is intentional:
- Keeps the search space tractable
- Matches human cognitive constraints (limited working memory)
- Rules beyond counter automata may be cognitively unrealistic

---

## Design Decisions and Rationale

### Decision 1: Remove Abstract Combinators

**What was removed:** `compose`, `flip`, `const`, `id`, `fst`, `snd`

**Rationale:**
- Low cognitive reality: "compose get_suit with head" vs "the suit of the first card"
- Expand search space without proportional benefit
- Can be recovered via lambda expressions when needed

### Decision 2: Include "Compiled" Aggregates

**What was added:** `all_same_suit`, `n_unique_ranks`, `sum_ranks`, etc.

**Rationale:**
- These match natural language chunks ("is it a flush?")
- Dramatically reduce program depth (depth 2 vs depth 6)
- Depth reduction of 3-4 means 100-1000x faster search

### Decision 3: Remove Fold

**What was removed:** `fold` / `reduce` / `foldl` / `foldr`

**Rationale:**
- Extremely powerful but cognitively opaque
- Massively expands search space
- Common use cases covered by specific aggregates

### Decision 4: Minimal Numeric Constants

**What was removed:** Rank thresholds (10-14), game scores (17, 21)

**Rationale:**
- Rules should use relative comparisons
- "Higher than the first card" not "higher than 10"
- Reduces grammar size and improves generalization

### Decision 5: Projector Primitives

**What was added:** `get_suit`, `get_rank`, `get_color` as first-class primitives

**Rationale:**
- Essential for any card property extraction
- Enable composition with higher-order functions
- Without projectors, cannot express "map get_suit over hand"

### Decision 6: Half-Splitting Primitives

**What was added:** `first_half`, `second_half`, `half_len`

**Rationale:**
- Many rules compare halves of the hand
- More cognitively real than `take (div (length h) 2) h`
- Direct primitive reduces search depth

---

## Evolution History

### Version 1 (Initial)
- Abstract combinator style following original DreamCoder
- Included `compose`, `flip`, `const`, `id`
- ~60 primitives with mathematical completeness focus

### Version 2 (Cognitive Refocus)
- Removed abstract combinators
- Added domain-specific queries (`has_suit`, `count_suit`)
- Added gestalt primitives (`all_same_suit`, `all_same_color`)
- Added aggregates (`sum_ranks`, `max_rank`, `min_rank`)
- Removed rank thresholds (10-14) and game scores (17, 21)
- ~55 primitives

### Version 3 (Current)
- Removed `neq` (use `not (eq x y)` instead)
- Added list slicing (`take`, `drop`, `first_half`, `second_half`)
- Added `zip_with` for parallel comparison
- Added `adjacent_pairs` for sorted checks
- **59 base primitives**

### Version 3.1 (Experimental Extensions)
- Added 4 parallel primitives for structural pattern experiments
- `halves_equal_by`, `ends_equal_by`, `is_palindrome_by`, `all_adjacent_satisfy`
- These are not in base library, used only for specific experiments

---

## Appendix: Quick Reference Card

### Most Used Primitives by Task Type

| Task Type | Key Primitives |
|-----------|----------------|
| Counting | `count_suit`, `count_color`, `n_unique_*`, `length` |
| Membership | `has_suit`, `has_color`, `any`, `all` |
| Comparison | `eq`, `lt`, `le`, `gt`, `ge` |
| Position | `head`, `last`, `first_half`, `second_half` |
| Transformation | `map`, `filter`, `unique` |
| Aggregation | `sum_ranks`, `max_rank`, `min_rank` |

### Cognitive-to-Primitive Mapping

| Human Phrase | Primitive |
|--------------|-----------|
| "Is it a flush?" | `all_same_suit hand` |
| "How many hearts?" | `count_suit hand HEARTS` |
| "Is there an ace?" | `any (λc. eq 14 (rank_val c)) hand` |
| "Are the ends the same suit?" | `eq (get_suit (head h)) (get_suit (last h))` |
| "Is it sorted by rank?" | `all (λp. le (rank_val (head p)) (rank_val (last p))) (adjacent_pairs h)` |

---

*Document maintained as part of the card-games-modelling project. For implementation details, see `src/dreamcoder_core/primitives.py`.*
