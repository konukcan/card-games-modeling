# Recognition Model Problem Statement for DreamCoder Card Classification

## System Overview

This is a DreamCoder-inspired program synthesis system for learning compositional rules in a card game domain. The system has three main phases:

1. **Wake (Enumeration)**: Best-first search through program space guided by grammar probabilities
2. **Sleep (Recognition)**: Train neural network to predict which primitives are useful for a task
3. **Sleep (Compression)**: Extract common subprograms as reusable abstractions

The enumeration and compression phases work well. **The bottleneck is the recognition model.**

## Task Structure

- **Input**: Ordered list of 6 playing cards (each with suit, rank, position)
- **Output**: Boolean (True if hand satisfies rule, False otherwise)
- **Training signal**: ~20 labeled examples per task, 10 positive, 10 negative
- **Goal**: Given examples, predict which grammar productions (primitives) are likely useful

## The Core Problem

The recognition model produces **nearly identical embeddings and predictions for all tasks**. It fails to discriminate between different rules.

### Evidence of Failure

| Metric | Observed Value | Expected |
|--------|---------------|----------|
| Entropy std across tasks | 0.0005 - 0.005 | Much higher |
| Max prob difference between tasks | ~0.01-0.03 | Significant |
| Top predicted primitive | Same for all tasks | Should vary |
| Cosine similarity of task embeddings | 0.95-0.99 | Much lower |

### Root Cause Analysis

**The fundamental issue**: In standard DreamCoder (list transformation tasks), the output is informative - e.g., `[1,2,3] → [6]` encodes "this is a sum". The GRU learns which outputs indicate which primitives.

In our **classification task**, the output is just True/False (2 bits). The network sees:
- Hand A → True
- Hand B → False

But it has no direct way to learn *why* Hand A is True and Hand B is False from the labels alone.

### Why Standard Architectures Fail

1. **Weak output signal**: True/False provides minimal information about which primitives to use
2. **Cross-task confusion**: Negative examples across different tasks look similar (random hands)
3. **Feature space is huge**: Cards have suit × rank × position, but for any given rule, most features are irrelevant
4. **Different rules attend to different features**:
   - "Halves have same suits" → position + suit (ignore rank)
   - "Is sorted" → position + rank (ignore suit)
   - "At least 3 hearts" → suit only (ignore position)

## DSL Structure

### Primitive Categories (~60 total)

1. **Card Accessors**: `get_suit`, `get_rank`, `rank_val`, `get_color`
2. **Positional**: `head`, `last`, `at`, `first_half`, `second_half`
3. **List Operations**: `map`, `filter`, `length`, `unique`, `reverse`
4. **Quantifiers**: `all`, `any`, `count_suit`, `count_color`
5. **Predicates**:  `all_same_suit`, `all_same_color`, `n_unique_suits`
6. **Comparisons**: `eq`, `lt`, `gt`, `le`, `ge`
7. **Combinators**: `zip_with`, `adjacent_pairs`, `take`, `drop`
8. **Aggregates**: `sum_ranks`, `max_rank`, `min_rank`

### Type System

- Base types: `bool`, `int`, `card`, `suit`, `rank`, `color`
- Composite types: `hand = list(card)`, `list(int)`, `list(bool)`
- Arrow types: e.g., `card -> suit`, `hand -> bool`

## Rule Diversity (56 Rules Across 15 Families)

| Family | Count | Example | Key Features |
|--------|-------|---------|--------------|
| LOCAL | 4 | "First and last share suit" | Position + property |
| COUNT | 6 | "At least 3 hearts" | Suit/color counting |
| PAL | 5 | "Suits form palindrome" | Position + property + symmetry |
| COPY | 6 | "Left half suits = right half suits" | Position + comparison |
| SHIFT | 3 | "Right half ranks ≥ left half" | Position + rank |
| ADJ | 3 | "Adjacent cards share suit or rank" | Local structure |
| PARITY | 2 | "All ranks same parity" | Rank arithmetic |
| AP | 3 | "Ranks form arithmetic progression" | Rank arithmetic |
| HIER | 6 | "Both halves either have ♥ or neither" | Hierarchical predicates |
| LANG | 3 | "Suits form matched brackets" | Sequential pattern |
| MAP | 6 | "Right half = suit-cycle(left half)" | Transform + compare |
| ... | ... | ... | ... |

## What the Recognition Model Must Learn

For each rule, the model must identify which features distinguish positive from negative examples:

### Example: "Halves_copy_suits"
- **Positive**: `[♠K, ♥7, ♣2, | ♠Q, ♥5, ♣9]` (suits match: ♠♥♣ = ♠♥♣)
- **Negative**: `[♠K, ♥7, ♣2, | ♦Q, ♥5, ♣9]` (suits differ: ♠♥♣ ≠ ♦♥♣)
- **Relevant primitives**: `first_half`, `second_half`, `map`, `get_suit`, `eq`
- **Key insight**: Must compare POSITION and SUIT, ignore RANK

### Example: "Sorted_by_rank"
- **Positive**: `[3♠, 5♥, 7♣, 9♦, J♠, K♥]` (ranks: 3,5,7,9,11,13 - increasing)
- **Negative**: `[K♠, 5♥, 3♣, 9♦, J♠, 7♥]` (ranks: 13,5,3,9,11,7 - not sorted)
- **Relevant primitives**: `is_sorted`, `map`, `rank_val`, `adjacent_pairs`, `le`
- **Key insight**: Must compare RANK across POSITIONS, ignore SUIT

## What Would Success Look Like?

A good recognition model should:

1. **Produce different embeddings** for different tasks (cosine similarity < 0.5)
2. **Predict different top primitives** for different task families
3. **Show meaningful attention** to relevant card features
4. **Improve enumeration speed** by 2-10x vs uniform grammar

## Architecture Constraints

1. **Must be neuro-symbolic**: Neural recognition + symbolic enumeration
2. **Cognitive plausibility valued**: Should exhibit human-like sample efficiency
3. **Must interface with existing infrastructure**: Output is log-probabilities over primitives
4. **Training data**: Only solved tasks provide training signal (primitives used in solution)

## Key Research Questions

1. **How should the model encode labeled examples for classification?**
   - Set-based encoding of positive vs negative examples?
   - Contrastive encoding between positive and negative sets?
   - Attention over feature dimensions to learn relevance?

2. **How should individual cards be encoded?**
   - Flat one-hot (suit × rank × position)?
   - Factored embedding (separate embeddings combined)?
   - With explicit feature extractors matching DSL primitives?

3. **How should the model produce primitive distributions?**
   - Direct multi-label classification?
   - Program embedding that biases enumeration?
   - Hierarchical prediction matching grammar structure?

4. What is the relevant library of primitives for this task?
   - It shouldn't integrate ready-made patterns like is_sorted, or palindrome, or arithmetic progression, etc. But ideally find a way to discover them from the tasks.
   - It is ok however to assume thereotically that the model could in principle discover these from a larger dataset of tasks that spans beyond the present one.
   - It would actually be ideal if tasks that involve relations between ranks could be implemented in a way that collapse rank value and rank index, since intuitively it seems like the information they contain is largely redundant. Here I only use rank index because it makes it easier to raison about rules that involve relations between ranks insensitive to how valuable, they are, such as pairs, triples, etc. (as well as more complex ones).

## Failed Approaches

1. **BiGRU over card features + output label**: All tasks collapse to same embedding
2. **Attention-weighted pooling**: Slight improvement, still 0.95+ similarity
3. **Set Transformer**: Better card interactions, still weak task discrimination
4. **FiLM conditioning on label**: Label signal too weak to modulate meaningfully



