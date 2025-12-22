# Card Game Rules Catalogue

This document describes the 56 experimental rules organized by family.
Each rule is a predicate `Hand → bool` that can be expressed compositionally
using the DSL primitives.

## Rule Families Overview

| Family | Count | Description | Key Features |
|--------|-------|-------------|--------------|
| LOCAL | 4 | Positional/ordering | Position + property |
| COUNT | 6 | Cardinality | Suit/color counting |
| POSITION | 2 | Specific position checks | Fixed positions |
| TOKEN | 2 | Specific card presence | Card identity |
| AP | 3 | Arithmetic progressions | Rank arithmetic |
| SCORE | 3 | Scoring formulas | Aggregation |
| HIER | 6 | Hierarchical predicates | Boolean property per half |
| LANG | 3 | Bracket/grammar matching | Sequential patterns |
| PAL | 3 | Palindrome patterns | Symmetry |
| ALTCLR | 3 | Alternative colorings | Alternative groupings |
| COPY | 6 | Halves copy sequence | Halves comparison |
| SHIFT | 3 | Positional rank shifts | Positional arithmetic |
| MAP | 6 | Suit cycle mappings | Transform + compare |
| ADJ | 3 | Adjacent constraints | Local relationships |
| PARITY | 2 | Odd/even patterns | Rank parity |
| CENTER | 2 | Distance from center | Radial patterns |

---

## Family: LOCAL (Positional/Ordering)

### r1x: Sorted_by_rank
- **Description**: Ranks are in non-decreasing order left-to-right
- **Example Positive**: [3♠, 5♥, 7♣, 9♦, J♠, K♥] (3,5,7,9,11,13)
- **Example Negative**: [K♠, 5♥, 3♣, 9♦, J♠, 7♥] (13,5,3,9,11,7)
- **Primitives**: is_sorted, map, rank_val

### r4x: S_before_H
- **Description**: Some ♠ appears before some ♥
- **Example Positive**: [♠K, ♣7, ♥2, ♦Q, ♠5, ♥9]
- **Example Negative**: [♥K, ♣7, ♣2, ♦Q, ♠5, ♠9]
- **Primitives**: exists_ordered, get_suit, eq

### r44x: Ends_same_suit
- **Description**: First and last cards share the same suit
- **Example Positive**: [♠K, ♣7, ♥2, ♦Q, ♣5, ♠9] (both ♠)
- **Example Negative**: [♠K, ♣7, ♥2, ♦Q, ♣5, ♥9] (♠ ≠ ♥)
- **Primitives**: terminals_equal, get_suit, head, last, eq

### r45x: Ends_same_color
- **Description**: First and last cards share the same color
- **Example Positive**: [♠K, ♣7, ♥2, ♦Q, ♣5, ♣9] (black, black)
- **Example Negative**: [♠K, ♣7, ♥2, ♦Q, ♣5, ♥9] (black, red)
- **Primitives**: terminals_equal, get_color, head, last, eq

---

## Family: COUNT (Cardinality)

### r2x: Has_pair_ranks
- **Description**: At least two cards share the same rank
- **Example Positive**: [K♠, 7♣, K♥, 2♦, Q♣, 9♠] (two Kings)
- **Example Negative**: [K♠, 7♣, 8♥, 2♦, Q♣, 9♠] (all different)
- **Primitives**: unique_count, get_rank, length, lt

### r3x: Uniform_color
- **Description**: All cards have the same color (all red or all black)
- **Example Positive**: [♠K, ♣7, ♠2, ♣Q, ♠5, ♣9] (all black)
- **Example Negative**: [♠K, ♣7, ♥2, ♣Q, ♠5, ♣9] (mixed)
- **Primitives**: uniform, get_color, unique_count, eq

### p7x: Exactly_two_suits
- **Description**: Exactly two distinct suits appear
- **Example Positive**: [♠K, ♥7, ♠2, ♥Q, ♠5, ♥9] (♠ and ♥ only)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♠5, ♥9] (four suits)
- **Primitives**: unique_count, get_suit, eq, 2

### r11x: Half_or_more_same_suit
- **Description**: At least half the cards share one suit
- **Example Positive**: [♠K, ♠7, ♠2, ♦Q, ♠5, ♥9] (4/6 = ♠)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♠5, ♥9] (max 2 of any suit)
- **Primitives**: max_count, get_suit, length, div, gte

### r43x: Exactly_one_club
- **Description**: The hand contains exactly one ♣
- **Example Positive**: [♠K, ♣7, ♠2, ♦Q, ♠5, ♥9] (one ♣)
- **Example Negative**: [♠K, ♣7, ♣2, ♦Q, ♠5, ♥9] (two ♣)
- **Primitives**: count_equal, get_suit, eq, CLUBS, 1

### r55x: At_most_three_suits
- **Description**: At most three distinct suits appear
- **Example Positive**: [♠K, ♥7, ♠2, ♥Q, ♣5, ♥9] (♠, ♥, ♣)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♠5, ♥9] (all four)
- **Primitives**: unique_count, get_suit, lte, 3

---

## Family: PAL (Palindrome)

### r18x: Suits_palindrome
- **Description**: Suit sequence reads same forward and backward
- **Example Positive**: [♠K, ♥7, ♣2, ♣Q, ♥5, ♠9] (♠♥♣♣♥♠)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♥5, ♠9] (♠♥♣♦♥♠)
- **Primitives**: seq_palindrome, map, get_suit, reverse, eq

### r19x: Colors_palindrome
- **Description**: Color sequence (R/B) reads same forward and backward
- **Example Positive**: [♠K, ♥7, ♥2, ♥Q, ♥5, ♠9] (BRRRRRB)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♥5, ♠9] (BRBRRB)
- **Primitives**: seq_palindrome, map, get_color, reverse, eq

### r20x: Ranks_palindrome
- **Description**: Rank sequence reads same forward and backward
- **Example Positive**: [K♠, 7♥, 2♣, 2♦, 7♥, K♠] (K,7,2,2,7,K)
- **Example Negative**: [K♠, 7♥, 2♣, 3♦, 7♥, K♠] (K,7,2,3,7,K)
- **Primitives**: seq_palindrome, map, get_rank, reverse, eq

---

## Family: COPY (Halves Copy Sequence)

### r21x: Halves_copy_suits
- **Description**: Left half suits match right half suits (in order)
- **Example Positive**: [♠K, ♥7, ♣2, ♠Q, ♥5, ♣9] (♠♥♣ = ♠♥♣)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♥5, ♣9] (♠♥♣ ≠ ♦♥♣)
- **Primitives**: halves_equal, first_half, second_half, map, get_suit, eq

### r22x: Halves_copy_colors
- **Description**: Left half colors match right half colors
- **Example Positive**: [♠K, ♥7, ♠2, ♣Q, ♦5, ♣9] (BRB = BRB)
- **Example Negative**: [♠K, ♥7, ♠2, ♣Q, ♥5, ♣9] (BRB ≠ BRB)
- **Primitives**: halves_equal, first_half, second_half, map, get_color, eq

### r23x: Halves_copy_ranks
- **Description**: Left half ranks match right half ranks
- **Example Positive**: [K♠, 7♥, 2♣, K♦, 7♥, 2♠] (K,7,2 = K,7,2)
- **Example Negative**: [K♠, 7♥, 2♣, Q♦, 7♥, 2♠] (K,7,2 ≠ Q,7,2)
- **Primitives**: halves_equal, first_half, second_half, map, get_rank, eq

---

## Family: SHIFT (Positional Rank Differences)

### r24x: Shift_half_plus_two
- **Description**: Each right-half card is exactly +2 rank from corresponding left-half card
- **Example Positive**: [3♠, 7♥, J♣, 5♦, 9♥, K♠] (3→5, 7→9, J→K)
- **Example Negative**: [3♠, 7♥, J♣, 6♦, 9♥, K♠] (3→6 is +3, not +2)
- **Primitives**: shifted_pairs, all, get_rank_val, diff, eq, 2

### r41x: Shift_half_ge
- **Description**: Each right-half rank ≥ corresponding left-half rank
- **Example Positive**: [3♠, 7♥, J♣, 5♦, 8♥, K♠] (3≤5, 7≤8, J≤K)
- **Example Negative**: [3♠, 7♥, J♣, 2♦, 8♥, K♠] (3>2)
- **Primitives**: shifted_pairs, all, get_rank_val, gte

---

## Family: ADJ (Adjacent Constraints)

### r38x: Adj_same_rank_or_suit
- **Description**: Every adjacent pair shares rank or suit
- **Example Positive**: [♠K, ♠7, 7♥, ♥2, 2♣, ♣9] (suit, rank, suit, rank, suit)
- **Example Negative**: [♠K, ♥7, ♣2, ♦Q, ♠5, ♣9] (no adjacents share)
- **Primitives**: adjacent_pairs, all, or, eq, get_rank, get_suit

### r40x: Adj_rank_gap_le3
- **Description**: Adjacent cards differ by at most 3 in rank
- **Example Positive**: [3♠, 5♥, 7♣, 9♦, J♠, K♥] (diffs: 2,2,2,2,2)
- **Example Negative**: [3♠, 5♥, 7♣, J♦, K♠, A♥] (7→J is gap of 4)
- **Primitives**: adjacent_pairs, all, abs, diff, get_rank_val, lte, 3

---

## Family: HIER (Hierarchical Boolean Properties)

### r12x: Halves_uniform_color_equal
- **Description**: Both halves are uniform in color, OR both are mixed
- **Example Positive**: [♠K, ♠7, ♠2, ♥Q, ♥5, ♥9] (left all black, right all red - both uniform)
- **Example Negative**: [♠K, ♥7, ♠2, ♥Q, ♥5, ♥9] (left mixed, right uniform)
- **Primitives**: halves, uniform, get_color, eq

### r26x: Halves_hearts_presence_equal
- **Description**: Both halves have a ♥ OR neither does
- **Example Positive**: [♠K, ♥7, ♣2, ♦Q, ♥5, ♠9] (both have ♥)
- **Example Negative**: [♠K, ♣7, ♣2, ♦Q, ♥5, ♠9] (only right has ♥)
- **Primitives**: halves, any, eq, get_suit, HEARTS

---

## Feature Relevance by Family

This table shows which card features are relevant for each family:

| Family | Position | Suit | Rank | Color |
|--------|----------|------|------|-------|
| LOCAL | ✓ | ✓ | ✓ | ✓ |
| COUNT | - | ✓ | ✓ | ✓ |
| PAL | ✓ | ✓ | ✓ | ✓ |
| COPY | ✓ | ✓ | ✓ | ✓ |
| SHIFT | ✓ | - | ✓ | - |
| ADJ | ✓ | ✓ | ✓ | - |
| PARITY | - | - | ✓ | - |
| AP | - | - | ✓ | - |
| HIER | ✓ | ✓ | - | ✓ |
| CENTER | ✓ | - | ✓ | - |

**Key insight**: The recognition model must learn to attend to different features
for different rule families. This is what makes the problem challenging.
