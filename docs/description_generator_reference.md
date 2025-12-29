# Description Generator Reference Document

**Date**: December 26, 2024
**Purpose**: Comprehensive reference for the description generator evaluation system

---

## Executive Summary

This document provides a complete reference of:
- **201 synthetic card game rules** organized by complexity level
- **Feature vocabulary**: 29 feature types the system can detect
- **Text-to-feature mappings**: 30 keywords that map to semantic features
- **Category-to-feature mappings**: 43 rule categories with acceptable feature types

---

## Part 1: Feature Vocabulary

The description generator detects these semantic features from card hands:

### Card Composition Features

| Feature | Description | Example Detection |
|---------|-------------|-------------------|
| `is_flush` | All cards share the same suit | "all hearts" |
| `is_uniform_color` | All cards share the same color | "all red cards" |
| `unique_suits` | Count of distinct suits (1-4) | "3 different suits" |
| `unique_colors` | Count of distinct colors (1-2) | "both colors present" |
| `unique_ranks` | Count of distinct ranks (1-6) | "4 different ranks" |
| `has_pair` | At least two cards share a rank | "pair of 7s" |
| `has_triple` | At least three cards share a rank | "three queens" |

### Positional Features

| Feature | Description | Example Detection |
|---------|-------------|-------------------|
| `first_card` | Properties of position 0 | "starts with spade" |
| `last_card` | Properties of final position | "ends with red" |
| `ends_same_suit` | First and last share suit | "bookends same suit" |
| `ends_same_color` | First and last share color | "red at both ends" |
| `first_half` | Properties of positions 0-2 | "left side..." |
| `second_half` | Properties of positions 3-5 | "right side..." |
| `halves_same_color` | Both halves have same color set | "halves match color" |
| `halves_same_suits` | Both halves have same suit set | "halves match suits" |

### Ordering Features

| Feature | Description | Example Detection |
|---------|-------------|-------------------|
| `is_sorted` | Ranks in non-decreasing order | "increasing ranks" |
| `is_palindrome_suits` | Suit sequence reads same both ways | "suit palindrome" |
| `is_palindrome_colors` | Color sequence reads same both ways | "color palindrome" |

### Aggregate Features

| Feature | Description | Example Detection |
|---------|-------------|-------------------|
| `sum_ranks` | Total of all rank values | "sum is 24" |
| `rank_spread` | Max rank - min rank | "spread of 8" |
| `max_rank` | Highest rank value | "contains king" |
| `min_rank` | Lowest rank value | "contains ace" |
| `suit_count` | Count of a specific suit | "3 hearts" |
| `color_count` | Count of a specific color | "4 red cards" |
| `rank_count` | Count of a specific rank | "2 sevens" |

### Meta Features

| Feature | Description |
|---------|-------------|
| `distinguishing` | General distinguishing characteristic |

---

## Part 2: Text-to-Feature Mappings

When descriptions contain these keywords, they map to the corresponding features:

| Text Keyword | Maps To Features |
|--------------|------------------|
| "same suit" | `is_flush`, `unique_suits` |
| "same color" | `is_uniform_color`, `unique_colors` |
| "pair" | `has_pair` |
| "triple" | `has_triple` |
| "three of a kind" | `has_triple` |
| "sorted" | `is_sorted` |
| "increasing order" | `is_sorted` |
| "decreasing" | `is_sorted` |
| "first card" | `first_card` |
| "last card" | `last_card` |
| "starts with" | `first_card` |
| "ends with" | `last_card` |
| "first and last" | `ends_same_color`, `ends_same_suit` |
| "sum" | `sum_ranks` |
| "rank sum" | `sum_ranks` |
| "total" | `sum_ranks` |
| "spread" | `rank_spread` |
| "unique" | `unique_colors`, `unique_ranks`, `unique_suits` |
| "different" | `unique_colors`, `unique_ranks`, `unique_suits` |
| "palindrome" | `is_palindrome_colors`, `is_palindrome_suits` |
| "symmetric" | `is_palindrome_colors`, `is_palindrome_suits` |
| "halves" | `halves_same_color` |
| "hearts" | `first_card`, `last_card`, `suit_count` |
| "diamonds" | `first_card`, `last_card`, `suit_count` |
| "clubs" | `first_card`, `last_card`, `suit_count` |
| "spades" | `first_card`, `last_card`, `suit_count` |
| "red" | `color_count`, `is_uniform_color` |
| "black" | `color_count`, `is_uniform_color` |
| "face card" | `max_rank`, `min_rank` |
| "ace" | `first_card`, `last_card`, `max_rank` |

---

## Part 3: Category-to-Feature Mappings (Semantic Correctness)

For evaluation, each rule category maps to **acceptable features**. A description is "semantically correct" if it mentions ANY feature from the acceptable set.

### Simple Categories (Specific Feature Sets)

| Category | Acceptable Features |
|----------|---------------------|
| `uniform` | `is_flush`, `is_uniform_color`, `unique_suits`, `unique_colors`, `unique_ranks` |
| `position_first` | `first_card` |
| `position_last` | `last_card` |
| `position_second` | `first_card` |
| `position_middle` | `first_card`, `last_card` |
| `terminals` | `first_card`, `last_card`, `ends_same_color`, `ends_same_suit` |
| `terminals_rank` | `first_card`, `last_card`, `rank_spread` |
| `sum` | `sum_ranks` |
| `sum_divisibility` | `sum_ranks` |
| `has` | `has_pair`, `has_triple`, `unique_ranks` |
| `no` | `has_pair`, `has_triple`, `unique_ranks` |
| `duplicates` | `has_pair`, `has_triple`, `unique_ranks` |
| `sorted` | `is_sorted` |
| `monotonic` | `is_sorted`, `rank_spread` |
| `alternating` | `is_uniform_color`, `unique_colors` |
| `palindrome` | `is_palindrome_colors`, `is_palindrome_suits` |
| `unique_count` | `unique_suits`, `unique_colors`, `unique_ranks` |
| `unique_bound` | `unique_suits`, `unique_colors`, `unique_ranks` |
| `count_specific` | `suit_count`, `color_count`, `rank_count` |
| `count_compare` | `suit_count`, `color_count` |
| `count_parity` | `suit_count`, `color_count` |
| `majority` | `suit_count`, `color_count`, `is_uniform_color` |
| `range` | `max_rank`, `min_rank`, `rank_spread` |
| `spread` | `max_rank`, `min_rank`, `rank_spread` |
| `adjacent` | `is_sorted`, `rank_spread` |
| `adjacent_constraint` | `is_sorted`, `unique_suits`, `unique_colors` |
| `runs` | `is_sorted`, `rank_spread` |
| `skip` | `is_sorted`, `rank_spread` |
| `position_pair` | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| `shape` | `has_pair`, `has_triple`, `unique_ranks` |
| `halves_property` | `is_uniform_color`, `halves_same_color` |
| `halves_set` | `unique_suits`, `halves_same_color` |
| `halves_sum` | `sum_ranks` |
| `halves_copy` | `is_palindrome_suits`, `halves_same_color` |
| `periodic` | `unique_suits`, `unique_colors` |

### Lenient Categories (Accept ANY Feature)

These compositional categories are lenient because they may have multiple valid descriptions:

- `and_combination`, `or_combination`, `conditional`, `complex`, `xor`, `triple`, `negation`, `biconditional`

All accept the full feature set.

---

## Part 4: Complete Rule Catalogue (201 Rules)

### Level 1: Atomic Rules (50 rules)

Simple rules checking a single property.

#### Uniform Property Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_001 | All Same Suit | All cards have the same suit | uniform | `is_flush`, `unique_suits`, `unique_colors`, `unique_ranks`, `is_uniform_color` |
| atomic_002 | All Same Color | All cards have the same color (red or black) | uniform | `is_flush`, `unique_suits`, `unique_colors`, `unique_ranks`, `is_uniform_color` |
| atomic_003 | All Same Parity | All cards have the same rank parity (all odd or all even) | uniform | `is_flush`, `unique_suits`, `unique_colors`, `unique_ranks`, `is_uniform_color` |
| atomic_004 | All Same Pointy/Round | All cards are pointy (spades/diamonds) or all are round (hearts/clubs) | uniform | `is_flush`, `unique_suits`, `unique_colors`, `unique_ranks`, `is_uniform_color` |
| atomic_005 | All Same SH/DC | All cards are SH (spades/hearts) or all are DC (diamonds/clubs) | uniform | `is_flush`, `unique_suits`, `unique_colors`, `unique_ranks`, `is_uniform_color` |

#### First Position Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_006 | First Card is Clubs | First card is a clubs | position_first | `first_card` |
| atomic_007 | First Card is Diamonds | First card is a diamonds | position_first | `first_card` |
| atomic_008 | First Card is Hearts | First card is a hearts | position_first | `first_card` |
| atomic_009 | First Card is Spades | First card is a spades | position_first | `first_card` |
| atomic_010 | First Card is Red | First card is red | position_first | `first_card` |
| atomic_011 | First Card is Black | First card is black | position_first | `first_card` |
| atomic_012 | First Card Rank is Odd | First card has odd rank | position_first | `first_card` |
| atomic_013 | First Card Rank is Even | First card has even rank | position_first | `first_card` |

#### Last Position Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_014 | Last Card is Clubs | Last card is a clubs | position_last | `last_card` |
| atomic_015 | Last Card is Diamonds | Last card is a diamonds | position_last | `last_card` |
| atomic_016 | Last Card is Hearts | Last card is a hearts | position_last | `last_card` |
| atomic_017 | Last Card is Spades | Last card is a spades | position_last | `last_card` |
| atomic_018 | Last Card is Red | Last card is red | position_last | `last_card` |
| atomic_019 | Last Card is Black | Last card is black | position_last | `last_card` |

#### Sum Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_020 | Sum of Ranks is Even | Sum of all rank values is even | sum | `sum_ranks` |
| atomic_021 | Sum of Ranks is Odd | Sum of all rank values is odd | sum | `sum_ranks` |
| atomic_044 | Sum of Ranks Divisible by 3 | Sum divisible by 3 | sum_divisibility | `sum_ranks` |
| atomic_045 | Sum of Ranks Divisible by 4 | Sum divisible by 4 | sum_divisibility | `sum_ranks` |
| atomic_046 | Sum of Ranks Divisible by 5 | Sum divisible by 5 | sum_divisibility | `sum_ranks` |

#### Has/No Suit Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_022 | Has at Least One Clubs | Hand contains at least one clubs | has | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_023 | Has at Least One Diamonds | Hand contains at least one diamonds | has | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_024 | Has at Least One Hearts | Hand contains at least one hearts | has | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_025 | Has at Least One Spades | Hand contains at least one spades | has | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_026 | No Clubs | Hand contains no clubs | no | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_027 | No Diamonds | Hand contains no diamonds | no | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_028 | No Hearts | Hand contains no hearts | no | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_029 | No Spades | Hand contains no spades | no | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_030 | No Red Cards | Hand contains no red cards | no | `has_pair`, `has_triple`, `unique_ranks` |
| atomic_031 | No Black Cards | Hand contains no black cards | no | `has_pair`, `has_triple`, `unique_ranks` |

#### Middle Position Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_032 | Middle Card is Red | The middle card is red | position_middle | `first_card`, `last_card` |
| atomic_033 | Middle Card is Black | The middle card is black | position_middle | `first_card`, `last_card` |
| atomic_034 | Middle Card is Clubs | The middle card is a clubs | position_middle | `first_card`, `last_card` |
| atomic_035 | Middle Card is Diamonds | The middle card is a diamonds | position_middle | `first_card`, `last_card` |
| atomic_036 | Middle Card is Hearts | The middle card is a hearts | position_middle | `first_card`, `last_card` |
| atomic_037 | Middle Card is Spades | The middle card is a spades | position_middle | `first_card`, `last_card` |

#### Second Position Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_038 | Second Card is Clubs | Second card is a clubs | position_second | `first_card` |
| atomic_039 | Second Card is Diamonds | Second card is a diamonds | position_second | `first_card` |
| atomic_040 | Second Card is Hearts | Second card is a hearts | position_second | `first_card` |
| atomic_041 | Second Card is Spades | Second card is a spades | position_second | `first_card` |
| atomic_042 | Second Card is Red | Second card is red | position_second | `first_card` |
| atomic_043 | Second Card is Black | Second card is black | position_second | `first_card` |

#### Range Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| atomic_047 | All Low Cards | All cards have rank 7 or lower | range | `max_rank`, `min_rank`, `rank_spread` |
| atomic_048 | All High Cards | All cards have rank 8 or higher | range | `max_rank`, `min_rank`, `rank_spread` |
| atomic_049 | Has Low Card | Contains card with rank 5 or lower | range | `max_rank`, `min_rank`, `rank_spread` |
| atomic_050 | Has High Card | Contains card with rank 10 or higher | range | `max_rank`, `min_rank`, `rank_spread` |

---

### Level 2: Comparison Rules (22 rules)

Rules comparing properties between positions or halves.

#### Terminal Comparison Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| compare_051 | First and Last Same Suit | First and last cards have the same suit | terminals | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_052 | First and Last Same Color | First and last cards have the same color | terminals | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_053 | First and Last Same Rank | First and last cards have the same rank | terminals | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_054 | First and Last Same Parity | First and last cards have the same rank parity | terminals | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_055 | First and Last Different Suits | First and last cards have different suits | terminals | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_056 | First and Last Different Colors | First and last cards have different colors | terminals | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_057 | First Rank Greater Than Last | First card has higher rank than last card | terminals_rank | `first_card`, `last_card`, `rank_spread` |
| compare_058 | First Rank Less Than Last | First card has lower rank than last card | terminals_rank | `first_card`, `last_card`, `rank_spread` |

#### Halves Comparison Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| compare_059 | Halves Uniform Color Equal | Both halves are uniformly colored, or both are not | halves_property | `is_uniform_color`, `halves_same_color` |
| compare_060 | Halves Uniform Suit Equal | Both halves have uniform suit, or both don't | halves_property | `is_uniform_color`, `halves_same_color` |
| compare_061 | Halves Same Suit Set | Both halves contain the same set of suits | halves_set | `unique_suits`, `halves_same_color` |
| compare_062 | Halves Same Color Set | Both halves contain the same set of colors | halves_set | `unique_suits`, `halves_same_color` |
| compare_063 | Left Half Sum Greater | Sum of ranks in left half is greater than right half | halves_sum | `sum_ranks` |
| compare_064 | Halves Equal Sum | Sum of ranks in left half equals right half | halves_sum | `sum_ranks` |

#### Adjacent Comparison Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| compare_065 | All Adjacent Same Color | All adjacent cards have the same color | adjacent | `is_sorted`, `rank_spread` |
| compare_066 | All Adjacent Different Colors | All adjacent cards have different colors (alternating) | adjacent | `is_sorted`, `rank_spread` |

#### Position Pair Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| compare_067 | Position 0 and 1 Same Suit | Cards at positions 0 and 1 have the same suit | position_pair | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_068 | Position 0 and 2 Same Suit | Cards at positions 0 and 2 have the same suit | position_pair | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_069 | Position 1 and 2 Same Suit | Cards at positions 1 and 2 have the same suit | position_pair | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_070 | Position 1 and 3 Same Suit | Cards at positions 1 and 3 have the same suit | position_pair | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_071 | Position 2 and 3 Same Suit | Cards at positions 2 and 3 have the same suit | position_pair | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |
| compare_072 | Position 2 and 4 Same Suit | Cards at positions 2 and 4 have the same suit | position_pair | `first_card`, `last_card`, `ends_same_suit`, `ends_same_color` |

---

### Level 3: Counting Rules (47 rules)

Rules based on counting elements.

#### Unique Count Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| count_073 | Exactly 1 Unique Suit | Hand contains exactly 1 different suit | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_074 | Exactly 2 Unique Suits | Hand contains exactly 2 different suits | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_075 | Exactly 3 Unique Suits | Hand contains exactly 3 different suits | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_076 | Exactly 4 Unique Suits | Hand contains exactly 4 different suits | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_077 | Exactly 1 Unique Color | Hand contains exactly 1 different color | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_078 | Exactly 2 Unique Colors | Hand contains exactly 2 different colors | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_079 | Exactly 2 Unique Ranks | Hand contains exactly 2 different ranks | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_080 | Exactly 3 Unique Ranks | Hand contains exactly 3 different ranks | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_081 | Exactly 4 Unique Ranks | Hand contains exactly 4 different ranks | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_082 | Exactly 5 Unique Ranks | Hand contains exactly 5 different ranks | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_083 | Exactly 6 Unique Ranks | Hand contains exactly 6 different ranks | unique_count | `unique_suits`, `unique_colors`, `unique_ranks` |

#### Unique Bound Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| count_084 | At Most 2 Unique Suits | Hand contains at most 2 different suits | unique_bound | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_085 | At Most 3 Unique Suits | Hand contains at most 3 different suits | unique_bound | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_086 | At Least 2 Unique Suits | Hand contains at least 2 different suits | unique_bound | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_087 | At Least 3 Unique Suits | Hand contains at least 3 different suits | unique_bound | `unique_suits`, `unique_colors`, `unique_ranks` |
| count_088 | At Least 4 Unique Suits | Hand contains at least 4 different suits | unique_bound | `unique_suits`, `unique_colors`, `unique_ranks` |

#### Specific Count Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| count_089 | Exactly 1 Clubs | Hand contains exactly 1 clubs | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_090 | Exactly 2 Clubs | Hand contains exactly 2 clubs | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_091 | Exactly 3 Clubs | Hand contains exactly 3 clubs | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_092 | Exactly 1 Diamonds | Hand contains exactly 1 diamonds | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_093 | Exactly 2 Diamonds | Hand contains exactly 2 diamonds | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_094 | Exactly 3 Diamonds | Hand contains exactly 3 diamonds | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_095 | Exactly 1 Hearts | Hand contains exactly 1 hearts | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_096 | Exactly 2 Hearts | Hand contains exactly 2 hearts | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_097 | Exactly 3 Hearts | Hand contains exactly 3 hearts | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_098 | Exactly 1 Spades | Hand contains exactly 1 spades | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_099 | Exactly 2 Spades | Hand contains exactly 2 spades | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_100 | Exactly 3 Spades | Hand contains exactly 3 spades | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_101 | Exactly 1 Red Card | Hand contains exactly 1 red card | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_102 | Exactly 2 Red Cards | Hand contains exactly 2 red cards | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_103 | Exactly 3 Red Cards | Hand contains exactly 3 red cards | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_104 | Exactly 4 Red Cards | Hand contains exactly 4 red cards | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_105 | Exactly 1 Black Card | Hand contains exactly 1 black card | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_106 | Exactly 2 Black Cards | Hand contains exactly 2 black cards | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_107 | Exactly 3 Black Cards | Hand contains exactly 3 black cards | count_specific | `suit_count`, `color_count`, `rank_count` |
| count_108 | Exactly 4 Black Cards | Hand contains exactly 4 black cards | count_specific | `suit_count`, `color_count`, `rank_count` |

#### Count Comparison Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| count_109 | More Red Than Black | More red cards than black cards | count_compare | `suit_count`, `color_count` |
| count_110 | More Black Than Red | More black cards than red cards | count_compare | `suit_count`, `color_count` |
| count_111 | Equal Red and Black | Equal number of red and black cards | count_compare | `suit_count`, `color_count` |

#### Majority Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| count_112 | Majority Same Suit | More than half the cards share the same suit | majority | `suit_count`, `color_count`, `is_uniform_color` |
| count_113 | Majority Same Color | More than half the cards share the same color | majority | `suit_count`, `color_count`, `is_uniform_color` |

#### Parity Count Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| count_114 | Exactly 1 Odd Rank | Exactly 1 card with odd rank | count_parity | `suit_count`, `color_count` |
| count_115 | Exactly 2 Odd Ranks | Exactly 2 cards with odd rank | count_parity | `suit_count`, `color_count` |
| count_116 | Exactly 3 Odd Ranks | Exactly 3 cards with odd rank | count_parity | `suit_count`, `color_count` |
| count_117 | Exactly 1 Even Rank | Exactly 1 card with even rank | count_parity | `suit_count`, `color_count` |
| count_118 | Exactly 2 Even Ranks | Exactly 2 cards with even rank | count_parity | `suit_count`, `color_count` |
| count_119 | Exactly 3 Even Ranks | Exactly 3 cards with even rank | count_parity | `suit_count`, `color_count` |

---

### Level 4: Pattern Rules (29 rules)

Rules detecting sequential or structural patterns.

#### Sorted Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_120 | Sorted by Rank (Non-decreasing) | Ranks are in non-decreasing order left to right | sorted | `is_sorted` |
| pattern_121 | Sorted by Rank (Strictly Increasing) | Ranks are in strictly increasing order | sorted | `is_sorted` |
| pattern_122 | Sorted by Rank (Non-increasing) | Ranks are in non-increasing order | sorted | `is_sorted` |
| pattern_148 | Monotonic Ranks | Ranks are either non-decreasing or non-increasing | monotonic | `is_sorted`, `rank_spread` |

#### Alternating Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_123 | Alternating Colors | Colors alternate (red-black-red-black...) | alternating | `is_uniform_color`, `unique_colors` |
| pattern_124 | Alternating Parities | Rank parities alternate (odd-even-odd-even...) | alternating | `is_uniform_color`, `unique_colors` |

#### Duplicates Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_125 | Has Pair (Same Rank) | At least two cards share the same rank | duplicates | `has_pair`, `has_triple`, `unique_ranks` |
| pattern_126 | Has Triple (Three of a Kind) | At least three cards share the same rank | duplicates | `has_pair`, `has_triple`, `unique_ranks` |
| pattern_127 | Has Adjacent Suit Pair | At least two adjacent cards share the same suit | duplicates | `has_pair`, `has_triple`, `unique_ranks` |

#### Palindrome Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_128 | Suits Palindrome | Suit sequence reads the same forwards and backwards | palindrome | `is_palindrome_colors`, `is_palindrome_suits` |
| pattern_129 | Colors Palindrome | Color sequence reads the same forwards and backwards | palindrome | `is_palindrome_colors`, `is_palindrome_suits` |
| pattern_130 | Ranks Palindrome | Rank sequence reads the same forwards and backwards | palindrome | `is_palindrome_colors`, `is_palindrome_suits` |
| pattern_131 | Parities Palindrome | Parity sequence reads the same forwards and backwards | palindrome | `is_palindrome_colors`, `is_palindrome_suits` |

#### Halves Copy Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_132 | Halves Copy Suits | Left and right halves have the same suit sequence | halves_copy | `is_palindrome_suits`, `halves_same_color` |
| pattern_133 | Halves Copy Colors | Left and right halves have the same color sequence | halves_copy | `is_palindrome_suits`, `halves_same_color` |
| pattern_134 | Halves Copy Ranks | Left and right halves have the same rank sequence | halves_copy | `is_palindrome_suits`, `halves_same_color` |

#### Run Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_135 | Has Run of 3 | Contains at least 3 consecutive ranks | runs | `is_sorted`, `rank_spread` |
| pattern_136 | Has Run of 4 | Contains at least 4 consecutive ranks | runs | `is_sorted`, `rank_spread` |

#### Adjacent Constraint Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_137 | Adjacent Rank Gap <= 2 | All adjacent cards differ by at most 2 in rank | adjacent_constraint | `is_sorted`, `unique_suits`, `unique_colors` |
| pattern_138 | Adjacent Rank Gap <= 3 | All adjacent cards differ by at most 3 in rank | adjacent_constraint | `is_sorted`, `unique_suits`, `unique_colors` |
| pattern_139 | Adjacent Same Rank or Suit | All adjacent cards share either rank or suit | adjacent_constraint | `is_sorted`, `unique_suits`, `unique_colors` |

#### Shape Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_140 | V-Shape Ranks | Ranks descend to middle then ascend (V shape) | shape | `has_pair`, `has_triple`, `unique_ranks` |
| pattern_141 | Inverted V-Shape Ranks | Ranks ascend to middle then descend (inverted V) | shape | `has_pair`, `has_triple`, `unique_ranks` |

#### Periodic Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_142 | Period-2 Colors | Colors repeat with period 2 (ABABAB...) | periodic | `unique_suits`, `unique_colors` |
| pattern_143 | Period-2 Suits | Suits repeat with period 2 (ABABAB...) | periodic | `unique_suits`, `unique_colors` |

#### Skip Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_144 | Skip-1 Same Suit | Cards at positions i and i+2 have same suit | skip | `is_sorted`, `rank_spread` |
| pattern_145 | Skip-1 Same Color | Cards at positions i and i+2 have same color | skip | `is_sorted`, `rank_spread` |

#### Spread Rules

| ID | Name | Description | Category | Acceptable Features |
|----|------|-------------|----------|---------------------|
| pattern_146 | Rank Spread >= 8 | Difference between highest and lowest rank is at least 8 | spread | `max_rank`, `min_rank`, `rank_spread` |
| pattern_147 | Rank Spread <= 5 | Difference between highest and lowest rank is at most 5 | spread | `max_rank`, `min_rank`, `rank_spread` |

---

### Level 5: Compositional Rules (53 rules)

Rules combining multiple conditions with logical connectives.

#### AND Combination Rules

| ID | Name | Description |
|----|------|-------------|
| comp_149 | All Same Color AND Sorted | All cards same color AND ranks are sorted |
| comp_150 | Has Pair AND Alternating Colors | Has a pair AND colors alternate |
| comp_151 | First Red AND Last Black | First card is red AND last card is black |
| comp_152 | First Black AND Last Red | First card is black AND last card is red |
| comp_153 | Exactly 2 Suits AND Has Pair | Exactly 2 suits AND has at least one pair |
| comp_154 | Sum Even AND First Red | Sum of ranks is even AND first card is red |
| comp_155 | Halves Same Colors AND Sorted | Both halves have same color set AND ranks are sorted |
| comp_171 | Alternating Colors AND First Spade | Colors alternate AND first card is a spade |
| comp_172 | Has Run of 3 AND Uniform Color | Contains 3 consecutive ranks AND all cards same color |
| comp_173 | Palindrome Colors AND Has Heart | Color sequence is palindrome AND has at least one heart |
| comp_174 | More Black AND Last Black | More black cards than red AND last card is black |
| comp_175 | At Least 3 Suits AND Sorted | At least 3 different suits AND ranks are sorted |
| comp_192 | Has Heart AND Has Club | Has at least one heart AND at least one club |
| comp_193 | Has Diamond AND Has Spade | Has at least one diamond AND at least one spade |
| comp_194 | All Four Suits Present | Hand contains cards from all four suits |
| comp_195 | Exactly One Suit Missing | Exactly three suits are present (one is missing) |
| comp_197 | First Pair AND Last Pair Same Suit | First two cards share suit AND last two cards share suit |
| comp_200 | Sum Div 3 AND Has Heart | Sum of ranks divisible by 3 AND has at least one heart |
| comp_201 | Halves Copy Colors AND First Black | Left and right halves have same color sequence AND first card is black |

#### OR Combination Rules

| ID | Name | Description |
|----|------|-------------|
| comp_156 | All Same Suit OR All Same Color | All cards same suit OR all cards same color |
| comp_157 | Has Pair OR Sorted | Has a pair OR ranks are sorted |
| comp_158 | First Red OR Last Red | First card is red OR last card is red |
| comp_159 | Palindrome Suits OR Palindrome Colors | Suits form palindrome OR colors form palindrome |
| comp_160 | Sum Even OR Has Spade | Sum of ranks is even OR has at least one spade |
| comp_176 | Has Triple OR All Same Color | Has three of a kind OR all cards same color |
| comp_177 | First Black OR Last Black | First card is black OR last card is black |
| comp_178 | Sum Even OR Alternating Colors | Sum of ranks is even OR colors alternate |
| comp_179 | More Suit Variety OR Has Pair | More unique suits than colors OR has a pair |
| comp_196 | All Even OR All Odd Ranks | All cards have even ranks OR all cards have odd ranks |
| comp_198 | First Pair OR Last Pair Same Color | First two cards share color OR last two cards share color |

#### Conditional Rules

| ID | Name | Description |
|----|------|-------------|
| comp_161 | If First Red Then Last Black | If first card is red, then last card must be black |
| comp_162 | If Has Spade Then Has Heart | If hand has a spade, it must also have a heart |
| comp_163 | If Sorted Then Has Pair | If ranks are sorted, then must have at least one pair |
| comp_164 | If Uniform Color Then Not Sorted | If all cards same color, then ranks must not be sorted |
| comp_180 | If Uniform Color Then Sum Even | If all cards same color, then sum of ranks must be even |
| comp_181 | If Has Pair Then Not Palindrome Suits | If hand has a pair, then suits must not form palindrome |
| comp_182 | If First Heart Then Last Diamond | If first card is heart, then last card must be diamond |
| comp_183 | If More Red Then First Red | If more red than black, then first card must be red |

#### XOR Rules

| ID | Name | Description |
|----|------|-------------|
| comp_169 | Same Suit XOR Same Color | All same suit XOR all same color (exactly one, not both) |
| comp_170 | Sorted XOR Has Pair | Sorted XOR has pair (exactly one, not both) |

#### Negation Rules

| ID | Name | Description |
|----|------|-------------|
| comp_186 | Not All Same Suit | Not all cards have the same suit (at least 2 different suits) |
| comp_187 | Not Sorted | Ranks are NOT in non-decreasing order |
| comp_188 | Not Palindrome Colors | Color sequence is NOT a palindrome |
| comp_191 | Neither Same Suit Nor Same Color | Not all same suit AND not all same color |
| comp_199 | No Pair AND Not Sorted | No pairs AND ranks are not sorted |

#### Complex Rules

| ID | Name | Description |
|----|------|-------------|
| comp_165 | Opposite Terminal Colors | First and last cards have opposite colors |
| comp_166 | Same Color AND (Sorted OR Pair) | All same color AND (sorted OR has pair) |
| comp_167 | Has Both of a Color Pair | Has (spade AND heart) OR (diamond AND club) |
| comp_168 | More Red AND First Red | More red cards than black AND first card is red |

#### Triple Combination Rules

| ID | Name | Description |
|----|------|-------------|
| comp_184 | Same Color AND Sorted AND Pair | All same color AND sorted AND has at least one pair |
| comp_185 | First Red AND Last Black AND Has Spade | First is red AND last is black AND has at least one spade |

#### Biconditional Rules

| ID | Name | Description |
|----|------|-------------|
| comp_189 | Same Suit IFF Same Color | All same suit if and only if all same color |
| comp_190 | Has Pair IFF <= 5 Unique Ranks | Has pair if and only if at most 5 unique ranks |

---

## Part 5: Evaluation Results Summary

From the quick evaluation (December 26, 2024):

| Metric | Value |
|--------|-------|
| **Semantic Correctness** | **52.9%** |
| Consistency | 94.1% |
| Discrimination | 98.9% |
| FCR@3 (legacy) | 0% |

### By Level

| Level | Description | Semantic Correctness |
|-------|-------------|---------------------|
| 1 | Atomic | 40.0% |
| 2 | Comparison | 66.7% |
| 3 | Counting | 83.3% |
| 4 | Pattern | 40.0% |
| 5 | Compositional | 50.0% (lenient) |

---

## Part 6: How Semantic Correctness Works

### The Evaluation Algorithm

```
For each rule:
  1. Get the rule's category (e.g., "uniform", "sorted")
  2. Look up acceptable features for that category
  3. Generate descriptions for task examples
  4. For each description in top-3:
     a. Extract features from description object
     b. Extract features from description text (keyword matching)
     c. Check if any feature overlaps with acceptable set
  5. Score = 1.0 if ANY description matches, else 0.0
```

### Example

**Rule**: "All Same Suit" (category: `uniform`)
**Acceptable features**: `is_flush`, `is_uniform_color`, `unique_suits`, `unique_colors`, `unique_ranks`

**Generated description**: "all cards share the same suit"
**Extracted features**: `is_flush` (from direct feature), `unique_suits` (from "same suit" text)
**Overlap with acceptable**: `is_flush` matches!
**Score**: 1.0 (semantically correct)

---

*Document generated December 26, 2024*
