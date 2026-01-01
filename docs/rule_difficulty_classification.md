# Rule Difficulty Classification

This document classifies the 45 catalogue rules by computational difficulty for curriculum learning. The classification is based on:
- Required primitives (aggregate vs list operations)
- Compositional depth
- Rule family complexity

**Key Finding**: Adding list primitives (take, drop, zip_with, adjacent_pairs, half_len) increased solve rate from ~8/45 to ~30-40/45 rules.

---

## Phase 1: Easy Rules (8 rules)

These rules use only aggregate primitives and should solve immediately.

### COUNT Family - Basic Cardinality
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Uniform_color` | All cards same color | `all_same_color(hand)` |
| `Exactly_two_suits` | Exactly 2 unique suits | `n_unique_suits == 2` |
| `At_most_three_suits` | At most 3 unique suits | `n_unique_suits <= 3` |
| `Exactly_one_club` | Exactly one club | `count_suit(CLUBS) == 1` |
| `Has_pair_ranks` | Has duplicate rank | `n_unique_ranks < length` |
| `Half_or_more_same_suit` | Majority suit exists | `max(count_suit) >= length/2` |

### LOCAL Family - Simple Comparisons
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Ends_same_suit` | First and last same suit | `get_suit(head) == get_suit(last)` |
| `Ends_same_color` | First and last same color | `get_color(head) == get_color(last)` |

---

## Phase 2: Medium Rules (12 rules)

These rules need list operations or position access.

### POSITION Family
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Pos3_is_JQK` | 3rd card is J/Q/K | `at(2) in {J,Q,K}` |
| `Pos4_is_2_5_7` | 4th card is 2/5/7 | `at(3) in {2,5,7}` |

### TOKEN Family
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Has_Ace_of_Spades` | Contains A♠ | `any(suit==S and rank==A)` |
| `Has_6_of_Diamonds` | Contains 6♦ | `any(suit==D and rank==6)` |

### PARITY Family
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Only_one_odd_rank` | Exactly one odd rank | `count(odd_rank) == 1` |
| `Uniform_rank_parity` | All same parity | `all_same(parity)` |

### HIER Family - Simple Halves
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Halves_uniform_color_equal` | Both halves uniform color or not | `uniform_color(left) == uniform_color(right)` |
| `Halves_hearts_presence_equal` | Both halves have hearts or neither | `has_heart(left) == has_heart(right)` |
| `Halves_same_suit_set` | Same unique suits in both halves | `unique_suits(left) == unique_suits(right)` |

### LOCAL Family - Ordering
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Sorted_by_rank` | Non-decreasing ranks | `all(adjacent_pairs, non_decreasing)` |
| `S_before_H` | Some spade before some heart | `exists(spade_before_heart)` |

### ALTCOLOR Family
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Ends_same_altcolor1` | First/last same pointy/round | `altcolor1(head) == altcolor1(last)` |

---

## Phase 3: Hard Rules (18 rules)

These rules require complex list operations like reverse, zip_with, take, drop.

### PAL Family - Palindromes (need reverse + zip_with)
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Suits_palindrome` | Suit sequence is palindrome | `suits == reverse(suits)` |
| `Colors_palindrome` | Color sequence is palindrome | `colors == reverse(colors)` |
| `Ranks_palindrome` | Rank sequence is palindrome | `ranks == reverse(ranks)` |
| `AltColor1_palindrome` | Altcolor1 sequence is palindrome | `altcolor1s == reverse(altcolor1s)` |
| `AltColor2_palindrome` | Altcolor2 sequence is palindrome | `altcolor2s == reverse(altcolor2s)` |

### COPY Family - Half Comparison (need take/drop + zip_with)
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Halves_copy_suits` | Left half suits == right half suits | `take(n/2, suits) == drop(n/2, suits)` |
| `Halves_copy_colors` | Left half colors == right half colors | `take(n/2, colors) == drop(n/2, colors)` |
| `Halves_copy_ranks` | Left half ranks == right half ranks | `take(n/2, ranks) == drop(n/2, ranks)` |
| `Halves_copy_altcolor1` | Left half altcolor1 == right half | `take(n/2, altcolor1) == drop(n/2, altcolor1)` |
| `Halves_copy_altcolor2` | Left half altcolor2 == right half | `take(n/2, altcolor2) == drop(n/2, altcolor2)` |

### SHIFT Family - Shifted Pairs
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Shift_half_plus_two` | rank[i+n/2] == rank[i] + 2 | `all(shifted_pairs(n/2), diff == 2)` |
| `Shift2_plus3` | rank[i+2] == rank[i] + 3 | `all(shifted_pairs(2), diff == 3)` |
| `Shift_half_ge` | rank[i+n/2] >= rank[i] | `all(shifted_pairs(n/2), ge)` |

### ADJ Family - Adjacent Comparisons
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Adj_same_rank_or_suit` | Adjacent cards share rank or suit | `all(adjacent_pairs, same_rank_or_suit)` |
| `Skip2_same_rank_or_suit` | Cards 2 apart share rank or suit | `all(skip2_pairs, same_rank_or_suit)` |
| `Adj_rank_gap_le3` | Adjacent rank difference <= 3 | `all(adjacent_pairs, abs(diff) <= 3)` |

### More HIER Family
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Halves_uniform_parity_equal` | Both halves same parity uniformity | `uniform_parity(left) == uniform_parity(right)` |
| `Halves_AP_step1_equal` | Both halves have step-1 AP or not | `has_AP_step1(left) == has_AP_step1(right)` |
| `Halves_AP_len2_step1_equal` | Both halves have len-2 step-1 AP | `has_AP_len2_step1(left) == has_AP_len2_step1(right)` |

---

## Phase 4: Very Hard Rules (7+ rules)

These rules involve complex compositions, state machines, or unusual primitives.

### LANG Family - Bracket Matching (may need PDA-style state)
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Well_formed_brackets_by_suit` | Spades/Hearts open, Clubs/Diamonds close | PDA with stack |
| `Even_opens_next_closes` | Even rank opens, next odd closes | Context-sensitive matching |
| `Odd_opens_next_closes` | Odd rank opens, next even closes | Context-sensitive matching |

### CENTER Family - Radial Patterns
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Halves_radial_nonincreasing` | Ranks decrease from center | Complex positional logic |
| `Global_radial_no_dominance` | No rank dominates radially | Complex comparison |

### SCORE Family - Sum Thresholds
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Score_threshold_Rstar` | Total score >= threshold | `sum(rank_vals) >= R*` |
| `Half_sum_diff_geN` | Half sum difference >= N | `abs(sum(left) - sum(right)) >= N` |
| `Half_sum_one_side_ge_2x_other` | One half sum >= 2x other | `sum(left) >= 2*sum(right)` |

### AP Family - Arithmetic Progressions
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `AP_len3_anywhere_anyk` | Any 3 ranks form AP | `exists(3-subset, is_AP)` |
| `AP_len3_step2_anywhere` | Any 3 ranks form step-2 AP | `exists(3-subset, is_AP_step2)` |
| `AP_len4_step2_anywhere` | Any 4 ranks form step-2 AP | `exists(4-subset, is_AP_step2)` |
| `Halves_AP_len3_any_equal` | Both halves have AP or neither | `has_AP_len3(left) == has_AP_len3(right)` |

### MAP Family - Suit Cycles (archived, not in current catalogue)
| Rule ID | Description | Program Pattern |
|---------|-------------|-----------------|
| `Half_map_samepos_M1` | Suit mapping M1 across halves | Custom suit permutation |
| `Half_map_samepos_M2` | Suit mapping M2 across halves | Custom suit permutation |
| `Step2_back_map_M1` | Suit mapping with step-2 offset | Custom suit permutation |
| `Step2_back_map_M2` | Suit mapping with step-2 offset | Custom suit permutation |
| `Adj_same_or_map_M1` | Adjacent same suit or M1-mapped | Custom suit permutation |
| `Adj_same_or_map_M2` | Adjacent same suit or M2-mapped | Custom suit permutation |

---

## Curriculum Learning Strategy

### Phase-Based Task Sets (Cumulative)
- **Phase 1**: Easy rules only (8 tasks) - Build foundation
- **Phase 2**: Easy + Medium (20 tasks) - Add list operations
- **Phase 3**: Easy + Medium + Hard (38 tasks) - Add complex compositions
- **Phase 4**: All rules (45 tasks) - Intensive final push

### Recommended Hyperparameters by Phase

| Phase | Budget | Max Depth | Dreams | Recognition Epochs |
|-------|--------|-----------|--------|-------------------|
| 1 | 300K | 10 | 100 | 15 |
| 2 | 500K | 12 | 150 | 20 |
| 3 | 800K | 14 | 200 | 20 |
| 4 | 1M | 15 | 250 | 25 |

---

## Key Primitives for Each Difficulty Level

### Easy (Aggregate Only)
- `all_same_color`, `all_same_suit`
- `n_unique_suits`, `n_unique_ranks`
- `count_suit`, `count_color`
- `get_suit`, `get_color`, `get_rank`
- `head`, `last`

### Medium (+ Position Access)
- `at(n)` - Position access
- `any`, `all` - Quantifiers
- `adjacent_pairs` - Pairwise iteration

### Hard (+ List Operations)
- `take(n)`, `drop(n)` - List slicing
- `reverse` - List reversal
- `zip_with` - Parallel iteration
- `half_len` - Length division

### Very Hard (+ State/Recursion)
- Stack-based matching (LANG family)
- Subset enumeration (AP family)
- Custom permutations (MAP family)

---

## Historical Note

This classification was developed in December 2024 for the `run_overnight_listprims.py` experiment. The key insight was that **list primitives are critical** for solving rules involving:
- Palindromes (need `reverse`)
- Half comparison (need `take`/`drop`)
- Adjacent patterns (need `adjacent_pairs`)
- Shifted patterns (need `zip_with` with offset)

Without list primitives, only ~8/45 rules could be solved. With them, ~30-40/45 rules became solvable.
