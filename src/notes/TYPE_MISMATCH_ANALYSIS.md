# Type Mismatch Analysis: Why List Primitives Didn't Help

*Analysis of 10 representative rules showing what's needed to solve them*

---

## The Core Problem

The list primitives we added have a **type gap**:

```
zip_with : (a -> b -> bool) -> list(a) -> list(b) -> list(bool)
                                                      ^^^^^^^^
                                                      Returns list(bool), NOT bool!
```

But all our rules have type `Hand -> bool`. There's no way to go from `list(bool)` to `bool` without an aggregator like `all` or `any` applied to the list.

We have `all : (a -> bool) -> list(a) -> bool` but this takes a **predicate**, not a `list(bool)`.

**What's missing:** `all_true : list(bool) -> bool` (or equivalently, `and_fold`)

---

## Analysis of 10 Representative Rules

### Legend
- **Original**: What it would take with the pre-list-primitives library
- **Current**: What it would take with current library (take, drop, zip_with, etc.)
- **Proposed**: What it would take with fixes (all_true, first_half, second_half)

---

### 1. Suits_palindrome
**Rule:** Suit sequence reads same forward and backward

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible - no reverse or zip_with | N/A | ❌ No |
| **Current** | `(λ ??? (zip_with eq (map get_suit $0) (reverse (map get_suit $0))))` | N/A | ❌ No - zip_with returns list(bool), need aggregator |
| **Proposed** | `(λ all_true (zip_with eq (map get_suit $0) (reverse (map get_suit $0))))` | 5 | ✅ Yes |

**What's needed:** `all_true : list(bool) -> bool`

---

### 2. Halves_copy_suits
**Rule:** Left half suits equal right half suits (element-wise)

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible - no take/drop | N/A | ❌ No |
| **Current** | `(λ ??? (zip_with eq (map get_suit (take 3 $0)) (map get_suit (drop 3 $0))))` | N/A | ❌ No - same list(bool) problem |
| **Proposed (v1)** | `(λ all_true (zip_with eq (map get_suit (take 3 $0)) (map get_suit (drop 3 $0))))` | 7 | ⚠️ Maybe - depth 7 is high |
| **Proposed (v2)** | `(λ all_true (zip_with eq (map get_suit (first_half $0)) (map get_suit (second_half $0))))` | 6 | ✅ Better |

**What's needed:** `all_true` + optionally `first_half`/`second_half` macros

---

### 3. Sorted_by_rank
**Rule:** Ranks are in non-decreasing order

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible - no adjacent comparison | N/A | ❌ No |
| **Current** | `(λ ??? (zip_with le (map rank_val $0) (drop 1 (map rank_val $0))))` | N/A | ❌ No - list(bool) problem |
| **Proposed** | `(λ all_true (zip_with le (map rank_val $0) (drop 1 (map rank_val $0))))` | 6 | ✅ Yes |

**Alternative with adjacent_pairs:** Would need a way to apply `le` to each pair and aggregate.

---

### 4. Halves_uniform_color_equal
**Rule:** Both halves are uniform in color (or both are not)

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible - no halves | N/A | ❌ No |
| **Current** | `(λ eq (all_same_color (take 3 $0)) (all_same_color (drop 3 $0)))` | 5 | ✅ Should work! |
| **Proposed** | Same, or use `first_half`/`second_half` | 4 | ✅ Even better |

**Wait - this SHOULD be solvable now!** Let's check why it wasn't solved...

The issue: `take 3` and `drop 3` assume hand size 6. But `half_len` returns an INT, and we'd need:
`(take (half_len $0) $0)` - but that requires passing the result of half_len to take, which is depth 3 just for the slice.

Full program: `(λ eq (all_same_color (take (half_len $0) $0)) (all_same_color (drop (half_len $0) $0)))` = depth 6

---

### 5. Shift2_plus3
**Rule:** Every card at position i+2 has rank 3 higher than position i

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible | N/A | ❌ No |
| **Current** | Would need shifted_pairs or zip of drop 2 with original, then check diff = 3 | N/A | ❌ No - list(bool) problem |
| **Proposed** | `(λ all_true (zip_with (λ x (λ y (eq 3 (- (rank_val y) (rank_val x))))) $0 (drop 2 $0)))` | 8+ | ⚠️ Very deep |

**Note:** This was "solved" as `(λ false)` which is spurious!

---

### 6. Adj_same_rank_or_suit
**Rule:** Every adjacent pair shares rank or suit

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible - no adjacent pairs | N/A | ❌ No |
| **Current** | `adjacent_pairs` returns `list(list(card))` but we can't easily apply predicates | N/A | ❌ No - need to map over pairs |
| **Proposed** | `(λ all (λ pair (or (eq (get_rank (head pair)) (get_rank (last pair))) (eq (get_suit (head pair)) (get_suit (last pair))))) (adjacent_pairs $0))` | 8+ | ⚠️ Very deep |

**Problem:** Even with `all`, the inner predicate is complex. Would benefit from `fst`/`snd` pair accessors.

---

### 7. Colors_palindrome
**Rule:** Color sequence is palindromic

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible | N/A | ❌ No |
| **Current** | Same as Suits_palindrome - blocked by list(bool) | N/A | ❌ No |
| **Proposed** | `(λ all_true (zip_with eq (map get_color $0) (reverse (map get_color $0))))` | 5 | ✅ Yes |

---

### 8. Ends_same_suit
**Rule:** First and last cards have same suit

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | `(λ eq (get_suit (head $0)) (get_suit (last $0)))` | 4 | ✅ Yes |
| **Current** | Same, or `(λ has_suit (take 1 $0) (get_suit (last $0)))` | 4 | ✅ Yes |
| **Proposed** | Same | 4 | ✅ Yes |

**This one was solved!** It doesn't actually need list primitives.

---

### 9. Half_or_more_same_suit
**Rule:** At least half the cards share a suit

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Would need: `(λ ge (max (count_suit $0 CLUBS) (max (count_suit $0 DIAMONDS) ...)) (half_len $0))` | 6+ | ⚠️ Needs max over all suits |
| **Current** | Same problem - no `max` over suits primitive | N/A | ⚠️ Hard |
| **Proposed** | Add `max_count_suit : hand -> int` as high-level primitive | 2 | ✅ Easy |

---

### 10. Halves_AP_step1_equal
**Rule:** Both halves are "runs" (consecutive ranks) or both are not

| Approach | Program | Depth | Feasible? |
|----------|---------|-------|-----------|
| **Original** | Not expressible | N/A | ❌ No |
| **Current** | Would need `is_run` as primitive, or build from adjacent pairs | 8+ | ❌ Too deep |
| **Proposed** | Add `is_run : hand -> bool` primitive, then `(λ eq (is_run (first_half $0)) (is_run (second_half $0)))` | 4 | ✅ Yes |

---

## Summary Table

| Rule | Original | Current | With all_true | With high-level |
|------|----------|---------|---------------|-----------------|
| Suits_palindrome | ❌ | ❌ | ✅ depth 5 | ✅ depth 5 |
| Halves_copy_suits | ❌ | ❌ | ⚠️ depth 7 | ✅ depth 5 |
| Sorted_by_rank | ❌ | ❌ | ✅ depth 6 | ✅ depth 6 |
| Halves_uniform_color_equal | ❌ | ⚠️ depth 6 | ⚠️ depth 6 | ✅ depth 4 |
| Shift2_plus3 | ❌ | ❌ | ⚠️ depth 8+ | ⚠️ depth 8+ |
| Adj_same_rank_or_suit | ❌ | ❌ | ⚠️ depth 8+ | ⚠️ depth 8+ |
| Colors_palindrome | ❌ | ❌ | ✅ depth 5 | ✅ depth 5 |
| Ends_same_suit | ✅ depth 4 | ✅ depth 4 | ✅ depth 4 | ✅ depth 4 |
| Half_or_more_same_suit | ⚠️ | ⚠️ | ⚠️ | ✅ depth 2 |
| Halves_AP_step1_equal | ❌ | ❌ | ❌ | ✅ depth 4 |

---

## Recommendations

### Minimal Fix: Add `all_true`
```python
Primitive(
    'all_true',
    arrow(LIST_BOOL, BOOL),
    lambda xs: all(xs)
)
```
This would unlock: Suits_palindrome, Colors_palindrome, Sorted_by_rank, and make Halves_copy_* feasible.

### Better Fix: Add Higher-Level Primitives
```python
# Halves operations
Primitive('first_half', arrow(HAND, HAND), lambda h: h[:len(h)//2])
Primitive('second_half', arrow(HAND, HAND), lambda h: h[len(h)//2:])

# Sequence checks
Primitive('is_palindrome_by', arrow(arrow(CARD, a), HAND, BOOL),
          lambda f: lambda h: [f(c) for c in h] == [f(c) for c in reversed(h)])
Primitive('is_run', arrow(HAND, BOOL), ...)

# Aggregates over suits
Primitive('max_count_suit', arrow(HAND, INT), ...)
```

### Current Depth Budget
The overnight run used max_depth=15, but:
- Enumeration at depth 15 with 65 primitives = astronomical search space
- Rules at depth 8+ are essentially unreachable
- Need either shallower programs OR much longer enumeration time

---

## Conclusion

The list primitives we added (take, drop, zip_with, adjacent_pairs, half_len) were **necessary but not sufficient**:

1. **Type gap**: `zip_with` returns `list(bool)` but we need `bool`. Missing `all_true`.
2. **Composition depth**: Even with the right primitives, programs are too deep (6-8+).
3. **Missing macros**: High-level operations like `first_half`, `is_palindrome_by` would dramatically reduce depth.

The same 8 rules were solved because they don't actually require the new primitives.
