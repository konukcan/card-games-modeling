# Theoretical Notes: Abstraction Learning in DreamCoder

*Aide-mémoire for resuming theoretical discussions – November 30, 2025*

---

## 1. Positional vs Non-Positional Functions

### The Core Distinction

| Aspect | Non-Positional (Aggregates) | Positional (Lists) |
|--------|-----------------------------|--------------------|
| **Data structure** | Bag / Multiset | List / Sequence |
| **Order matters?** | No | Yes |
| **Example operations** | count, sum, any, all | take, drop, at, zip |
| **Formal basis** | Set/multiset theory | Sequence/list theory |

### Current Primitives (Non-Positional)
```
count_suit, count_color      → counting occurrences
sum_ranks, max_rank, min_rank → numeric aggregation
n_unique_suits, n_unique_ranks → cardinality
all_same_suit, all_same_color → universal quantification
any (implicit in filter)      → existential quantification
```

### Missing Primitives (Positional)
```
take n xs   → first n elements
drop n xs   → all but first n elements
at n xs     → element at position n
reverse xs  → reverse order
zip_with f xs ys → combine element-wise with function f
```

### Why This Matters for Card Games

Many rules in the catalogue are inherently positional:
- **Halves rules**: Compare first 3 cards vs last 3 cards
- **Sorted rules**: Check if ranks are ascending/descending
- **Palindrome rules**: Compare position i with position (n-i)
- **Sequence patterns**: Adjacent cards must differ by 1

Without `take`, `drop`, `reverse`, and `zip_with`, these rules are **fundamentally inexpressible** in the current DSL.

---

## 2. Lambda Functions vs Formal Grammars

### When to Use Each

| Approach | Best For | Limitations |
|----------|----------|-------------|
| **Lambda calculus** | Compositional, aggregate operations | Awkward for pattern matching |
| **Formal grammars** | Positional patterns, sequences | Less compositional |

### Hybrid Approach for Card Games

For positional patterns like "sorted by rank", a pure lambda approach requires:
```
(λ all (zip_with le (map get_rank (take 5 $0))
                    (map get_rank (drop 1 $0))))
```

A grammar-based approach might express this more naturally:
```
Pattern: [r1 ≤ r2 ≤ r3 ≤ r4 ≤ r5 ≤ r6]
```

**Current recommendation**: Extend lambda calculus with list primitives first. Consider grammar-based patterns if enumeration remains intractable for positional rules.

---

## 3. Grounded vs Schematic Abstractions

### Grounded Abstractions
These have domain-specific meaning and contain concrete primitives.

**Example:**
```
#((λ eq (get_suit (head $0)) (get_suit (last $0))))
```
**Meaning:** "First and last cards have same suit"

**Characteristics:**
- Directly interpretable
- Transfer to related tasks
- Built from domain primitives

### Schematic Abstractions
These are pure combinators – higher-order patterns with no domain content.

**Example:**
```
#((λ (λ (λ (λ (λ $1 ($2 $3) $4))))))
```
**Meaning:** A combinator that takes 5 arguments and applies them as:
```
arg1(arg2(arg3), arg4)
```

**Characteristics:**
- Emerge from anti-unification of structurally similar programs
- Capture control flow, not domain knowledge
- Can be over-general (apply to many programs but add little compression)

### De Bruijn Indexing Refresher
```
$0 → innermost bound variable (most recent λ)
$1 → next outer
$2 → two lambdas out
...
```

**Example parsing:**
```
(λ (λ $1 $0))
     │  │ └─ second argument (inner λ)
     │  └─── first argument (outer λ)
     └────── applies $1 to $0
```

---

## 4. Compression Artifacts

### Why `#((λ n_unique_suits $0))` when we have `n_unique_suits`?

This is a **compression artifact**. The anti-unification algorithm extracts common subtrees from multiple programs. When a subtree like `(n_unique_suits $0)` appears frequently with a free variable, compression wraps it:

```
Original programs:
  (eq 2 (n_unique_suits hand))
  (le (n_unique_suits hand) 3)

Anti-unified pattern:
  (X (n_unique_suits hand))

Extracted abstraction:
  #((λ n_unique_suits $0))  ← wraps the pattern with its free variable
```

### When This Is Useful
- **Nested compositions**: `#((λ get_suit (head $0)))` captures "get suit of first card" as a reusable unit
- **Partial applications**: `#((λ eq 1))` captures "equals 1" as a predicate

### When This Is Wasteful
- **Trivial wrappers**: `#((λ n_unique_suits $0))` is equivalent to `n_unique_suits` for HAND→INT type
- **Overly general combinators**: `#((λ (λ $1)))` just selects first of two arguments

---

## 5. Anti-Unification Algorithm

### What It Does
Finds the **most specific generalization** of two programs.

### Example
```
Program A: (eq (get_suit (head hand)) HEARTS)
Program B: (eq (get_suit (last hand)) SPADES)

Anti-unification:
  (eq (get_suit (X hand)) Y)
  where X ∈ {head, last}, Y ∈ {HEARTS, SPADES}

Extracted abstraction:
  #((λ (λ (λ eq (get_suit ($1 $0)) $2))))

Usage:
  (abstraction head HEARTS hand)  → Program A
  (abstraction last SPADES hand)  → Program B
```

### Trade-offs
- More general → more reuse but less compression per use
- More specific → better compression but less reuse
- DreamCoder uses MDL (Minimum Description Length) to balance

---

## 6. Implications for Overnight Run

### Priority Primitives to Add
1. **`take`** and **`drop`** – essential for halves operations
2. **`reverse`** – needed for palindrome rules
3. **`zip_with`** – for element-wise comparison
4. **`all`** and **`any`** (if not present) – for universal/existential over lists

### Expected Impact
With list primitives added:
- ~25-30 of 49 currently unsolved rules should become expressible
- Halves rules, sorted rules, palindrome rules all unlock
- Compression may find powerful abstractions like "compare halves with property P"

### Curriculum Strategy
1. **Phase 1**: Easy rules (aggregates + simple comparisons)
2. **Phase 2**: Medium rules (introduce list primitives)
3. **Phase 3**: Hard rules (positional patterns, multi-step compositions)

---

## Questions for Future Investigation

1. **Should schematic abstractions be pruned?** They add grammar complexity but may not aid generalization.

2. **Can we seed the grammar with hand-crafted positional abstractions?** E.g., `first_half`, `second_half` as macros.

3. **Would a two-level system work better?** Formal grammar for structure, lambda for predicates.

4. **How does abstraction depth correlate with human difficulty?** Phase 6 solved mostly single-application rules.

---

*These notes capture the state of theoretical understanding as of Nov 30, 2025. To be expanded as experiments continue.*
