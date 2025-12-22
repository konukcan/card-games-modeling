# DreamCoder Alignment Plan

This document outlines the implementation plan to align our program synthesis system with the original DreamCoder (Ellis et al., 2021).

## Current State Assessment

### What We Have (Working)
- ✅ Type system with polymorphism and unification
- ✅ Program representation (Primitive, Application, Abstraction, Index, Invented)
- ✅ Grammar with type-indexed normalization
- ✅ Dirichlet prior for grammar updates (α=1, configurable)
- ✅ Top-down enumeration with hole-filling
- ✅ Neural recognition model (GRU-based)
- ✅ Basic compression via exact subtree matching
- ✅ Anti-unification for pattern discovery

### What's Missing or Incomplete
- ✅ Full MDL scoring for compression (COMPLETED - Phase 2)
- ✅ Program refactoring after learning abstractions (COMPLETED - Phase 1)
- ✅ Beam search over compression candidates (COMPLETED - Phase 3)
- ✅ Arity-aware abstraction search (COMPLETED - Phase 4)
- ✅ Grammar size penalty (COMPLETED - Phase 2, via grammar_weight parameter)
- ❌ Corpus-guided compression
- ❌ Recognition model integration with compression

---

## Implementation Phases

### Phase 1: Program Refactoring (Priority: CRITICAL) ✅ COMPLETED
**Status:** IMPLEMENTED (December 2024)
**Why First:** This is the foundation - without refactoring, other improvements have limited impact.

**Implementation Summary:**
- `rewrite_with_invention()` - fixed to NOT shift target in lambda bodies (syntactic matching)
- `rewrite_with_invention_detailed()` - returns detailed statistics
- `verify_rewrite_semantics()` - semantic equivalence checking
- `rewrite_frontier()` / `rewrite_all_frontiers()` - batch rewriting
- `compress_frontiers(refactor_programs=True)` - default behavior now rewrites programs
- `iterative_compression()` - passes rewritten frontiers between rounds
- Comprehensive test suite added to `compression.py`

#### 1.1 Enable Basic Refactoring in Compression Loop

**File:** `compression.py`

**Current State:**
```python
def compress_frontiers(...):
    # ... finds inventions, adds to grammar ...
    # Programs are NOT rewritten
    return CompressionResult(...)
```

**Target State:**
```python
def compress_frontiers(..., refactor_programs: bool = True):
    # ... finds inventions, adds to grammar ...

    if refactor_programs:
        # Rewrite ALL programs with new invention
        rewritten_frontiers = []
        for frontier in frontiers:
            rewritten = []
            for prog, ll in frontier:
                new_prog = rewrite_with_invention(prog, target, invention, n_args)
                rewritten.append((new_prog, ll))
            rewritten_frontiers.append(rewritten)
        frontiers = rewritten_frontiers

    return CompressionResult(..., rewritten_frontiers=rewritten_frontiers)
```

**Implementation Steps:**

```
Step 1.1.1: Fix rewrite_with_invention() edge cases
  - Handle nested lambdas correctly (de Bruijn shifting)
  - Handle multiple occurrences of target in same program
  - Add semantic preservation tests

Step 1.1.2: Update CompressionResult dataclass
  - Add field: rewritten_frontiers: List[List[Tuple[Program, float]]]
  - Add field: rewrite_map: Dict[str, str]  # old_prog_str -> new_prog_str

Step 1.1.3: Integrate rewriting into compress_frontiers()
  - After each invention is selected, rewrite all programs
  - Pass rewritten programs to subsequent invention search
  - This enables finding hierarchical patterns

Step 1.1.4: Update iterative_compression() to use rewritten programs
  - Pass rewritten_frontiers to next round
  - Track cumulative rewrites across rounds
```

**Tests to Add:**
```python
def test_rewrite_preserves_semantics():
    """Rewritten program evaluates to same result."""

def test_rewrite_handles_nested_lambdas():
    """Correctly shifts de Bruijn indices."""

def test_rewrite_multiple_occurrences():
    """All occurrences of pattern are replaced."""

def test_hierarchical_abstraction():
    """Round 2 finds patterns using Round 1 abstractions."""
```

---

#### 1.2 Add Semantic Verification

**File:** `compression.py`

**Purpose:** Ensure rewrites don't change program behavior.

```python
def verify_rewrite(original: Program, rewritten: Program,
                   test_inputs: List[Any]) -> bool:
    """
    Verify that rewritten program has same behavior as original.

    Args:
        original: Original program
        rewritten: Program after rewriting with invention
        test_inputs: Sample inputs to test on

    Returns:
        True if all outputs match
    """
    for inp in test_inputs:
        try:
            orig_result = original.evaluate([inp])
            new_result = rewritten.evaluate([inp])
            if orig_result != new_result:
                return False
        except Exception:
            # If either fails, they should both fail
            pass
    return True
```

---

### Phase 2: Full MDL Scoring (Priority: HIGH) ✅ COMPLETED
**Status:** IMPLEMENTED (December 2024)
**Why Second:** Principled scoring enables better abstraction selection.

**Implementation Summary:**
- `grammar_description_length()` in Grammar class - measures grammar complexity
- `_type_description_length()` helper - measures type complexity
- `compute_mdl()` - full MDL objective: λ × DL(grammar) + Σ DL(programs)
- `compute_mdl_detailed()` - detailed breakdown of MDL components
- `evaluate_invention_mdl()` - principled decision function for inventions
- `rank_inventions_by_mdl()` - rank candidates by MDL improvement
- `compress_frontiers_mdl()` - full MDL-based compression function
- `grammar_weight` parameter controls complexity/accuracy trade-off
- Comprehensive test suite for all MDL functions

#### 2.1 Add Grammar Description Length

**File:** `grammar.py`

```python
def grammar_description_length(self) -> float:
    """
    Compute description length of the grammar itself.

    DL(grammar) = Σ DL(production_i)

    Each production's DL includes:
    - Type complexity (number of type constructors)
    - Body complexity (for Invented: size of body)
    """
    total_dl = 0.0

    for prod in self.productions:
        # Type complexity: count type constructors
        type_dl = self._type_description_length(prod.tp)

        # Body complexity (only for Invented)
        if isinstance(prod.program, Invented):
            body_dl = prod.program.body.size()
        else:
            body_dl = 1  # Primitives have fixed cost

        total_dl += type_dl + body_dl

    return total_dl

def _type_description_length(self, tp: Type) -> float:
    """Count type constructors as proxy for type complexity."""
    if isinstance(tp, BaseType):
        return 1.0
    elif isinstance(tp, Arrow):
        return 1.0 + self._type_description_length(tp.arg) + \
                     self._type_description_length(tp.ret)
    elif isinstance(tp, ListType):
        return 1.0 + self._type_description_length(tp.elem)
    elif isinstance(tp, TypeVariable):
        return 0.5  # Type variables are "free"
    return 1.0
```

#### 2.2 Implement Full MDL Objective

**File:** `compression.py`

```python
def compute_mdl(grammar: Grammar, programs: List[Program],
                request_type: Type, grammar_weight: float = 1.0) -> float:
    """
    Compute full MDL objective.

    MDL = λ × DL(grammar) + Σ DL(program_i | grammar)

    Args:
        grammar: Current grammar
        programs: Programs to evaluate
        request_type: Type of programs
        grammar_weight: λ parameter (how much to penalize grammar complexity)

    Returns:
        Total MDL score (lower is better)
    """
    grammar_dl = grammar.grammar_description_length()
    programs_dl = sum(grammar.description_length(p, request_type) for p in programs)

    return grammar_weight * grammar_dl + programs_dl


def evaluate_invention_mdl(
    grammar: Grammar,
    programs: List[Program],
    invention: Invented,
    target: Program,
    n_args: int,
    request_type: Type,
    grammar_weight: float = 1.0
) -> Tuple[float, float, List[Program]]:
    """
    Evaluate MDL change from adding an invention.

    Returns:
        (old_mdl, new_mdl, rewritten_programs)
    """
    # Current MDL
    old_mdl = compute_mdl(grammar, programs, request_type, grammar_weight)

    # Create new grammar with invention
    ctx = TypeContext()
    tp = invention.infer_type(ctx, [])
    new_grammar = grammar.with_production(Production(invention, tp, 0.0))
    new_grammar = new_grammar.normalize_probabilities()

    # Rewrite programs
    rewritten = [rewrite_with_invention(p, target, invention, n_args)
                 for p in programs]

    # New MDL
    new_mdl = compute_mdl(new_grammar, rewritten, request_type, grammar_weight)

    return old_mdl, new_mdl, rewritten
```

#### 2.3 Update compress_frontiers() to Use MDL

```python
def compress_frontiers_mdl(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    request_type: Type,
    max_inventions: int = 5,
    grammar_weight: float = 1.0,
    min_mdl_improvement: float = 1.0
) -> CompressionResult:
    """
    Compression using full MDL scoring.
    """
    all_programs = [p for frontier in frontiers for p, _ in frontier]

    # Find candidate subtrees
    common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    new_inventions = []
    current_grammar = grammar
    current_programs = all_programs

    for occ in common:
        if len(new_inventions) >= max_inventions:
            break

        invention, n_args = abstract_subtree(occ.subtree)

        # Evaluate MDL change
        old_mdl, new_mdl, rewritten = evaluate_invention_mdl(
            current_grammar, current_programs, invention,
            occ.subtree, n_args, request_type, grammar_weight
        )

        mdl_improvement = old_mdl - new_mdl

        if mdl_improvement >= min_mdl_improvement:
            # Accept this invention
            new_inventions.append(invention)
            current_grammar = current_grammar.with_invented(invention)
            current_programs = rewritten

    return CompressionResult(...)
```

---

### Phase 3: Beam Search (Priority: MEDIUM) ✅ COMPLETED
**Status:** IMPLEMENTED (December 2024)
**Why Third:** Avoids local optima once we have good scoring.

**Implementation Summary:**
- `CompressionState` dataclass - tracks grammar, programs, inventions, MDL, and history
- `beam_search_compression()` - main beam search function with configurable beam width
- `beam_search_compression_with_arity()` - convenience wrapper combining Phase 3 + Phase 4
- State deduplication to avoid redundant exploration
- Comprehensive statistics tracking (states explored, candidates evaluated, MDL history)
- Integration with arity-aware search via `use_arity_search` parameter

#### 3.1 Define Search State

**File:** `compression.py`

```python
@dataclass
class CompressionState:
    """State in beam search over compressions."""
    grammar: Grammar
    programs: List[Program]          # Current (possibly rewritten) programs
    inventions: List[Invented]       # Inventions added so far
    mdl: float                       # Current MDL score
    history: List[str]               # For debugging: what happened

    def __lt__(self, other):
        """For heap: lower MDL is better."""
        return self.mdl < other.mdl
```

#### 3.2 Implement Beam Search

```python
def beam_search_compression(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    request_type: Type,
    beam_width: int = 10,
    max_inventions: int = 5,
    grammar_weight: float = 1.0
) -> CompressionResult:
    """
    Beam search over compression choices.

    Maintains beam_width best states, explores adding each candidate
    invention to each state, keeps best results.
    """
    all_programs = [p for frontier in frontiers for p, _ in frontier]

    # Initial state
    initial_mdl = compute_mdl(grammar, all_programs, request_type, grammar_weight)
    initial_state = CompressionState(
        grammar=grammar,
        programs=all_programs,
        inventions=[],
        mdl=initial_mdl,
        history=[]
    )

    beam = [initial_state]

    for iteration in range(max_inventions):
        candidates = []

        for state in beam:
            # Find candidate inventions for this state
            subtrees = find_common_subtrees(state.programs, min_size=2, min_count=2)

            for occ in subtrees[:20]:  # Limit candidates per state
                invention, n_args = abstract_subtree(occ.subtree)

                # Skip if already in this state's grammar
                if state.grammar.get_production(invention) is not None:
                    continue

                # Evaluate
                old_mdl, new_mdl, rewritten = evaluate_invention_mdl(
                    state.grammar, state.programs, invention,
                    occ.subtree, n_args, request_type, grammar_weight
                )

                if new_mdl < state.mdl:  # Improvement
                    new_state = CompressionState(
                        grammar=state.grammar.with_invented(invention),
                        programs=rewritten,
                        inventions=state.inventions + [invention],
                        mdl=new_mdl,
                        history=state.history + [f"Added {invention}"]
                    )
                    candidates.append(new_state)

        if not candidates:
            break

        # Keep top beam_width states
        candidates.sort(key=lambda s: s.mdl)
        beam = candidates[:beam_width]

    # Return best state
    best = min(beam, key=lambda s: s.mdl)
    return CompressionResult(
        new_inventions=best.inventions,
        old_grammar=grammar,
        new_grammar=best.grammar.normalize_probabilities(),
        total_savings=initial_mdl - best.mdl,
        subtree_analysis=[]
    )
```

---

### Phase 4: Arity-Aware Search (Priority: MEDIUM) ✅ COMPLETED
**Status:** IMPLEMENTED (December 2024)
**Why Fourth:** Better abstractions once we have good search.

**Implementation Summary:**
- `Factorization` dataclass - represents a specific way to abstract a subtree
- `enumerate_factorizations()` - generates all valid arities for a subtree
- `abstract_subtree_partial()` - creates abstraction over subset of free variables
- `best_factorization()` - picks optimal arity by MDL improvement
- `rank_factorizations_by_mdl()` - ranks all factorizations for debugging/analysis
- Handles edge cases: no free vars, too many free vars, invalid subsets

#### 4.1 Enumerate Factorizations

**File:** `compression.py`

```python
def enumerate_factorizations(
    subtree: Program,
    max_args: int = 4
) -> List[Tuple[Invented, int, Set[int]]]:
    """
    Enumerate different ways to abstract a subtree.

    For a subtree with free variables {0, 2, 3}, we could create:
    - 3-arg: abstract over all three
    - 2-arg: abstract over any two, inline the third
    - 1-arg: abstract over any one
    - 0-arg: if subtree has subexpressions that could become args

    Returns:
        List of (invention, n_args, which_vars_abstracted)
    """
    free_vars = subtree.free_indices()

    if len(free_vars) > max_args:
        # Too many - only consider full abstraction
        inv, n = abstract_subtree(subtree, free_vars)
        return [(inv, n, free_vars)]

    factorizations = []

    # Consider all subsets of free variables
    from itertools import combinations

    for r in range(len(free_vars), 0, -1):  # From all vars down to 1
        for subset in combinations(sorted(free_vars), r):
            subset_set = set(subset)

            # Create abstraction over just this subset
            inv, n = abstract_subtree_partial(subtree, subset_set)
            if inv is not None:
                factorizations.append((inv, n, subset_set))

    # Also consider abstracting over subexpressions
    for subexpr in subtree.subprograms():
        if subexpr.size() >= 2 and subexpr != subtree:
            # Could abstract over this subexpression
            aug_free = free_vars | {-1}  # -1 = "subexpression becomes arg"
            inv = create_subexpr_abstraction(subtree, subexpr)
            if inv is not None:
                factorizations.append((inv, len(free_vars) + 1, aug_free))

    return factorizations


def abstract_subtree_partial(
    subtree: Program,
    vars_to_abstract: Set[int]
) -> Tuple[Optional[Invented], int]:
    """
    Abstract over only some free variables.

    Variables not in vars_to_abstract remain free (will be captured
    from context when abstraction is used).
    """
    all_free = subtree.free_indices()

    if not vars_to_abstract.issubset(all_free):
        return None, 0

    # Reindex only the vars we're abstracting
    vars_list = sorted(vars_to_abstract)
    index_map = {old: new for new, old in enumerate(vars_list)}

    rewritten = _reindex_partial(subtree, index_map)

    body = rewritten
    for _ in range(len(vars_list)):
        body = Abstraction(body)

    return Invented(body), len(vars_list)
```

#### 4.2 Score Factorizations by MDL

```python
def best_factorization(
    subtree: Program,
    grammar: Grammar,
    programs: List[Program],
    request_type: Type,
    grammar_weight: float = 1.0
) -> Tuple[Invented, int, float]:
    """
    Find the best factorization of a subtree by MDL.

    Returns:
        (best_invention, n_args, mdl_improvement)
    """
    factorizations = enumerate_factorizations(subtree)

    best_inv = None
    best_n_args = 0
    best_improvement = 0.0

    for inv, n_args, _ in factorizations:
        old_mdl, new_mdl, _ = evaluate_invention_mdl(
            grammar, programs, inv, subtree, n_args,
            request_type, grammar_weight
        )
        improvement = old_mdl - new_mdl

        if improvement > best_improvement:
            best_improvement = improvement
            best_inv = inv
            best_n_args = n_args

    return best_inv, best_n_args, best_improvement
```

---

### Phase 5: Corpus-Guided Compression (Priority: LOW)
**Why Last:** Requires recognition model to be well-trained first.

#### 5.1 Score Inventions by Predicted Usefulness

**File:** `compression.py` (with `neural_recognition.py` integration)

```python
def score_invention_by_recognition(
    invention: Invented,
    recognition_model: RecognitionModel,
    unsolved_tasks: List[Task],
    grammar: Grammar
) -> float:
    """
    Score an invention by how useful the recognition model thinks it is.

    Intuition: If the recognition model predicts this invention for
    unsolved tasks, it's probably useful.
    """
    # Add invention temporarily to grammar
    temp_grammar = grammar.with_invented(invention)

    # Get recognition model predictions for unsolved tasks
    total_score = 0.0

    for task in unsolved_tasks:
        # Get task embedding and predict primitives
        predictions = recognition_model.predict_primitives(task, temp_grammar)

        # Score = predicted probability for this invention
        inv_key = str(invention)
        if inv_key in predictions:
            total_score += predictions[inv_key]

    return total_score / len(unsolved_tasks)


def corpus_guided_compression(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    unsolved_tasks: List[Task],
    recognition_model: RecognitionModel,
    request_type: Type,
    max_inventions: int = 5,
    mdl_weight: float = 0.7,
    recognition_weight: float = 0.3
) -> CompressionResult:
    """
    Compression guided by both MDL and recognition model predictions.

    Combined score = mdl_weight × MDL_improvement +
                     recognition_weight × recognition_score
    """
    all_programs = [p for frontier in frontiers for p, _ in frontier]
    common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    # Score each candidate
    scored_candidates = []

    for occ in common:
        invention, n_args = abstract_subtree(occ.subtree)

        # MDL score
        old_mdl, new_mdl, _ = evaluate_invention_mdl(
            grammar, all_programs, invention, occ.subtree,
            n_args, request_type
        )
        mdl_improvement = old_mdl - new_mdl

        # Recognition score
        recog_score = score_invention_by_recognition(
            invention, recognition_model, unsolved_tasks, grammar
        )

        # Combined score
        combined = mdl_weight * mdl_improvement + recognition_weight * recog_score

        scored_candidates.append((invention, n_args, occ.subtree, combined))

    # Select top inventions
    scored_candidates.sort(key=lambda x: -x[3])  # Higher is better

    # ... rest of selection logic ...
```

---

## Implementation Order and Dependencies

```
Phase 1: Program Refactoring
    ├── 1.1 Fix rewrite_with_invention()     [No dependencies]
    ├── 1.2 Update CompressionResult         [No dependencies]
    ├── 1.3 Integrate into compress_frontiers [Depends on 1.1, 1.2]
    └── 1.4 Update iterative_compression     [Depends on 1.3]

Phase 2: Full MDL Scoring
    ├── 2.1 Add grammar_description_length() [No dependencies]
    ├── 2.2 Implement compute_mdl()          [Depends on 2.1]
    └── 2.3 Update compress_frontiers()      [Depends on 2.2, Phase 1]

Phase 3: Beam Search
    ├── 3.1 Define CompressionState          [No dependencies]
    └── 3.2 Implement beam_search_compression [Depends on 3.1, Phase 2]

Phase 4: Arity-Aware Search
    ├── 4.1 enumerate_factorizations()       [No dependencies]
    └── 4.2 best_factorization()             [Depends on 4.1, Phase 2]

Phase 5: Corpus-Guided
    ├── 5.1 score_invention_by_recognition() [Depends on recognition model]
    └── 5.2 corpus_guided_compression()      [Depends on 5.1, Phase 2]
```

---

## Testing Strategy

### Unit Tests

```python
# test_compression.py

class TestProgramRefactoring:
    def test_rewrite_simple_substitution(self):
        """Replace (+ 1 1) with #add11."""

    def test_rewrite_preserves_semantics(self):
        """Rewritten program evaluates identically."""

    def test_rewrite_nested_lambdas(self):
        """De Bruijn indices shift correctly."""

    def test_rewrite_multiple_occurrences(self):
        """All occurrences replaced."""


class TestMDLScoring:
    def test_grammar_dl_increases_with_inventions(self):
        """Adding abstractions increases grammar DL."""

    def test_mdl_improvement_from_common_pattern(self):
        """Abstracting common code reduces total MDL."""

    def test_mdl_rejects_useless_abstraction(self):
        """Rare pattern increases MDL (grammar cost > savings)."""


class TestBeamSearch:
    def test_beam_finds_better_than_greedy(self):
        """Beam search finds solution greedy misses."""

    def test_beam_respects_width(self):
        """Never more than beam_width states."""


class TestAritySearch:
    def test_enumerate_all_factorizations(self):
        """Generates all valid factorizations."""

    def test_best_factorization_by_mdl(self):
        """Selects factorization with best MDL."""
```

### Integration Tests

```python
class TestFullCompressionPipeline:
    def test_hierarchical_abstraction(self):
        """
        Given programs that share nested structure:
        - Round 1 finds inner pattern
        - Round 2 finds outer pattern using inner
        """

    def test_compression_improves_enumeration(self):
        """
        After compression:
        - Programs using abstractions are found faster
        - Description lengths decrease
        """
```

---

## Estimated Effort

| Phase | Effort | Files Modified |
|-------|--------|----------------|
| Phase 1: Refactoring | 2-3 days | compression.py |
| Phase 2: MDL Scoring | 2-3 days | grammar.py, compression.py |
| Phase 3: Beam Search | 1-2 days | compression.py |
| Phase 4: Arity Search | 2-3 days | compression.py |
| Phase 5: Corpus-Guided | 3-4 days | compression.py, neural_recognition.py |

**Total: ~10-15 days of focused work**

---

## Success Metrics

After implementation, we should see:

1. **Hierarchical abstractions**: Compression finds patterns that use previously learned patterns

2. **Better abstraction selection**: MDL-based selection avoids useless abstractions

3. **Improved enumeration speed**: Programs using abstractions found in fewer steps

4. **Reduced description lengths**: Total DL decreases across iterations

5. **More solved tasks**: Abstractions enable solving previously-hard tasks

---

## References

- Ellis, K., et al. (2021). DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. PLDI.
- Original implementation: https://github.com/ellisk42/ec
