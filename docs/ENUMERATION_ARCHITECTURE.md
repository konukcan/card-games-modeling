# Enumeration Architecture Documentation

> **Generated**: January 2026
> **Purpose**: Comprehensive reference for the DreamCoder-style program enumeration system

---

## Overview

The enumeration system is the "WAKE" phase of DreamCoder's wake-sleep learning loop. It systematically searches through the space of all possible programs to find ones that solve given tasks (defined by input-output examples).

### File Architecture

```
dreamcoder_core/
├── type_system.py      # Type definitions and unification (667 lines)
├── program.py          # AST representation (2134 lines)
├── grammar.py          # PCFG for program synthesis (1130 lines)
├── enumeration.py      # Main enumeration algorithms (2111 lines)
├── enumeration_optimized.py  # Multi-processing variants (598 lines)
├── enumeration_worker.py     # PyPy worker for parallel execution (150 lines)
├── lean_primitives.py        # DSL primitives (780 lines)
└── wake_sleep.py             # Full learning loop integration (539 lines)
```

---

## 1. TYPE SYSTEM (`type_system.py`)

The foundation enabling type-safe program synthesis.

### Type Hierarchy

```
Type (Abstract Base)
├── BaseType(name)        # Ground types: bool, int, card, suit, rank
├── TypeVariable(id)      # Polymorphic variables: 'a, 'b, 'c...
├── Arrow(arg, ret)       # Function types: arg -> ret
└── ListType(element)     # List types: list(element)
```

### Key Classes

#### `BaseType` (lines 165-203)
Ground types with no parameters.
```python
BOOL = BaseType('bool')
INT = BaseType('int')
CARD = BaseType('card')
SUIT = BaseType('suit')
RANK = BaseType('rank')
HAND = ListType(CARD)  # A hand is list(card)
```

#### `TypeVariable` (lines 209-271)
Polymorphic type placeholders for generic functions like `map : ('a -> 'b) -> list('a) -> list('b)`.
- Uses integer IDs: id=0 → 'a, id=1 → 'b
- Implements `apply_substitution()` with cycle detection

#### `Arrow` (lines 278-321)
Function types that are **curried and right-associative**:
```python
int -> int -> bool  means  int -> (int -> bool)
# A function taking two ints, returning bool
```

#### `TypeContext` (lines 425-580)
The workhorse managing type inference:
- `fresh_type_variable()`: Generate unique type variables
- `instantiate(t)`: Replace all type variables with fresh ones (for polymorphism)
- `unify(t1, t2)`: Find substitution making types equal (Robinson's algorithm)
- `apply(t)`: Resolve bound variables through substitution chain

### Key Algorithm: Robinson's Unification (lines 480-545)
```
1. Apply current substitution to both types
2. If equal, done
3. If one is a variable, bind it (with occurs check)
4. If both are compound (Arrow, List), unify components
5. Otherwise, fail with UnificationError
```

The **occurs check** prevents infinite types: can't unify 'a with list('a).

---

## 2. PROGRAM AST (`program.py`)

The representation for synthesized programs.

### AST Node Types

```
Program (Abstract Base)
├── Primitive(name, tp, value)      # Built-in operation: +, get_suit, etc.
├── Index(i)                        # De Bruijn variable: $0, $1, $2
├── Application(func, arg)          # Function application: (f x)
├── Abstraction(body)               # Lambda: λ.body
├── Invented(body)                  # Learned abstractions from compression
└── Hole(tp, id)                    # Typed placeholder for top-down synthesis
```

### De Bruijn Indices

Variables are represented numerically, relative to their binding lambda:
```
λx. λy. x + y  =  λ. λ. ($1 + $0)
     │    │           │    │
     └────┼───────────┘    │
          └────────────────┘
```
- `$0` = innermost bound variable
- `$1` = next outer
- No variable capture issues during substitution

### Key Methods (all AST nodes)

| Method | Purpose |
|--------|---------|
| `evaluate(env)` | Execute program with environment stack |
| `infer_type(ctx)` | Deduce type using unification |
| `size()` | Count AST nodes |
| `depth()` | Maximum nesting level |
| `free_indices()` | Find unbound variables |
| `shift(amount, cutoff)` | Adjust de Bruijn indices |
| `substitute(index, value)` | Beta reduction helper |

### Hole Utilities (lines ~1900-2100)

Critical for top-down enumeration:
- `has_holes(prog)`: Check if program has unfilled holes
- `find_first_hole(prog)`: Get leftmost hole for expansion
- `substitute_hole(prog, hole_id, value)`: Fill a specific hole
- `count_holes(prog)`: Total holes remaining
- `reset_hole_counter()`: Reset global hole ID counter

---

## 3. GRAMMAR (`grammar.py`)

The Probabilistic Context-Free Grammar (PCFG) assigning probabilities to derivations.

### Core Data Structures

#### `Production` (dataclass)
```python
@dataclass
class Production:
    program: Program    # The primitive/invention
    log_probability: float  # log P(choosing this)
```

#### `Grammar` class (lines 100-600)
```python
Grammar:
    productions: List[Production]  # All available programs
    log_variable: float            # log P(using a bound variable)
    _type_index: Dict[str, List[...]]  # Fast lookup by type skeleton
```

### Key Methods

#### `candidates_for_type(tp, ctx, env)` (lines ~250-350)
Find all productions that can fill a hole of type `tp`:
1. Look up type skeleton in index (O(1))
2. For each candidate, instantiate fresh type variables
3. Attempt unification with target type
4. Return compatible productions with their instantiated types

#### `variable_candidates(tp, ctx, env)` (lines ~360-400)
Find bound variables (de Bruijn indices) that match the target type:
1. Iterate through environment (list of types at each index)
2. Unify each with target
3. Return matching indices with their log probabilities

#### `program_log_likelihood(prog)` (lines ~420-500)
Compute P(program | grammar):
- Sum log probabilities of all productions used
- Add log_variable for each variable reference
- Type-indexed normalization (normalize per return type)

#### `sample(tp)` (lines ~520-600)
Stochastically sample a random program:
- Used in "dreaming" to generate training data
- Weighted by grammar probabilities

#### `inside_outside_update(frontiers)` (lines ~650-750)
Bayesian grammar update using solved task programs:
- Count usage of each production across all solutions
- Apply Dirichlet prior (α = 1 for Laplace smoothing)
- Normalize to get new probabilities

---

## 4. MAIN ENUMERATION (`enumeration.py`)

The core enumeration algorithms.

### Enumeration Strategies

The module provides **three strategies**:

| Strategy | Class/Function | Use Case |
|----------|----------------|----------|
| **Memoized (default)** | `TopDownEnumerator.enumerate_memoized()` | Production (1000-8000x faster) |
| **Priority Queue** | `TopDownEnumerator.enumerate_priority_queue()` | Legacy, edge cases |
| **Iterative Deepening** | `enumerate_simple()` | Workers, backward compat |

### Data Structures

#### `EnumerationResult` (lines 108-128)
```python
@dataclass
class EnumerationResult:
    program: Program
    log_probability: float       # P(program | grammar) in log-space
    log_likelihood: float        # 0 = perfect, negative = partial
    description_length: float    # MDL = -log_prob / log(2) bits
    programs_enumerated: int
    partial_programs_explored: int
    time_seconds: float
```

#### `Frontier` (lines 131-173)
Top-k solutions for a task:
- `add(result)`: Add if improves frontier
- `best`: Best solution by description length
- `solved`: Has at least one perfect solution?
- `n_solutions`: Count of perfect solutions

#### `PriorityItem` (lines 176-190)
Priority queue entry:
```python
@dataclass(order=True)
class PriorityItem:
    priority: float          # Cost = -log_probability (min-heap)
    program: Program         # Partial program with holes
    tp: Type                 # Type being synthesized
    env: List[Type]          # Type environment for bound variables
    hole_contexts: Dict[...]  # [CGN-INTEGRATION] For context-aware guidance
```

### TopDownEnumerator Class (lines 197-1476)

The main enumerator using hole-filling.

#### `__init__()` (lines 224-261)
```python
TopDownEnumerator(
    grammar: Grammar,
    max_depth: int = 8,
    max_programs: int = 100000,
    # [CGN-INTEGRATION] Optional contextual grammar
    contextual_grammar: Optional[Any] = None,
    task_embedding: Optional[Any] = None,
    contextual_weight: float = 0.5
)
```

#### `enumerate()` (lines 312-343) - DEFAULT ENTRY POINT
Delegates to memoized enumeration (1000-8000x faster at depth 7-8).

#### `enumerate_memoized()` (lines 1028-1101)
**DreamCoder-style dynamic programming with iterative cost deepening:**
```
1. For each cost level (0, 1, 2, ...):
   a. Build all programs of cost ≤ current using cached subproblems
   b. Yield any new complete programs
   c. Increment cost and repeat
2. Cache by (type, env, cost_bucket) to avoid re-enumeration
```

#### `_enumerate_type_at_cost()` (lines 1103-1197)
Core memoized enumeration:
```python
def _enumerate_type_at_cost(tp, env, max_cost, depth_remaining, ...):
    # CASE 1: Arrow type → enumerate lambdas
    if isinstance(tp, Arrow):
        for body in enumerate(body_type, [arg_type] + env, ...):
            yield Abstraction(body)

    # CASE 2: Base type → check cache or enumerate
    cache_key = (str(tp), env_tuple, cost_bucket)
    if cache_key in cache:
        yield from cache[cache_key]
    else:
        # Variables
        for idx in variable_candidates(tp, env):
            results.append(Index(idx))

        # Productions
        for prod in candidates_for_type(tp):
            if needs_args:
                for applied in apply_args(prod, ...):
                    results.append(applied)
            else:
                results.append(prod.program)

        cache[cache_key] = results
```

#### `enumerate_priority_queue()` (lines 345-410)
Legacy priority-queue approach:
```
1. Start with Hole(request_type) at cost 0
2. While queue not empty:
   a. Pop lowest-cost partial program
   b. If complete (no holes), yield it
   c. Otherwise, expand first hole with all productions
   d. Push new partial programs to queue
```

#### `_expand_first_hole()` (lines 412-580)
Expand a hole with all valid productions:
```
1. Find first hole and its type
2. CASE 1: Arrow type → create λ.Hole(body_type)
3. CASE 2: Productions → for each matching production:
   - If needs args, apply with holes
   - Substitute for hole, push to queue
4. CASE 3: Variables → substitute Index for hole
```

**Predictive Depth Pruning** (Option B fix):
```python
def can_complete(new_prog):
    n_holes = count_holes(new_prog)
    current_depth = new_prog.depth()
    min_additional_depth = max(0, (n_holes - 1) // 2)
    return current_depth + min_additional_depth <= max_depth
```

#### `enumerate_with_cost_bands()` (lines 647-849)
Iterative cost deepening with persistent queue:
- Single queue across all bands (not restarted)
- Process items band by band
- Defer high-cost items to later bands

### [CGN-INTEGRATION] Context-Aware Enumeration (lines 263-640)

Support for `ContextualGrammarNetwork` that predicts primitives based on context:

```python
def get_contextual_log_prob(prim_name, parent_name, position, base_log_prob):
    if contextual_grammar is None:
        return base_log_prob

    contextual_log_prob = contextual_grammar.predict_for_context(
        task_embedding, parent_name, position
    )[prim_idx]

    # Blend: (1-w)*base + w*contextual
    return (1 - contextual_weight) * base_log_prob + contextual_weight * contextual_log_prob
```

### enumerate_simple() - LEGACY (lines 1842-1929)

Iterative deepening by depth (not cost):
```
for depth in range(1, max_depth + 1):
    for prog in enumerate_at_depth(depth):
        yield prog
```
- **DEPRECATED** for main code paths
- Kept for PyPy workers and backward compatibility
- Ignores grammar probabilities (depth order only)

---

## 5. OPTIMIZED ENUMERATION (`enumeration_optimized.py`)

Variants with multiprocessing and early pruning.

### LikelihoodMode (enum)
```python
class LikelihoodMode(Enum):
    ALL_OR_NOTHING = "all_or_nothing"  # Only 100% correct programs
    RELAXED = "relaxed"                 # Keep partial solutions too
```

### Key Optimization: Early Pruning

In ALL_OR_NOTHING mode, stop evaluating examples as soon as one fails:
```python
for inp, expected in examples:
    result = eval_fn(program, inp)
    if result != expected:
        break  # Early exit!
```

### `enumerate_for_task_optimized()` (lines 204-318)

Main function with early pruning:
- Uses `enumerate_simple()` for enumeration
- Applies likelihood mode logic
- Returns `TaskFrontier` with best solutions

### `enumerate_tasks_parallel()` (lines 406-492)

Multiprocessing across tasks:
```python
with mp.Pool(processes=n_workers) as pool:
    results = pool.map(_worker_enumerate_task, worker_args)
```

### `enumerate_tasks_sequential_optimized()` (lines 499-581)

Sequential with optimizations, drop-in replacement for overnight runs.

---

## 6. WORKER (`enumeration_worker.py`)

PyPy-compatible worker for parallel enumeration.

### Protocol
- **Input**: JSON task specification via stdin
- **Output**: JSON results via stdout
- Designed to run in PyPy for performance

### Key Functions

#### `deserialize_hand(data)` (lines ~30-50)
Convert JSON → Card objects:
```python
def deserialize_hand(data):
    return [Card(Suit[card_data['suit']], Rank[card_data['rank']])
            for card_data in data]
```

#### `evaluate_program(prog_str, hand, grammar)` (lines ~55-80)
Safe program evaluation with error handling.

#### `enumerate_task(task_data, grammar)` (lines ~85-145)
Main enumeration with early pruning:
- Stop evaluating examples as soon as one fails
- Stop after finding 5 solutions
- Respect max_programs and timeout

---

## 7. PRIMITIVES (`lean_primitives.py`)

The Domain-Specific Language for card games.

### Design Philosophy
**Cognitive Realism**: Primitives should be "directly nameable" - expressible in short natural language phrases that humans use.

### Primitive Categories (59 total)

| Category | Count | Examples |
|----------|-------|----------|
| Constants | 14 | CLUBS, DIAMONDS, RED, BLACK, 0-5, true, false |
| Card Accessors | 4 | get_suit, get_rank, rank_val, get_color |
| Position Ops | 5 | head, last, at, length, reverse |
| List Slicing | 7 | take, drop, zip_with, adjacent_pairs, first_half, second_half |
| Direct Queries | 9 | has_suit, has_color, count_suit, all_same_suit, n_unique_suits |
| Aggregates | 3 | sum_ranks, max_rank, min_rank |
| Comparisons | 5 | eq, lt, le, gt, ge |
| Boolean Ops | 4 | and, or, not, if |
| Higher-Order | 5 | map, filter, all, any, unique |
| Arithmetic | 3 | +, -, mod |

### Key Design Decisions

**Removed** (low cognitive reality):
- compose, flip, const, id (abstract combinators)
- cons, nil (list construction)
- Rank constants 10-14, game thresholds 17/21

**Added** (high cognitive reality):
- Direct queries: `has_suit`, `all_same_suit`, `n_unique_suits`
- Aggregates: `sum_ranks`, `max_rank`, `min_rank`
- Halves: `first_half`, `second_half`

### Build Functions

```python
build_lean_primitives() -> List[Primitive]  # Get all 59 primitives
build_lean_grammar() -> Grammar             # Uniform grammar over primitives
```

---

## 8. WAKE-SLEEP LOOP (`wake_sleep.py`)

The full DreamCoder learning cycle.

### Learning Phases

```
1. WAKE (Enumeration): Search for programs that solve tasks
2. SLEEP (Compression): Extract common abstractions into library
3. SLEEP (Recognition): Train neural network to guide search [optional]
4. Iterate until convergence or budget exhausted
```

### Key Classes

#### `IterationResult` (lines 49-62)
```python
@dataclass
class IterationResult:
    iteration: int
    tasks_solved: int
    total_tasks: int
    new_inventions: List[Invented]
    programs_enumerated: int
    time_seconds: float
    frontiers: Dict[str, Frontier]
```

#### `LearningMetrics` (lines 65-91)
Per-task tracking:
- When solved, how many programs tried
- Solution size and description length
- Time to solve

#### `DreamCoderResult` (lines 94-150)
Complete run results with summary generation and JSON export.

### DreamCoder Class (lines 153-404)

#### `__init__()` (lines 163-204)
```python
DreamCoder(
    grammar: Grammar,          # Initial primitives
    tasks: List[Task],         # Tasks to learn
    eval_fn: Callable,         # How to run programs
    max_iterations: int = 10,
    enumeration_timeout: float = 30.0,
    enumeration_budget: int = 10000,
    max_depth: int = 6,
    compress_every: int = 1,   # Run compression every N iterations
    verbose: bool = True
)
```

#### `run()` (lines 206-258)
Main learning loop:
```python
for iteration in range(max_iterations):
    iter_result = _run_iteration(iteration)

    if iter_result.tasks_solved == total_tasks:
        break  # All solved, stop early
```

#### `_run_iteration()` (lines 260-325)
Single wake-sleep cycle:
```python
# WAKE: Enumerate for unsolved tasks
for task in tasks:
    if already_solved:
        continue
    frontier = _enumerate_task(task)

# SLEEP: Compress to find abstractions
if iteration % compress_every == 0:
    new_inventions = _compress()
```

#### `_enumerate_task()` (lines 327-377)
Enumerate using TopDownEnumerator:
```python
enumerator = TopDownEnumerator(grammar, max_depth, max_programs)
for program, log_prob in enumerator.enumerate(request_type):
    # Evaluate on examples
    if all_correct:
        frontier.add(result)
        break  # Found solution
```

---

## Enumeration Flow Summary

```
1. Task: (examples, request_type) e.g., [(hand1, True), (hand2, False)], HAND -> BOOL

2. Start: Hole(HAND -> BOOL)

3. Expand: λ.Hole(BOOL)  [Arrow type gets lambda]

4. For each grammar production matching BOOL:
   - all_same_suit : HAND -> BOOL
     → Substitute: λ.(all_same_suit ?:HAND)
   - eq : 'a -> 'a -> BOOL  [needs args]
     → Substitute: λ.(eq ?:'a ?:'a)
   - Variables: $0 : HAND (if coercible to BOOL)

5. Continue expanding holes until complete program

6. Evaluate complete programs on examples

7. Return first program where all examples pass
```

---

## Key Metrics for Cognitive Modeling

| Metric | Purpose |
|--------|---------|
| `programs_enumerated` | Search difficulty |
| `partial_programs_explored` | Pruning effectiveness |
| `description_length` | Solution complexity (MDL) |
| `time_seconds` | Wall clock time |
| `iteration_solved` | Learning curve position |

---

## Enumeration Variants Comparison

| Variant | Complexity | Order | Memoization | Best For |
|---------|------------|-------|-------------|----------|
| Memoized | O(?) | By cost | Yes | Production (default) |
| Priority Queue | O(n log n) | By cost | No | Edge cases |
| Cost Bands | O(?) | By cost | Partial | Memory-constrained |
| Iterative Deepening | O(?) | By depth | No | PyPy workers |

---

## Appendix: Quick Reference

### Creating a Grammar
```python
from dreamcoder_core.lean_primitives import build_lean_grammar
grammar = build_lean_grammar()
```

### Running Enumeration
```python
from dreamcoder_core.enumeration import TopDownEnumerator

enumerator = TopDownEnumerator(grammar, max_depth=8)
for program, log_prob in enumerator.enumerate(request_type):
    # Evaluate program...
```

### Full Wake-Sleep Learning
```python
from dreamcoder_core.wake_sleep import DreamCoder
from dreamcoder_core.task import Task

tasks = [Task(name="...", request_type=..., examples=[...])]
dc = DreamCoder(grammar, tasks, eval_fn)
result = dc.run()
print(result.summary())
```
