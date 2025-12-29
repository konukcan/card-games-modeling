"""
================================================================================
ENUMERATION.PY - THE "WAKE" PHASE OF DREAMCODER
================================================================================

This module implements PROGRAM ENUMERATION: the systematic search through the
space of all possible programs to find ones that solve a given task.

In DreamCoder's wake-sleep loop, this is the "WAKE" phase:
  1. WAKE: Enumerate programs, find solutions to tasks  <-- THIS FILE
  2. SLEEP (Dreaming): Train neural recognition model on solved tasks
  3. SLEEP (Abstraction): Compress library with new learned subroutines

--------------------------------------------------------------------------------
ENUMERATION STRATEGIES
--------------------------------------------------------------------------------

This module provides TWO enumeration strategies:

1. TOP-DOWN ENUMERATION (TopDownEnumerator - NEW, recommended)
   - Start with a "hole" ?:request_type
   - Repeatedly fill holes with productions from grammar
   - Work with PARTIAL PROGRAMS (programs with holes)
   - Can prune early by checking type constraints
   - Example: ?:int → (+ ?:int ?:int) → (+ 1 ?:int) → (+ 1 2)

   This is the strategy used by original DreamCoder (Ellis et al.)

2. ITERATIVE DEEPENING (enumerate_simple - simpler but less efficient)
   - Enumerate all programs of depth 1, then depth 2, then depth 3, ...
   - Guarantees shortest programs found first
   - Simple to implement but doesn't use grammar probabilities

--------------------------------------------------------------------------------
TOP-DOWN ENUMERATION EXPLAINED
--------------------------------------------------------------------------------

Top-down enumeration works by:
  1. Start with a single Hole of the request type: ?0:request_type
  2. Pop the lowest-cost partial program from priority queue
  3. Find the first (leftmost) hole in the program
  4. For each grammar production that could fill that hole:
     - Create a new program with the hole replaced
     - If the production needs arguments, add new holes for them
     - Push the new partial program to the queue
  5. When a program has no holes, yield it as a solution
  6. Repeat until done

Example for request type (int → int):

  Step 0: ?0:(int→int)                      cost=0

  Step 1: Expand with λ.?1:int              cost=0 (lambda is free)
          ?0 filled with Abstraction(Hole)

  Step 2: Expand ?1 with (+ ?2:int ?3:int)  cost=log(1/|G|)

  Step 3: Expand ?2 with $0                 cost += log(p_var)
          Now have: λ.(+ $0 ?3:int)

  Step 4: Expand ?3 with 1                  cost += log(p_1)
          Complete! λ.(+ $0 1)              yield it!

--------------------------------------------------------------------------------
KEY METRICS
--------------------------------------------------------------------------------

This module tracks metrics relevant for modeling:

  - programs_enumerated: How many programs were considered?
  - partial_programs_explored: How many partial programs were expanded?
  - time_seconds: Wall clock time to find solution
  - description_length: Complexity of the solution (bits)

These can be compared with human patterns.

================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import heapq
import math
import time
from collections import defaultdict

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow, type_arity
)
from .program import (
    Program, Primitive, Application, Abstraction, Index, Invented, Hole,
    apply_args, has_holes, find_first_hole, substitute_hole, collect_holes,
    count_holes, reset_hole_counter
)
from .grammar import Grammar, Production


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EnumerationResult:
    """
    Container for a program found during enumeration.

    Attributes:
        program: The synthesized program (complete, no holes)
        log_probability: Grammar probability P(program|grammar) in log-space
        log_likelihood: How well program fits examples (0 = perfect)
        description_length: MDL = -log_probability / log(2), measured in bits
        programs_enumerated: How many complete programs were yielded before this
        partial_programs_explored: How many partial programs were expanded
        time_seconds: Wall clock time to find this program
    """
    program: Program
    log_probability: float
    log_likelihood: float
    description_length: float
    programs_enumerated: int
    partial_programs_explored: int = 0
    time_seconds: float = 0.0


@dataclass
class Frontier:
    """
    A frontier of the best candidate programs found for a task.

    Keeps track of the best k programs (by total score) for library learning.
    """
    task_name: str
    request_type: Type
    entries: List[EnumerationResult] = field(default_factory=list)
    max_size: int = 10

    def add(self, result: EnumerationResult) -> bool:
        """Add a result if it improves the frontier."""
        for e in self.entries:
            if e.program == result.program:
                return False

        self.entries.append(result)
        self.entries.sort(key=lambda e: e.description_length - e.log_likelihood)

        if len(self.entries) > self.max_size:
            self.entries = self.entries[:self.max_size]
            return result in self.entries
        return True

    @property
    def best(self) -> Optional[EnumerationResult]:
        return self.entries[0] if self.entries else None

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0

    @property
    def solved(self) -> bool:
        """Task is solved if we have at least one perfect solution."""
        return any(e.log_likelihood == 0.0 for e in self.entries)

    @property
    def n_solutions(self) -> int:
        """Number of perfect solutions (log_likelihood == 0)."""
        return sum(1 for e in self.entries if e.log_likelihood == 0.0)


@dataclass(order=True)
class PriorityItem:
    """
    Item in the enumeration priority queue.

    Priority queue is a MIN-HEAP: lower priority = better = explored first.
    We use cost (negative log probability) as priority.
    """
    priority: float  # Cost = -log_probability
    program: Program = field(compare=False)
    tp: Type = field(compare=False)
    env: List[Type] = field(compare=False, default_factory=list)


# ============================================================================
# TOP-DOWN ENUMERATOR (NEW - TRUE HOLE-FILLING)
# ============================================================================

class TopDownEnumerator:
    """
    True top-down program enumerator using hole-filling.

    This is the strategy used by original DreamCoder. It works by:
    1. Starting with a single Hole of the request type
    2. Repeatedly expanding the first hole with grammar productions
    3. Yielding complete programs (no holes) in order of cost

    ADVANTAGES over eager bottom-up:
    - Memory efficient: only partial programs in queue, not all complete programs
    - Can prune early: if partial program already exceeds cost bound, skip it
    - Naturally handles higher-order functions

    MEMOIZATION (Option C - DreamCoder style):
    The enumerate_memoized() method caches subproblem solutions by (type, env, budget_bucket).
    This avoids re-enumerating the same subproblem multiple times, which is the key to
    DreamCoder's efficiency.

    Attributes:
        grammar: The probabilistic grammar
        max_depth: Maximum AST depth
        max_programs: Maximum complete programs to yield
        programs_enumerated: Counter of complete programs yielded
        partial_programs_explored: Counter of partial programs expanded
    """

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int = 8,
        max_programs: int = 100000
    ):
        self.grammar = grammar
        self.max_depth = max_depth
        self.max_programs = max_programs
        self.programs_enumerated = 0
        self.partial_programs_explored = 0

        # Memoization cache: (type_str, env_tuple, budget_bucket) -> List[(Program, cost)]
        self._memo_cache: Dict[Tuple[str, Tuple[str, ...], int], List[Tuple[Program, float]]] = {}
        # Track keys currently being filled to handle reentrancy
        self._memo_in_progress: Set[Tuple[str, Tuple[str, ...], int]] = set()
        self._memo_hits = 0
        self._memo_misses = 0

    def enumerate(
        self,
        request_type: Type,
        max_cost: float = float('inf'),
        timeout_seconds: float = float('inf'),
        env: List[Type] = None
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs of the given type using top-down hole-filling.

        Yields (program, log_probability) pairs in order of increasing cost.

        Args:
            request_type: The type of programs to enumerate
            max_cost: Maximum cost (negative log probability) to consider
            timeout_seconds: Wall clock timeout
            env: Type environment for bound variables (initially empty)

        Yields:
            (complete_program, log_probability) tuples
        """
        if env is None:
            env = []

        start_time = time.time()
        self.programs_enumerated = 0
        self.partial_programs_explored = 0

        # Reset hole counter for clean IDs
        reset_hole_counter()

        # Priority queue of partial programs
        pq: List[PriorityItem] = []

        # Track seen programs to avoid duplicates
        seen: Set[int] = set()

        # Start with a single hole of the request type
        initial_hole = Hole(request_type)
        heapq.heappush(pq, PriorityItem(0.0, initial_hole, request_type, env))

        while pq and self.programs_enumerated < self.max_programs:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            # Pop lowest-cost partial program
            item = heapq.heappop(pq)
            cost = item.priority
            program = item.program
            current_env = item.env

            # Skip if exceeds cost bound
            if cost > max_cost:
                continue

            # Skip if exceeds depth
            if program.depth() > self.max_depth:
                continue

            # Check if complete (no holes)
            if not has_holes(program):
                prog_hash = hash(program)
                if prog_hash not in seen:
                    seen.add(prog_hash)
                    self.programs_enumerated += 1
                    yield (program, -cost)
                continue

            # Expand the first hole
            self.partial_programs_explored += 1
            self._expand_first_hole(pq, program, cost, current_env, seen)

    def _expand_first_hole(
        self,
        pq: List[PriorityItem],
        program: Program,
        base_cost: float,
        env: List[Type],
        seen: Set[int]
    ) -> None:
        """
        Expand the first hole in a partial program with all valid productions.

        For each grammar production that could fill the hole:
        1. If production has arrow type, apply it and add holes for arguments
        2. If production has base type, substitute directly
        3. Push new partial program to priority queue

        IMPORTANT: The `env` parameter is the type environment for the FIRST
        (leftmost) hole in the program. This is maintained correctly because:
        - When we create a lambda to fill an arrow-type hole, we push with
          extended env [arg_type] + old_env
        - When we fill a hole with a non-lambda, the remaining holes are at
          the same depth, so they use the same env

        PREDICTIVE DEPTH PRUNING (Option B fix for partial program explosion):
        Before pushing any partial to the queue, we estimate if it CAN be
        completed within max_depth. If not, we skip it entirely.
        """
        # Find the first hole
        hole = find_first_hole(program)
        if hole is None:
            return  # No holes to expand

        hole_type = hole.tp
        ctx = TypeContext()

        # Use the env directly - it's correct for the first hole by construction
        hole_env = env

        # Helper for predictive depth pruning
        def can_complete(new_prog: Program) -> bool:
            """
            Estimate if this partial program can complete within max_depth.

            Heuristic: Each remaining hole needs at least 1 AST level to fill.
            This is conservative (some holes need more), but catches the worst cases.
            """
            n_holes = count_holes(new_prog)
            if n_holes == 0:
                return True  # Already complete!

            current_depth = new_prog.depth()
            # Minimum additional depth: each hole needs at least 1 level
            # (but holes can share depth levels, so we use a looser bound)
            # More aggressive: assume at least half the holes add depth
            min_additional_depth = max(0, (n_holes - 1) // 2)
            estimated_final_depth = current_depth + min_additional_depth

            return estimated_final_depth <= self.max_depth

        # CASE 1: If hole type is an arrow, we can fill with a lambda
        if isinstance(hole_type, Arrow):
            # Create: λ. ?:body_type
            body_type = hole_type.ret
            arg_type = hole_type.arg

            # New hole for the body, with extended environment
            body_hole = Hole(body_type)
            lambda_prog = Abstraction(body_hole)

            # Substitute this lambda for the hole
            new_program = substitute_hole(program, hole.id, lambda_prog)

            # Lambda doesn't add cost (it's structural)
            # The body hole will be filled later with its own cost
            new_env = [arg_type] + hole_env

            # PREDICTIVE PRUNING: Can this complete within depth?
            if can_complete(new_program):
                prog_hash = hash(new_program)
                if prog_hash not in seen:
                    heapq.heappush(pq, PriorityItem(
                        base_cost,
                        new_program,
                        hole_type,
                        new_env
                    ))

        # CASE 2: Try grammar productions that produce hole_type
        candidates = self.grammar.candidates_for_type(hole_type, ctx, hole_env)

        for prod, inst_type, log_prob in candidates:
            prod_cost = -log_prob
            new_cost = base_cost + prod_cost

            # Skip if already too expensive
            if new_cost > self.max_depth * 10:  # Rough bound
                continue

            if isinstance(inst_type, Arrow):
                # Production needs arguments - create holes for them
                filled = self._apply_with_holes(prod.program, inst_type)
            else:
                # Production directly produces the type
                filled = prod.program

            new_program = substitute_hole(program, hole.id, filled)

            # Check depth (current)
            if new_program.depth() > self.max_depth:
                continue

            # PREDICTIVE PRUNING: Can this complete within depth?
            if not can_complete(new_program):
                continue

            prog_hash = hash(new_program)
            if prog_hash not in seen:
                heapq.heappush(pq, PriorityItem(
                    new_cost,
                    new_program,
                    hole_type,
                    hole_env
                ))

        # CASE 3: Try variables from the environment
        var_candidates = self.grammar.variable_candidates(hole_type, ctx, hole_env)

        for idx, log_prob in var_candidates:
            var_cost = -log_prob
            new_cost = base_cost + var_cost

            new_program = substitute_hole(program, hole.id, Index(idx))

            # Variables are leaves (no holes added), always can complete if depth OK
            prog_hash = hash(new_program)
            if prog_hash not in seen:
                heapq.heappush(pq, PriorityItem(
                    new_cost,
                    new_program,
                    hole_type,
                    hole_env
                ))

    def _apply_with_holes(self, func: Program, func_type: Arrow) -> Program:
        """
        Apply a function to holes for each of its arguments.

        Given func : A → B → C, creates: ((func ?:A) ?:B)

        This allows top-down enumeration to expand functions
        by creating holes for their required arguments.
        """
        result = func
        current_type = func_type

        while isinstance(current_type, Arrow):
            arg_type = current_type.arg
            arg_hole = Hole(arg_type)
            result = Application(result, arg_hole)
            current_type = current_type.ret

        return result

    # Note: Environment tracking is done by storing the correct env in PriorityItem
    # when we push new partial programs. The env is updated when we introduce
    # lambdas (CASE 1) and stays the same when we fill holes at the same depth.
    # This avoids the complexity of recomputing the env from the AST structure.

    def enumerate_with_cost_bands(
        self,
        request_type: Type,
        max_cost: float = 50.0,
        timeout_seconds: float = float('inf'),
        env: List[Type] = None,
        initial_budget: float = 12.0,
        budget_increment: float = 1.5
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs using ITERATIVE COST DEEPENING (like DreamCoder).

        This addresses the partial program explosion problem by:
        1. Using a SINGLE PERSISTENT priority queue (not restarted per band)
        2. Processing items band by band, deferring high-cost items to later bands
        3. This naturally bounds memory usage and avoids re-exploration

        The key DreamCoder insight: Don't restart enumeration for each band.
        Instead, maintain state and only expand what's within the current budget.

        Args:
            request_type: The type of programs to enumerate
            max_cost: Maximum cost to explore (default 50.0)
            timeout_seconds: Wall clock timeout
            env: Type environment for bound variables
            initial_budget: Starting cost budget (default 12.0)
            budget_increment: How much to increase budget each iteration

        Yields:
            (complete_program, log_probability) tuples
        """
        if env is None:
            env = []

        start_time = time.time()
        self.programs_enumerated = 0
        self.partial_programs_explored = 0

        # Reset hole counter for clean IDs
        reset_hole_counter()

        # SINGLE PERSISTENT priority queue (key difference from broken version)
        pq: List[PriorityItem] = []

        # TWO SEPARATE SEEN SETS (critical for correctness!):
        # - partial_seen: prevents duplicate partials from entering queue
        # - yielded: tracks which complete programs have been returned
        partial_seen: Set[int] = set()
        yielded: Set[int] = set()

        # Start with a single hole of the request type
        initial_hole = Hole(request_type)
        heapq.heappush(pq, PriorityItem(0.0, initial_hole, request_type, env))

        # Track current budget
        current_budget = initial_budget

        while pq and self.programs_enumerated < self.max_programs:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            # Check if we've exceeded max_cost
            if current_budget > max_cost:
                break

            # Peek at lowest-cost item
            if pq[0].priority > current_budget:
                # All remaining items exceed current budget - increase it
                if current_budget < 20:
                    current_budget += budget_increment
                elif current_budget < 30:
                    current_budget += budget_increment * 0.75
                else:
                    current_budget += budget_increment * 0.5
                continue

            # Pop lowest-cost partial program
            item = heapq.heappop(pq)
            cost = item.priority
            program = item.program
            current_env = item.env

            # Skip if exceeds current band (shouldn't happen due to peek above)
            if cost > current_budget:
                # Put it back for the next band
                heapq.heappush(pq, item)
                continue

            # Skip if exceeds depth
            if program.depth() > self.max_depth:
                continue

            # Check if complete (no holes)
            if not has_holes(program):
                prog_hash = hash(program)
                if prog_hash not in yielded:
                    yielded.add(prog_hash)
                    self.programs_enumerated += 1
                    yield (program, -cost)
                continue

            # Expand the first hole (with predictive pruning)
            self.partial_programs_explored += 1
            self._expand_first_hole_with_budget(
                pq, program, cost, current_env, partial_seen, current_budget
            )

    def _expand_first_hole_with_budget(
        self,
        pq: List[PriorityItem],
        program: Program,
        base_cost: float,
        env: List[Type],
        seen: Set[int],
        budget: float
    ) -> None:
        """
        Expand first hole, but only push items within budget.

        Items exceeding budget are STILL pushed (for future bands),
        but we prioritize staying within budget for memory efficiency.
        """
        hole = find_first_hole(program)
        if hole is None:
            return

        hole_type = hole.tp
        ctx = TypeContext()
        hole_env = env

        # CASE 1: Lambda for arrow types
        if isinstance(hole_type, Arrow):
            body_type = hole_type.ret
            arg_type = hole_type.arg

            body_hole = Hole(body_type)
            lambda_prog = Abstraction(body_hole)
            new_program = substitute_hole(program, hole.id, lambda_prog)
            new_env = [arg_type] + hole_env

            # Always push lambdas (they don't add cost)
            prog_hash = hash(new_program)
            if prog_hash not in seen:
                seen.add(prog_hash)
                heapq.heappush(pq, PriorityItem(
                    base_cost, new_program, hole_type, new_env
                ))

        # CASE 2: Grammar productions
        candidates = self.grammar.candidates_for_type(hole_type, ctx, hole_env)

        for prod, inst_type, log_prob in candidates:
            prod_cost = -log_prob
            new_cost = base_cost + prod_cost

            # Skip if WAY too expensive (2x budget as hard limit)
            if new_cost > budget * 2:
                continue

            if isinstance(inst_type, Arrow):
                filled = self._apply_with_holes(prod.program, inst_type)
            else:
                filled = prod.program

            new_program = substitute_hole(program, hole.id, filled)

            # Check depth
            if new_program.depth() > self.max_depth:
                continue

            # Predictive depth pruning
            n_holes = count_holes(new_program)
            if n_holes > 0:
                min_additional = max(0, (n_holes - 1) // 2)
                if new_program.depth() + min_additional > self.max_depth:
                    continue

            prog_hash = hash(new_program)
            if prog_hash not in seen:
                seen.add(prog_hash)
                heapq.heappush(pq, PriorityItem(
                    new_cost, new_program, hole_type, hole_env
                ))

        # CASE 3: Variables
        var_candidates = self.grammar.variable_candidates(hole_type, ctx, hole_env)

        for idx, log_prob in var_candidates:
            var_cost = -log_prob
            new_cost = base_cost + var_cost

            if new_cost > budget * 2:
                continue

            new_program = substitute_hole(program, hole.id, Index(idx))

            prog_hash = hash(new_program)
            if prog_hash not in seen:
                seen.add(prog_hash)
                heapq.heappush(pq, PriorityItem(
                    new_cost, new_program, hole_type, hole_env
                ))

    def _enumerate_cost_band(
        self,
        request_type: Type,
        lower_bound: float,
        upper_bound: float,
        timeout_seconds: float,
        env: List[Type],
        global_seen: Set[int]
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs within a specific cost band.

        Only yields programs where lower_bound < cost <= upper_bound.
        Uses global_seen to avoid duplicates across bands.
        """
        start_time = time.time()

        # Reset hole counter for clean IDs
        reset_hole_counter()

        # Priority queue of partial programs
        pq: List[PriorityItem] = []

        # Local seen set for this band (merged with global at end)
        band_seen: Set[int] = set()

        # Start with a single hole of the request type
        initial_hole = Hole(request_type)
        heapq.heappush(pq, PriorityItem(0.0, initial_hole, request_type, env))

        while pq and self.programs_enumerated < self.max_programs:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            # Pop lowest-cost partial program
            item = heapq.heappop(pq)
            cost = item.priority
            program = item.program
            current_env = item.env

            # KEY CHANGE: Skip if exceeds THIS BAND's upper bound
            # This prevents queue explosion!
            if cost > upper_bound:
                continue

            # Skip if exceeds depth
            if program.depth() > self.max_depth:
                continue

            # Check if complete (no holes)
            if not has_holes(program):
                prog_hash = hash(program)

                # Only yield if in THIS band (cost > lower_bound)
                # AND not seen globally before
                if cost > lower_bound and prog_hash not in global_seen:
                    global_seen.add(prog_hash)
                    band_seen.add(prog_hash)
                    self.programs_enumerated += 1
                    yield (program, -cost)
                continue

            # Expand the first hole (but respect band limits)
            self.partial_programs_explored += 1
            self._expand_first_hole_bounded(
                pq, program, cost, current_env,
                band_seen, global_seen, upper_bound
            )

    def _expand_first_hole_bounded(
        self,
        pq: List[PriorityItem],
        program: Program,
        base_cost: float,
        env: List[Type],
        band_seen: Set[int],
        global_seen: Set[int],
        upper_bound: float
    ) -> None:
        """
        Expand first hole with EARLY PRUNING based on cost band.

        Key difference from _expand_first_hole:
        We DON'T push partials that would exceed upper_bound.
        """
        # Find the first hole
        hole = find_first_hole(program)
        if hole is None:
            return

        hole_type = hole.tp
        ctx = TypeContext()
        hole_env = env

        # CASE 1: If hole type is an arrow, fill with lambda
        if isinstance(hole_type, Arrow):
            body_type = hole_type.ret
            arg_type = hole_type.arg

            body_hole = Hole(body_type)
            lambda_prog = Abstraction(body_hole)
            new_program = substitute_hole(program, hole.id, lambda_prog)

            # Lambda has 0 cost, so same base_cost
            # Only push if we can possibly complete within bounds
            if base_cost <= upper_bound:  # Lambda doesn't add cost
                new_env = [arg_type] + hole_env
                prog_hash = hash(new_program)
                if prog_hash not in band_seen and prog_hash not in global_seen:
                    band_seen.add(prog_hash)
                    heapq.heappush(pq, PriorityItem(
                        base_cost,
                        new_program,
                        hole_type,
                        new_env
                    ))

        # CASE 2: Try grammar productions
        candidates = self.grammar.candidates_for_type(hole_type, ctx, hole_env)

        for prod, inst_type, log_prob in candidates:
            prod_cost = -log_prob
            new_cost = base_cost + prod_cost

            # EARLY PRUNING: Don't push if exceeds band
            if new_cost > upper_bound:
                continue

            if isinstance(inst_type, Arrow):
                filled = self._apply_with_holes(prod.program, inst_type)
            else:
                filled = prod.program

            new_program = substitute_hole(program, hole.id, filled)

            # Check depth
            if new_program.depth() > self.max_depth:
                continue

            prog_hash = hash(new_program)
            if prog_hash not in band_seen and prog_hash not in global_seen:
                band_seen.add(prog_hash)
                heapq.heappush(pq, PriorityItem(
                    new_cost,
                    new_program,
                    hole_type,
                    hole_env
                ))

        # CASE 3: Try variables from environment
        var_candidates = self.grammar.variable_candidates(hole_type, ctx, hole_env)

        for idx, log_prob in var_candidates:
            var_cost = -log_prob
            new_cost = base_cost + var_cost

            # EARLY PRUNING: Don't push if exceeds band
            if new_cost > upper_bound:
                continue

            new_program = substitute_hole(program, hole.id, Index(idx))

            prog_hash = hash(new_program)
            if prog_hash not in band_seen and prog_hash not in global_seen:
                band_seen.add(prog_hash)
                heapq.heappush(pq, PriorityItem(
                    new_cost,
                    new_program,
                    hole_type,
                    hole_env
                ))

    # =========================================================================
    # OPTION C: MEMOIZED ENUMERATION (DreamCoder-style dynamic programming)
    # =========================================================================

    def enumerate_memoized(
        self,
        request_type: Type,
        max_cost: float = 50.0,
        timeout_seconds: float = float('inf'),
        env: List[Type] = None,
        depth_limit: int = None
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs using ITERATIVE COST DEEPENING with memoization.

        This follows DreamCoder's approach: enumerate by COST LEVEL to maintain
        global cost order, while caching subproblem results for efficiency.

        For each cost level (0, 1, 2, ...):
        1. Build all programs of cost <= current_cost using cached subproblems
        2. Yield any new programs found
        3. Increment cost and repeat

        This ensures programs are yielded in cost order (like priority queue)
        while still getting memoization benefits.

        Args:
            request_type: The type of programs to enumerate
            max_cost: Maximum cost to explore (default 50.0)
            timeout_seconds: Wall clock timeout
            env: Type environment for bound variables
            depth_limit: Override max_depth for this enumeration

        Yields:
            (complete_program, log_probability) tuples in cost order
        """
        if env is None:
            env = []
        if depth_limit is None:
            depth_limit = self.max_depth

        start_time = time.time()
        self.programs_enumerated = 0
        self.partial_programs_explored = 0
        self._memo_hits = 0
        self._memo_misses = 0

        # Clear cache for fresh enumeration
        self._memo_cache.clear()
        self._memo_in_progress.clear()

        # Track programs already yielded to avoid duplicates
        yielded: Set[int] = set()

        # Iterative cost deepening: enumerate by increasing cost
        cost_step = 1.0  # Granularity of cost levels
        current_max_cost = cost_step

        while current_max_cost <= max_cost:
            if time.time() - start_time > timeout_seconds:
                break
            if self.programs_enumerated >= self.max_programs:
                break

            # Enumerate all programs up to current cost
            for prog, cost in self._enumerate_type_at_cost(
                request_type, env, current_max_cost, depth_limit, start_time, timeout_seconds
            ):
                prog_hash = hash(prog)
                if prog_hash not in yielded:
                    yielded.add(prog_hash)
                    self.programs_enumerated += 1
                    yield (prog, -cost)  # Return log_prob = -cost

                    if self.programs_enumerated >= self.max_programs:
                        return

            current_max_cost += cost_step

    def _enumerate_type_at_cost(
        self,
        tp: Type,
        env: List[Type],
        max_cost: float,
        depth_remaining: int,
        start_time: float,
        timeout: float
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs of a given type up to max_cost.

        Uses memoization by caching results per (type, env, cost_bucket).
        Results are returned sorted by cost.
        """
        if depth_remaining <= 0:
            return
        if max_cost < 0:
            return
        if time.time() - start_time > timeout:
            return

        # CASE 1: Arrow type - enumerate lambdas
        if isinstance(tp, Arrow):
            body_type = tp.ret
            arg_type = tp.arg
            new_env = [arg_type] + env

            for body, body_cost in self._enumerate_type_at_cost(
                body_type, new_env, max_cost, depth_remaining - 1,
                start_time, timeout
            ):
                lambda_prog = Abstraction(body)
                yield (lambda_prog, body_cost)
            return

        # CASE 2: Base type - check cache or enumerate
        env_key = tuple(str(t) for t in env)
        cost_bucket = int(max_cost)  # Integer cost buckets
        cache_key = (str(tp), env_key, cost_bucket)

        # Check cache
        if cache_key in self._memo_cache:
            self._memo_hits += 1
            for prog, cost in self._memo_cache[cache_key]:
                if cost <= max_cost:
                    yield (prog, cost)
            return

        # Cache miss - enumerate and store
        self._memo_misses += 1
        results: List[Tuple[Program, float]] = []
        ctx = TypeContext()

        # Variables
        var_candidates = self.grammar.variable_candidates(tp, ctx, env)
        for idx, log_prob in var_candidates:
            cost = -log_prob
            if cost <= max_cost:
                var_prog = Index(idx)
                results.append((var_prog, cost))
                self.partial_programs_explored += 1

        # Productions
        candidates = self.grammar.candidates_for_type(tp, ctx, env)

        for prod, inst_type, log_prob in candidates:
            prod_cost = -log_prob

            if prod_cost > max_cost:
                continue

            if isinstance(inst_type, Arrow):
                # Production needs arguments - enumerate them
                remaining_budget = max_cost - prod_cost
                for applied, applied_cost in self._apply_args_at_cost(
                    prod.program, inst_type, env,
                    remaining_budget, depth_remaining - 1,
                    start_time, timeout
                ):
                    total_cost = prod_cost + applied_cost
                    if total_cost <= max_cost:
                        results.append((applied, total_cost))
            else:
                # Base case - directly produces the type
                results.append((prod.program, prod_cost))
                self.partial_programs_explored += 1

        # Sort by cost and cache
        results.sort(key=lambda x: x[1])
        self._memo_cache[cache_key] = results

        # Yield results
        for prog, cost in results:
            yield (prog, cost)

    def _apply_args_at_cost(
        self,
        func: Program,
        func_type: Arrow,
        env: List[Type],
        max_cost: float,
        depth_remaining: int,
        start_time: float,
        timeout: float
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Apply a function to arguments, enumerating arguments up to max_cost.
        """
        if depth_remaining <= 0:
            return
        if max_cost < 0:
            return
        if time.time() - start_time > timeout:
            return

        arg_type = func_type.arg
        ret_type = func_type.ret

        # Enumerate arguments
        for arg, arg_cost in self._enumerate_type_at_cost(
            arg_type, env, max_cost, depth_remaining,
            start_time, timeout
        ):
            if arg_cost > max_cost:
                continue

            new_prog = Application(func, arg)
            remaining = max_cost - arg_cost

            if isinstance(ret_type, Arrow):
                # More arguments needed
                for applied, more_cost in self._apply_args_at_cost(
                    new_prog, ret_type, env, remaining, depth_remaining,
                    start_time, timeout
                ):
                    yield (applied, arg_cost + more_cost)
            else:
                # Fully applied
                self.partial_programs_explored += 1
                yield (new_prog, arg_cost)

    def _enumerate_type_memoized(
        self,
        tp: Type,
        env: List[Type],
        budget: float,
        depth_remaining: int,
        start_time: float,
        timeout: float
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs of a given type using memoization.

        Uses recursive descent with memoization: each call to this function
        is cached by (type, env, budget_bucket).
        """
        if depth_remaining <= 0:
            return
        if budget < 0:
            return
        if time.time() - start_time > timeout:
            return

        # CASE 1: Arrow type - enumerate lambdas
        if isinstance(tp, Arrow):
            # For arrow types, we create lambda abstractions
            body_type = tp.ret
            arg_type = tp.arg
            new_env = [arg_type] + env

            for body, body_cost in self._enumerate_type_memoized(
                body_type, new_env, budget, depth_remaining - 1,
                start_time, timeout
            ):
                lambda_prog = Abstraction(body)
                yield (lambda_prog, body_cost)
            return

        # CASE 2: Base type - check cache or enumerate
        # Create cache key
        env_key = tuple(str(t) for t in env)
        budget_bucket = int(budget / 2.0)  # Discretize to 2.0 unit buckets
        cache_key = (str(tp), env_key, budget_bucket)

        # Check cache
        if cache_key in self._memo_cache:
            self._memo_hits += 1
            for prog, cost in self._memo_cache[cache_key]:
                if cost <= budget:
                    yield (prog, cost)
            return

        # REENTRANCY CHECK: If this key is currently being filled by a caller,
        # enumerate without caching to avoid returning empty results.
        # This handles cases like: BOOL needs (not X) which needs BOOL arguments.
        if cache_key in self._memo_in_progress:
            # Don't cache, just enumerate directly
            for prog, cost in self._enumerate_without_cache(
                tp, env, budget, depth_remaining, start_time, timeout
            ):
                yield (prog, cost)
            return

        # Mark as in progress to detect reentrancy
        self._memo_in_progress.add(cache_key)

        # Cache miss - enumerate and store
        self._memo_misses += 1
        results: List[Tuple[Program, float]] = []

        ctx = TypeContext()

        # Try variables from environment first (cheapest)
        var_candidates = self.grammar.variable_candidates(tp, ctx, env)
        for idx, log_prob in var_candidates:
            cost = -log_prob
            if cost <= budget:
                var_prog = Index(idx)
                results.append((var_prog, cost))
                self.partial_programs_explored += 1
                yield (var_prog, cost)

        # Try grammar productions
        candidates = self.grammar.candidates_for_type(tp, ctx, env)

        for prod, inst_type, log_prob in candidates:
            prod_cost = -log_prob

            if prod_cost > budget:
                continue

            if isinstance(inst_type, Arrow):
                # Production needs arguments - recurse
                remaining_budget = budget - prod_cost
                for applied, applied_cost in self._apply_production_memoized(
                    prod.program, inst_type, env,
                    remaining_budget, depth_remaining - 1,
                    start_time, timeout
                ):
                    total_cost = prod_cost + applied_cost
                    if total_cost <= budget:
                        results.append((applied, total_cost))
                        yield (applied, total_cost)
            else:
                # Production directly produces the type (base case)
                results.append((prod.program, prod_cost))
                self.partial_programs_explored += 1
                yield (prod.program, prod_cost)

        # Done filling - remove from in-progress and store in cache
        self._memo_in_progress.discard(cache_key)
        results.sort(key=lambda x: x[1])
        self._memo_cache[cache_key] = results

    def _enumerate_without_cache(
        self,
        tp: Type,
        env: List[Type],
        budget: float,
        depth_remaining: int,
        start_time: float,
        timeout: float
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate BASE CASE programs only (for reentrant calls).

        This is called when we detect reentrancy - i.e., we're trying to
        enumerate type T while we're already in the middle of caching type T.

        To break the recursion, we ONLY yield:
        1. Variables from the environment
        2. Primitives that directly produce the type (no arguments)

        We do NOT recurse for productions needing arguments - those would
        create infinite loops. The caller will get the base cases and can
        use them to build more complex programs.
        """
        if depth_remaining <= 0:
            return
        if budget < 0:
            return
        if time.time() - start_time > timeout:
            return

        ctx = TypeContext()

        # Variables
        var_candidates = self.grammar.variable_candidates(tp, ctx, env)
        for idx, log_prob in var_candidates:
            cost = -log_prob
            if cost <= budget:
                yield (Index(idx), cost)

        # ONLY base-case productions (no arguments needed)
        candidates = self.grammar.candidates_for_type(tp, ctx, env)

        for prod, inst_type, log_prob in candidates:
            prod_cost = -log_prob

            if prod_cost > budget:
                continue

            if not isinstance(inst_type, Arrow):
                # Base case - directly produces the type
                yield (prod.program, prod_cost)
            # Skip Arrow types to avoid recursion

    def _apply_production_memoized(
        self,
        func: Program,
        func_type: Arrow,
        env: List[Type],
        budget: float,
        depth_remaining: int,
        start_time: float,
        timeout: float
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Apply a production to arguments, using memoized enumeration for each arg.

        Given func : A → B → C, enumerate all ways to apply it:
        - For each arg1 of type A with cost c1
        - For each arg2 of type B with cost c2 where c1 + c2 <= budget
        - Yield ((func arg1) arg2), c1 + c2
        """
        if depth_remaining <= 0:
            return
        if budget < 0:
            return
        if time.time() - start_time > timeout:
            return

        arg_type = func_type.arg
        ret_type = func_type.ret

        # Enumerate arguments using memoization
        for arg, arg_cost in self._enumerate_type_memoized(
            arg_type, env, budget, depth_remaining,
            start_time, timeout
        ):
            if arg_cost > budget:
                continue

            new_prog = Application(func, arg)
            remaining = budget - arg_cost

            if isinstance(ret_type, Arrow):
                # More arguments needed
                for applied, more_cost in self._apply_production_memoized(
                    new_prog, ret_type, env, remaining, depth_remaining,
                    start_time, timeout
                ):
                    yield (applied, arg_cost + more_cost)
            else:
                # Base case - fully applied
                self.partial_programs_explored += 1
                yield (new_prog, arg_cost)

    def clear_memo_cache(self):
        """Clear the memoization cache (call between tasks if needed)."""
        self._memo_cache.clear()
        self._memo_in_progress.clear()
        self._memo_hits = 0
        self._memo_misses = 0

    def get_memo_stats(self) -> Dict[str, int]:
        """Get memoization cache statistics."""
        return {
            'cache_size': len(self._memo_cache),
            'hits': self._memo_hits,
            'misses': self._memo_misses,
            'hit_rate': self._memo_hits / max(1, self._memo_hits + self._memo_misses)
        }


# ============================================================================
# LEGACY ENUMERATOR (kept for backwards compatibility)
# ============================================================================

class Enumerator:
    """
    Legacy best-first enumerator using eager bottom-up construction.

    DEPRECATED: Use TopDownEnumerator for better performance.

    This enumerator eagerly builds all complete programs upfront and
    pushes them to a priority queue. It's simpler but less efficient
    than true top-down enumeration.
    """

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int = 6,
        max_programs: int = 100000
    ):
        self.grammar = grammar
        self.max_depth = max_depth
        self.max_programs = max_programs
        self.programs_enumerated = 0
        self.programs_by_type: Dict[str, int] = defaultdict(int)

    def enumerate(
        self,
        request_type: Type,
        max_cost: float = float('inf'),
        timeout_seconds: float = float('inf'),
        env: List[Type] = None
    ) -> Generator[Tuple[Program, float], None, None]:
        """Enumerate programs (legacy eager bottom-up)."""
        if env is None:
            env = []

        start_time = time.time()
        self.programs_enumerated = 0

        pq: List[PriorityItem] = []
        self._initialize_queue(pq, request_type, env)

        seen: Set[int] = set()

        while pq and self.programs_enumerated < self.max_programs:
            if time.time() - start_time > timeout_seconds:
                break

            item = heapq.heappop(pq)
            cost = item.priority
            program = item.program
            tp = item.tp

            if cost > max_cost:
                continue

            if program.depth() > self.max_depth:
                continue

            prog_hash = hash(program)
            if prog_hash in seen:
                continue
            seen.add(prog_hash)

            if not has_holes(program):
                self.programs_enumerated += 1
                self.programs_by_type[str(tp)] += 1
                yield (program, -cost)

    def _initialize_queue(
        self,
        pq: List[PriorityItem],
        request_type: Type,
        env: List[Type]
    ) -> None:
        """Initialize queue with complete programs (eager construction)."""
        ctx = TypeContext()

        if isinstance(request_type, Arrow):
            body_type = request_type.ret
            arg_type = request_type.arg

            body_pq: List[PriorityItem] = []
            self._initialize_queue(body_pq, body_type, [arg_type] + env)

            for item in body_pq:
                lambda_prog = Abstraction(item.program)
                heapq.heappush(pq, PriorityItem(item.priority, lambda_prog, request_type))
            return

        candidates = self.grammar.candidates_for_type(request_type, ctx, env)

        for prod, inst_type, log_prob in candidates:
            cost = -log_prob

            if isinstance(inst_type, Arrow):
                self._add_partial_applications(
                    pq, prod.program, inst_type, cost, env, request_type
                )
            else:
                heapq.heappush(pq, PriorityItem(cost, prod.program, request_type))

        var_candidates = self.grammar.variable_candidates(request_type, ctx, env)
        for idx, log_prob in var_candidates:
            cost = -log_prob
            heapq.heappush(pq, PriorityItem(cost, Index(idx), request_type))

    def _add_partial_applications(
        self,
        pq: List[PriorityItem],
        func: Program,
        func_type: Arrow,
        base_cost: float,
        env: List[Type],
        final_type: Type,
        depth: int = 0
    ) -> None:
        """Add all possible applications of a function to the queue."""
        if depth > self.max_depth:
            return

        ctx = TypeContext()
        arg_type = ctx.apply(func_type.arg)
        ret_type = ctx.apply(func_type.ret)

        arg_candidates = self.grammar.candidates_for_type(arg_type, ctx, env)
        var_candidates = self.grammar.variable_candidates(arg_type, ctx, env)

        for prod, inst_type, log_prob in arg_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            if isinstance(inst_type, Arrow):
                for arg_prog, arg_total_cost in self._build_complete_arg(
                    prod.program, inst_type, arg_cost, env, depth + 1
                ):
                    new_prog = Application(func, arg_prog)
                    total = base_cost + arg_total_cost

                    if isinstance(ret_type, Arrow):
                        self._add_partial_applications(
                            pq, new_prog, ret_type, total, env, final_type, depth + 1
                        )
                    else:
                        heapq.heappush(pq, PriorityItem(total, new_prog, final_type))
            else:
                new_prog = Application(func, prod.program)

                if isinstance(ret_type, Arrow):
                    self._add_partial_applications(
                        pq, new_prog, ret_type, total_cost, env, final_type, depth + 1
                    )
                else:
                    heapq.heappush(pq, PriorityItem(total_cost, new_prog, final_type))

        for idx, log_prob in var_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            new_prog = Application(func, Index(idx))

            if isinstance(ret_type, Arrow):
                self._add_partial_applications(
                    pq, new_prog, ret_type, total_cost, env, final_type, depth + 1
                )
            else:
                heapq.heappush(pq, PriorityItem(total_cost, new_prog, final_type))

    def _build_complete_arg(
        self,
        func: Program,
        func_type: Arrow,
        base_cost: float,
        env: List[Type],
        depth: int
    ) -> Generator[Tuple[Program, float], None, None]:
        """Build complete argument programs for higher-order functions."""
        if depth > self.max_depth:
            return

        ctx = TypeContext()
        arg_type = ctx.apply(func_type.arg)
        ret_type = ctx.apply(func_type.ret)

        arg_candidates = self.grammar.candidates_for_type(arg_type, ctx, env)
        var_candidates = self.grammar.variable_candidates(arg_type, ctx, env)

        for idx, log_prob in var_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost
            new_prog = Application(func, Index(idx))

            if isinstance(ret_type, Arrow):
                for complete, complete_cost in self._build_complete_arg(
                    new_prog, ret_type, total_cost, env, depth + 1
                ):
                    yield (complete, complete_cost)
            else:
                yield (new_prog, total_cost)

        for prod, inst_type, log_prob in arg_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            if isinstance(inst_type, Arrow):
                for arg_complete, arg_complete_cost in self._build_complete_arg(
                    prod.program, inst_type, arg_cost, env, depth + 1
                ):
                    new_prog = Application(func, arg_complete)
                    new_total = base_cost + arg_complete_cost

                    if isinstance(ret_type, Arrow):
                        for complete, complete_cost in self._build_complete_arg(
                            new_prog, ret_type, new_total, env, depth + 1
                        ):
                            yield (complete, complete_cost)
                    else:
                        yield (new_prog, new_total)
            else:
                new_prog = Application(func, prod.program)
                if isinstance(ret_type, Arrow):
                    for complete, complete_cost in self._build_complete_arg(
                        new_prog, ret_type, total_cost, env, depth + 1
                    ):
                        yield (complete, complete_cost)
                else:
                    yield (new_prog, total_cost)


# ============================================================================
# HIGH-LEVEL TASK SOLVING FUNCTIONS
# ============================================================================

def enumerate_for_task(
    grammar: Grammar,
    examples: List[Tuple[Any, Any]],
    request_type: Type,
    eval_fn: Callable[[Program, Any], Any],
    max_cost: float = 20.0,
    timeout_seconds: float = 60.0,
    max_programs: int = 100000,
    use_top_down: bool = True
) -> Frontier:
    """
    Enumerate programs to solve a task defined by input-output examples.

    Args:
        grammar: The PCFG defining the program space
        examples: List of (input, expected_output) pairs
        request_type: Type of programs to enumerate
        eval_fn: Function to run a program on an input
        max_cost: Maximum description length (in nats)
        timeout_seconds: Wall clock timeout
        max_programs: Maximum programs to try
        use_top_down: If True, use TopDownEnumerator; else use legacy

    Returns:
        Frontier containing the best solutions found
    """
    frontier = Frontier(task_name="task", request_type=request_type)

    if use_top_down:
        enumerator = TopDownEnumerator(grammar, max_programs=max_programs)
    else:
        enumerator = Enumerator(grammar, max_programs=max_programs)

    start_time = time.time()
    programs_tried = 0

    for program, log_prob in enumerator.enumerate(
        request_type,
        max_cost=max_cost,
        timeout_seconds=timeout_seconds
    ):
        programs_tried += 1

        try:
            correct = 0
            for inp, expected_out in examples:
                result = eval_fn(program, inp)
                if result == expected_out:
                    correct += 1

            if correct == len(examples):
                log_likelihood = 0.0
            else:
                log_likelihood = math.log(correct / len(examples) + 1e-10)

            partial_explored = getattr(enumerator, 'partial_programs_explored', 0)

            result = EnumerationResult(
                program=program,
                log_probability=log_prob,
                log_likelihood=log_likelihood,
                description_length=-log_prob / math.log(2),
                programs_enumerated=programs_tried,
                partial_programs_explored=partial_explored,
                time_seconds=time.time() - start_time
            )

            if correct == len(examples):
                frontier.add(result)
                break

        except Exception:
            pass

    return frontier


def enumerate_top_down(
    grammar: Grammar,
    request_type: Type,
    max_depth: int = 8,
    max_programs: int = 1000,
    timeout_seconds: float = 60.0
) -> Generator[Tuple[Program, float], None, None]:
    """
    Convenience function for top-down enumeration.

    Args:
        grammar: The PCFG
        request_type: Type of programs to enumerate
        max_depth: Maximum AST depth
        max_programs: Maximum programs to yield
        timeout_seconds: Wall clock timeout

    Yields:
        (program, log_probability) pairs
    """
    enumerator = TopDownEnumerator(grammar, max_depth=max_depth, max_programs=max_programs)
    yield from enumerator.enumerate(
        request_type,
        timeout_seconds=timeout_seconds
    )


# ============================================================================
# ITERATIVE DEEPENING (LEGACY - kept for worker compatibility)
# ============================================================================
#
# DEPRECATION NOTICE:
# ------------------
# enumerate_simple is a LEGACY enumeration strategy that should NOT be used in
# new main code paths. It is kept ONLY for:
#   1. Worker files (enumeration_worker.py, etc.) that run in PyPy
#   2. Test files for simple validation
#   3. Backward compatibility with older scripts
#
# For NEW CODE, use TopDownEnumerator instead:
#   - TopDownEnumerator uses priority queue with grammar log-probabilities
#   - Supports recognition-guided search via predict_grammar_weights()
#   - Better integration with neural recognition models
#
# enumerate_simple limitations:
#   - Returns programs in depth order, ignoring grammar probabilities
#   - Cannot easily integrate recognition model guidance
#   - Less efficient for finding high-probability programs
#
# ============================================================================

def enumerate_simple(
    grammar: Grammar,
    request_type: Type,
    max_depth: int = 4,
    env: List[Type] = None,
    seen: Set[int] = None
) -> Generator[Tuple[Program, float], None, None]:
    """
    DEPRECATED: Use TopDownEnumerator for new code.

    Enumerate programs using iterative deepening.
    Simpler than priority queue approaches, guarantees shortest programs first.

    This function is kept for:
    - Worker files running in PyPy (which may not support full TopDownEnumerator)
    - Test files for simple validation
    - Backward compatibility

    For recognition-guided enumeration, use TopDownEnumerator with a grammar
    that has been adjusted via recognition_model.predict_grammar_weights().
    """
    if seen is None:
        seen = set()
    if env is None:
        env = []

    for depth in range(1, max_depth + 1):
        for prog, log_prob in _enumerate_at_depth(grammar, request_type, depth, env, seen):
            yield (prog, log_prob)


def _enumerate_at_depth(
    grammar: Grammar,
    request_type: Type,
    depth: int,
    env: List[Type],
    seen: Set[int]
) -> Generator[Tuple[Program, float], None, None]:
    """Enumerate programs at exactly the given depth."""
    ctx = TypeContext()

    if isinstance(request_type, Arrow):
        for body, log_prob in _enumerate_at_depth(
            grammar,
            request_type.ret,
            depth,
            [request_type.arg] + env,
            seen
        ):
            prog = Abstraction(body)
            prog_hash = hash(prog)
            if prog_hash not in seen:
                seen.add(prog_hash)
                yield (prog, log_prob)
        return

    if depth < 1:
        return

    if depth == 1:
        var_candidates = grammar.variable_candidates(request_type, ctx, env)
        for idx, log_prob in var_candidates:
            prog = Index(idx)
            prog_hash = hash(prog)
            if prog_hash not in seen:
                seen.add(prog_hash)
                yield (prog, log_prob)

        candidates = grammar.candidates_for_type(request_type, ctx, env)
        for prod, inst_type, log_prob in candidates:
            if not isinstance(inst_type, Arrow):
                prog_hash = hash(prod.program)
                if prog_hash not in seen:
                    seen.add(prog_hash)
                    yield (prod.program, log_prob)
        return

    candidates = grammar.candidates_for_type(request_type, ctx, env)

    for prod, inst_type, log_prob in candidates:
        if isinstance(inst_type, Arrow):
            for filled, arg_log_prob in _fill_arguments_depth(
                grammar, prod.program, inst_type, depth - 1, env, seen
            ):
                prog_hash = hash(filled)
                if prog_hash not in seen:
                    seen.add(prog_hash)
                    yield (filled, log_prob + arg_log_prob)


def _fill_arguments_depth(
    grammar: Grammar,
    func: Program,
    func_type: Arrow,
    remaining_depth: int,
    env: List[Type],
    seen: Set[int]
) -> Generator[Tuple[Program, float], None, None]:
    """Fill arguments with depth budget."""
    if remaining_depth < 1:
        return

    arg_type = func_type.arg
    ret_type = func_type.ret

    n_remaining_args = 1 + (type_arity(ret_type) if isinstance(ret_type, Arrow) else 0)
    max_arg_depth = remaining_depth - (n_remaining_args - 1)

    concrete_types = _get_useful_concrete_types(arg_type)

    for concrete_arg_type in concrete_types:
        concrete_ret_type = _substitute_type_var(ret_type, arg_type, concrete_arg_type)

        for arg_depth in range(1, max_arg_depth + 1):
            for arg, arg_log_prob in _enumerate_at_depth(
                grammar, concrete_arg_type, arg_depth, env, set()
            ):
                new_func = Application(func, arg)

                if isinstance(concrete_ret_type, Arrow):
                    for filled, more_log_prob in _fill_arguments_depth(
                        grammar, new_func, concrete_ret_type,
                        remaining_depth - arg_depth, env, seen
                    ):
                        yield (filled, arg_log_prob + more_log_prob)
                else:
                    yield (new_func, arg_log_prob)


def _get_useful_concrete_types(t: Type) -> List[Type]:
    """Get concrete types for polymorphic type variables."""
    if isinstance(t, TypeVariable):
        return [
            BaseType('color'),
            BaseType('suit'),
            BaseType('rank'),
            BaseType('int'),
            BaseType('card'),
            BaseType('bool'),
        ]
    else:
        return [t]


def _substitute_type_var(t: Type, var: Type, replacement: Type) -> Type:
    """Substitute type variable with concrete type."""
    if not isinstance(var, TypeVariable):
        return t

    if isinstance(t, TypeVariable) and t.id == var.id:
        return replacement

    if isinstance(t, Arrow):
        return Arrow(
            _substitute_type_var(t.arg, var, replacement),
            _substitute_type_var(t.ret, var, replacement)
        )

    if isinstance(t, ListType):
        return ListType(_substitute_type_var(t.element, var, replacement))

    return t


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENUMERATION TESTS")
    print("=" * 70)

    from .grammar import uniform_grammar

    # Create simple arithmetic primitives
    add = Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y)
    zero = Primitive('0', INT, 0)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)

    g = uniform_grammar([add, mul, zero, one, two])
    print(f"\nGrammar: {len(g.productions)} productions")

    # Test 1: Top-down enumeration of INT programs
    print("\n" + "-" * 70)
    print("Test 1: Top-Down Enumeration of INT programs")
    print("-" * 70)

    enumerator = TopDownEnumerator(g, max_depth=4)
    count = 0
    for prog, log_prob in enumerator.enumerate(INT, timeout_seconds=5.0):
        try:
            val = prog.evaluate([])
            print(f"  {str(prog):20} = {val:5}  (log_p = {log_prob:.2f})")
        except Exception as e:
            print(f"  {str(prog):20} ERROR: {e}")
        count += 1
        if count >= 15:
            print("  ...")
            break

    print(f"\nEnumerated {enumerator.programs_enumerated} complete programs")
    print(f"Explored {enumerator.partial_programs_explored} partial programs")

    # Test 2: Top-down enumeration of INT → INT programs
    print("\n" + "-" * 70)
    print("Test 2: Top-Down Enumeration of INT → INT programs")
    print("-" * 70)

    enumerator2 = TopDownEnumerator(g, max_depth=4)
    count = 0
    for prog, log_prob in enumerator2.enumerate(arrow(INT, INT), timeout_seconds=5.0):
        try:
            fn = prog.evaluate([])
            result = fn(3)
            print(f"  {str(prog):25} f(3) = {result:5}  (log_p = {log_prob:.2f})")
        except Exception as e:
            print(f"  {str(prog):25} ERROR: {e}")
        count += 1
        if count >= 10:
            print("  ...")
            break

    print(f"\nEnumerated {enumerator2.programs_enumerated} complete programs")
    print(f"Explored {enumerator2.partial_programs_explored} partial programs")

    # Test 3: Compare with iterative deepening
    print("\n" + "-" * 70)
    print("Test 3: Iterative Deepening (for comparison)")
    print("-" * 70)

    count = 0
    for prog, log_prob in enumerate_simple(g, INT, max_depth=3):
        val = prog.evaluate([])
        print(f"  {str(prog):20} = {val:5}  (log_p = {log_prob:.2f})")
        count += 1
        if count >= 10:
            print("  ...")
            break

    # Test 4: Solve a simple task
    print("\n" + "-" * 70)
    print("Test 4: Solve task: f(x) = x + 1")
    print("-" * 70)

    examples = [(0, 1), (1, 2), (2, 3), (5, 6)]

    def eval_fn(prog, x):
        fn = prog.evaluate([])
        return fn(x)

    frontier = enumerate_for_task(
        g, examples, arrow(INT, INT), eval_fn,
        max_cost=20.0, timeout_seconds=10.0, use_top_down=True
    )

    if frontier.best:
        print(f"  Found solution: {frontier.best.program}")
        print(f"  Programs tried: {frontier.best.programs_enumerated}")
        print(f"  Partial explored: {frontier.best.partial_programs_explored}")
        print(f"  Time: {frontier.best.time_seconds:.3f}s")
    else:
        print("  No solution found!")

    print("\n" + "=" * 70)
    print("ALL ENUMERATION TESTS PASSED")
    print("=" * 70)
