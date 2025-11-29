# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cythonized Program Enumeration for DreamCoder

Optimized version of enumeration.py with:
- cdef classes for speed
- Typed attributes and local variables
- Disabled bounds checking for performance

The enumeration is the "wake" phase of DreamCoder - searching for
programs that solve given tasks.
"""

from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import heapq
import math
import time
from collections import defaultdict

# Import from Cython modules
from .type_system_cy import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow, type_arity
)
from .program_cy import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args
)
from .grammar_cy import Grammar, Production


# ============================================================================
# DATA CLASSES (using cdef class for performance)
# ============================================================================

cdef class EnumerationResult:
    """Result of enumerating programs for a task."""
    cdef readonly object program
    cdef readonly double log_probability
    cdef readonly double log_likelihood
    cdef readonly double description_length
    cdef readonly int programs_enumerated
    cdef readonly double time_seconds

    def __init__(self, object program, double log_probability, double log_likelihood,
                 double description_length, int programs_enumerated, double time_seconds):
        self.program = program
        self.log_probability = log_probability
        self.log_likelihood = log_likelihood
        self.description_length = description_length
        self.programs_enumerated = programs_enumerated
        self.time_seconds = time_seconds


def _frontier_sort_key(e):
    """Sort key for frontier entries - lower is better."""
    return e.description_length - e.log_likelihood


cdef class Frontier:
    """
    A frontier of candidate programs for a task.

    Stores the best programs found so far, sorted by total score
    (description length + log likelihood).
    """
    cdef readonly str task_name
    cdef readonly object request_type
    cdef public list entries
    cdef readonly int max_size

    def __init__(self, str task_name, object request_type, int max_size=10):
        self.task_name = task_name
        self.request_type = request_type
        self.entries = []
        self.max_size = max_size

    cpdef bint add(self, EnumerationResult result):
        """Add a result if it improves the frontier. Returns True if added."""
        cdef EnumerationResult e
        # Check if this program is already in the frontier
        for e in self.entries:
            if e.program == result.program:
                return False

        self.entries.append(result)
        # Sort by total score (lower is better)
        self.entries.sort(key=_frontier_sort_key)
        # Keep only the best
        if len(self.entries) > self.max_size:
            self.entries = self.entries[:self.max_size]
            return result in self.entries
        return True

    @property
    def best(self):
        """Return the best program found."""
        return self.entries[0] if self.entries else None

    @property
    def empty(self):
        return len(self.entries) == 0


# ============================================================================
# PRIORITY QUEUE ITEM
# ============================================================================

cdef class PriorityItem:
    """Item in the enumeration priority queue."""
    cdef public double priority
    cdef public object program
    cdef public object tp

    def __init__(self, double priority, object program, object tp):
        self.priority = priority
        self.program = program
        self.tp = tp

    def __lt__(self, other):
        return self.priority < (<PriorityItem>other).priority

    def __eq__(self, other):
        return self.priority == (<PriorityItem>other).priority


# ============================================================================
# ENUMERATOR CLASS
# ============================================================================

cdef class Enumerator:
    """
    Best-first program enumerator.

    Enumerates programs in order of their description length (grammar probability),
    respecting type constraints.
    """
    cdef public object grammar
    cdef public int max_depth
    cdef public int max_programs
    cdef public int programs_enumerated
    cdef public dict programs_by_type

    def __init__(self, object grammar, int max_depth=6, int max_programs=100000):
        """
        Initialize enumerator.

        Args:
            grammar: The PCFG to sample from
            max_depth: Maximum AST depth
            max_programs: Maximum programs to enumerate
        """
        self.grammar = grammar
        self.max_depth = max_depth
        self.max_programs = max_programs
        self.programs_enumerated = 0
        self.programs_by_type = defaultdict(int)

    def enumerate(self, object request_type, double max_cost=1e308,
                  double timeout_seconds=1e308, list env=None):
        """
        Enumerate programs of the given type in order of description length.

        Yields (program, log_probability) pairs.
        """
        if env is None:
            env = []

        cdef double start_time = time.time()
        self.programs_enumerated = 0

        # Priority queue: (cost, program, type)
        cdef list pq = []
        cdef set seen = set()
        cdef PriorityItem item
        cdef double cost
        cdef object program
        cdef object tp
        cdef str prog_str

        # Initialize with all ways to start a program of request_type
        self._initialize_queue(pq, request_type, env)

        while pq and self.programs_enumerated < self.max_programs:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            item = heapq.heappop(pq)
            cost = item.priority
            program = item.program
            tp = item.tp

            # Check cost bound
            if cost > max_cost:
                continue

            # Check depth
            if program.depth() > self.max_depth:
                continue

            # Skip duplicates
            prog_str = str(program)
            if prog_str in seen:
                continue
            seen.add(prog_str)

            # If the program is complete (no holes), yield it
            if self._is_complete(program):
                self.programs_enumerated += 1
                self.programs_by_type[str(tp)] += 1
                yield (program, -cost)  # Convert cost back to log-prob
            else:
                # Expand the program (fill in a hole)
                self._expand_program(pq, program, tp, cost, env)

    cdef void _initialize_queue(self, list pq, object request_type, list env):
        """Initialize the priority queue with starting programs."""
        cdef object ctx = TypeContext()
        cdef list body_pq
        cdef PriorityItem item
        cdef object lambda_prog
        cdef list candidates
        cdef tuple cand
        cdef object prod
        cdef object inst_type
        cdef double log_prob
        cdef double cost
        cdef int n_args
        cdef list var_candidates
        cdef int idx

        # If request type is an arrow, we need to produce a lambda
        if isinstance(request_type, Arrow):
            body_type = request_type.ret
            arg_type = request_type.arg

            body_pq = []
            self._initialize_queue(body_pq, body_type, [arg_type] + env)

            for item in body_pq:
                lambda_prog = Abstraction(item.program)
                heapq.heappush(pq, PriorityItem(item.priority, lambda_prog, request_type))
            return

        # Add primitives that produce request_type
        candidates = self.grammar.candidates_for_type(request_type, ctx, env)
        for cand in candidates:
            prod = cand[0]
            inst_type = cand[1]
            log_prob = cand[2]
            cost = -log_prob

            if isinstance(inst_type, Arrow):
                self._add_partial_applications(pq, prod.program, inst_type, cost, env, request_type, 0)
            else:
                heapq.heappush(pq, PriorityItem(cost, prod.program, request_type))

        # Add variables from environment
        var_candidates = self.grammar.variable_candidates(request_type, ctx, env)
        for var_cand in var_candidates:
            idx = var_cand[0]
            log_prob = var_cand[1]
            cost = -log_prob
            heapq.heappush(pq, PriorityItem(cost, Index(idx), request_type))

    cdef void _add_partial_applications(self, list pq, object func, object func_type,
                                         double base_cost, list env, object final_type, int depth):
        """Add partial applications of a function to the queue."""
        if depth > self.max_depth:
            return

        cdef object ctx = TypeContext()
        cdef object arg_type = ctx.apply(func_type.arg)
        cdef object ret_type = ctx.apply(func_type.ret)

        # Get all possible arguments
        cdef list arg_candidates = self.grammar.candidates_for_type(arg_type, ctx, env)
        cdef list var_candidates = self.grammar.variable_candidates(arg_type, ctx, env)

        cdef tuple cand
        cdef object prod
        cdef object inst_type
        cdef double log_prob
        cdef double arg_cost
        cdef double total_cost
        cdef object new_prog
        cdef int idx

        # For each possible first argument
        for cand in arg_candidates:
            prod = cand[0]
            inst_type = cand[1]
            log_prob = cand[2]
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            if isinstance(inst_type, Arrow):
                # Argument needs its own arguments - build complete args
                for arg_prog, arg_total_cost in self._build_complete_arg(
                    prod.program, inst_type, arg_cost, env, depth + 1
                ):
                    new_prog = Application(func, arg_prog)
                    total = base_cost + arg_total_cost
                    if isinstance(ret_type, Arrow):
                        self._add_partial_applications(pq, new_prog, ret_type, total, env, final_type, depth + 1)
                    else:
                        heapq.heappush(pq, PriorityItem(total, new_prog, final_type))
            else:
                # Simple argument
                new_prog = Application(func, prod.program)
                if isinstance(ret_type, Arrow):
                    self._add_partial_applications(pq, new_prog, ret_type, total_cost, env, final_type, depth + 1)
                else:
                    heapq.heappush(pq, PriorityItem(total_cost, new_prog, final_type))

        # Variable arguments
        for var_cand in var_candidates:
            idx = var_cand[0]
            log_prob = var_cand[1]
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            new_prog = Application(func, Index(idx))
            if isinstance(ret_type, Arrow):
                self._add_partial_applications(pq, new_prog, ret_type, total_cost, env, final_type, depth + 1)
            else:
                heapq.heappush(pq, PriorityItem(total_cost, new_prog, final_type))

    def _build_complete_arg(self, object func, object func_type, double base_cost,
                            list env, int depth):
        """Build complete arguments that need their own arguments."""
        if depth > self.max_depth:
            return

        cdef object ctx = TypeContext()
        cdef object arg_type = ctx.apply(func_type.arg)
        cdef object ret_type = ctx.apply(func_type.ret)

        cdef list arg_candidates = self.grammar.candidates_for_type(arg_type, ctx, env)
        cdef list var_candidates = self.grammar.variable_candidates(arg_type, ctx, env)

        cdef tuple cand
        cdef object prod
        cdef object inst_type
        cdef double log_prob
        cdef double arg_cost
        cdef double total_cost
        cdef object new_prog
        cdef int idx

        # Variable arguments (simplest)
        for var_cand in var_candidates:
            idx = var_cand[0]
            log_prob = var_cand[1]
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

        # Primitive arguments
        for cand in arg_candidates:
            prod = cand[0]
            inst_type = cand[1]
            log_prob = cand[2]
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

    cdef bint _is_complete(self, object program):
        """Check if a program has no holes (is complete)."""
        return True

    cdef void _expand_program(self, list pq, object program, object tp,
                               double cost, list env):
        """Expand a partial program by filling in a hole."""
        pass


# ============================================================================
# SIMPLE ENUMERATION WITH PROPER DEPTH HANDLING
# ============================================================================

def enumerate_simple(object grammar, object request_type, int max_depth=4,
                     list env=None, set seen=None):
    """
    Enumerate programs of a given type, up to max_depth.

    Uses iterative deepening to find shorter programs first.
    """
    if seen is None:
        seen = set()
    if env is None:
        env = []

    cdef int depth
    # Iterative deepening: try depth 1, then 2, etc.
    for depth in range(1, max_depth + 1):
        for prog, log_prob in _enumerate_at_depth(grammar, request_type, depth, env, seen):
            yield (prog, log_prob)


def _enumerate_at_depth(object grammar, object request_type, int depth, list env, set seen):
    """Enumerate programs at exactly the given depth."""
    cdef object ctx = TypeContext()
    cdef list var_candidates
    cdef list candidates
    cdef tuple cand
    cdef object prod
    cdef object inst_type
    cdef double log_prob
    cdef str prog_str
    cdef object prog
    cdef int idx

    # Handle arrow types by creating lambdas
    if isinstance(request_type, Arrow):
        for body, log_prob in _enumerate_at_depth(
            grammar,
            request_type.ret,
            depth,  # Lambda doesn't consume depth
            [request_type.arg] + env,
            seen
        ):
            prog = Abstraction(body)
            prog_str = str(prog)
            if prog_str not in seen:
                seen.add(prog_str)
                yield (prog, log_prob)
        return

    if depth < 1:
        return

    # Depth 1: just primitives with no arguments, or variables
    if depth == 1:
        # Variables
        var_candidates = grammar.variable_candidates(request_type, ctx, env)
        for var_cand in var_candidates:
            idx = var_cand[0]
            log_prob = var_cand[1]
            prog = Index(idx)
            prog_str = str(prog)
            if prog_str not in seen:
                seen.add(prog_str)
                yield (prog, log_prob)

        # Primitives with no arguments
        candidates = grammar.candidates_for_type(request_type, ctx, env)
        for cand in candidates:
            prod = cand[0]
            inst_type = cand[1]
            log_prob = cand[2]
            if not isinstance(inst_type, Arrow):
                prog_str = str(prod.program)
                if prog_str not in seen:
                    seen.add(prog_str)
                    yield (prod.program, log_prob)
        return

    # Depth > 1: primitives with arguments
    candidates = grammar.candidates_for_type(request_type, ctx, env)
    for cand in candidates:
        prod = cand[0]
        inst_type = cand[1]
        log_prob = cand[2]
        if isinstance(inst_type, Arrow):
            # Need to fill arguments - distribute remaining depth among args
            for filled, arg_log_prob in _fill_arguments_depth(
                grammar, prod.program, inst_type, depth - 1, env, seen
            ):
                prog_str = str(filled)
                if prog_str not in seen:
                    seen.add(prog_str)
                    yield (filled, log_prob + arg_log_prob)


def _fill_arguments_depth(object grammar, object func, object func_type,
                          int remaining_depth, list env, set seen):
    """Fill in arguments for a function with depth budget."""
    if remaining_depth < 1:
        return

    cdef object arg_type = func_type.arg
    cdef object ret_type = func_type.ret

    # Count how many arguments we need
    cdef int n_remaining_args = 1 + (type_arity(ret_type) if isinstance(ret_type, Arrow) else 0)

    # For this arg, try all depths from 1 to remaining_depth - (n_remaining_args - 1)
    cdef int max_arg_depth = remaining_depth - (n_remaining_args - 1)

    # Handle polymorphic types by trying only the most useful concrete types
    cdef list concrete_types = _get_useful_concrete_types(arg_type)
    cdef object concrete_arg_type
    cdef object concrete_ret_type
    cdef int arg_depth
    cdef object arg
    cdef double arg_log_prob
    cdef object new_func
    cdef object filled
    cdef double more_log_prob

    for concrete_arg_type in concrete_types:
        # Also update ret_type if it shares the same type variable
        concrete_ret_type = _substitute_type_var(ret_type, arg_type, concrete_arg_type)

        for arg_depth in range(1, max_arg_depth + 1):
            for arg, arg_log_prob in _enumerate_at_depth(grammar, concrete_arg_type, arg_depth, env, set()):
                new_func = Application(func, arg)

                if isinstance(concrete_ret_type, Arrow):
                    # More arguments needed
                    for filled, more_log_prob in _fill_arguments_depth(
                        grammar, new_func, concrete_ret_type, remaining_depth - arg_depth, env, seen
                    ):
                        yield (filled, arg_log_prob + more_log_prob)
                else:
                    # Done
                    yield (new_func, arg_log_prob)


cdef list _get_useful_concrete_types(object t):
    """
    For a type variable, return the most useful concrete types.
    Prioritize types that are commonly compared in card games.
    """
    if isinstance(t, TypeVariable):
        return [
            BaseType('color'),   # Most common comparison
            BaseType('suit'),    # Common comparison
            BaseType('rank'),    # Common comparison
            BaseType('int'),     # For numeric comparisons
            BaseType('card'),    # Full card comparison
            BaseType('bool'),    # Boolean comparison
        ]
    else:
        return [t]


cdef object _substitute_type_var(object t, object var, object replacement):
    """Substitute a type variable with a concrete type."""
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
# TASK ENUMERATION HELPER
# ============================================================================

def enumerate_for_task(object grammar, list examples, object request_type,
                       object eval_fn, double max_cost=20.0,
                       double timeout_seconds=60.0, int max_programs=100000):
    """
    Enumerate programs to solve a task defined by input-output examples.

    This is the main entry point for the "wake" phase.

    Args:
        grammar: The PCFG
        examples: List of (input, output) pairs
        request_type: Type of programs to enumerate (usually input -> output)
        eval_fn: Function to evaluate a program on an input
        max_cost: Maximum description length in bits
        timeout_seconds: Timeout
        max_programs: Max programs to try

    Returns:
        Frontier of best programs found
    """
    cdef Frontier frontier = Frontier(task_name="task", request_type=request_type)
    cdef Enumerator enumerator = Enumerator(grammar, max_programs=max_programs)

    cdef double start_time = time.time()
    cdef int programs_tried = 0
    cdef object program
    cdef double log_prob
    cdef int correct
    cdef object inp
    cdef object expected_out
    cdef object result
    cdef double log_likelihood
    cdef EnumerationResult enum_result

    for program, log_prob in enumerator.enumerate(
        request_type,
        max_cost=max_cost,
        timeout_seconds=timeout_seconds
    ):
        programs_tried += 1

        # Evaluate on examples
        try:
            correct = 0
            for inp, expected_out in examples:
                result = eval_fn(program, inp)
                if result == expected_out:
                    correct += 1

            if correct == len(examples):
                # Found a solution!
                log_likelihood = 0.0  # Perfect fit
            else:
                # Partial fit
                log_likelihood = math.log(correct / len(examples) + 1e-10)

            enum_result = EnumerationResult(
                program=program,
                log_probability=log_prob,
                log_likelihood=log_likelihood,
                description_length=-log_prob / math.log(2),
                programs_enumerated=programs_tried,
                time_seconds=time.time() - start_time
            )

            # Add to frontier if it's a perfect solution
            if correct == len(examples):
                frontier.add(enum_result)
                # Early exit if we found a solution
                break

        except Exception as e:
            # Program crashed - skip it
            pass

    return frontier


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Enumeration Tests ===\n")

    from .grammar_cy import uniform_grammar

    # Create simple primitives
    add = Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y)
    zero = Primitive('0', INT, 0)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)

    g = uniform_grammar([add, mul, zero, one, two])
    print(g)

    # Enumerate INT programs
    print("\n=== Enumerating INT programs ===")
    count = 0
    for prog, log_prob in enumerate_simple(g, INT, max_depth=3):
        print(f"  {prog} : log_p = {log_prob:.2f}, value = {prog.evaluate([])}")
        count += 1
        if count >= 15:
            print("  ...")
            break

    print("\n=== Enumeration Tests OK ===")
