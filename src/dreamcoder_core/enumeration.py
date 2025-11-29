"""
Program Enumeration for DreamCoder

This module implements best-first enumeration of programs guided by:
1. Grammar probabilities (description length)
2. Type constraints (only type-correct programs)
3. Budget constraints (max description length)

The enumeration is the "wake" phase of DreamCoder - searching for
programs that solve given tasks.

Key metrics for cognitive realism:
- Number of programs enumerated before finding a solution
- Time/effort to solve each task
- How enumeration budget affects success
"""

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
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args
)
from .grammar import Grammar, Production


@dataclass
class EnumerationResult:
    """Result of enumerating programs for a task."""
    program: Program
    log_probability: float  # Grammar probability
    log_likelihood: float   # How well it fits the examples
    description_length: float  # = -log_probability
    programs_enumerated: int  # How many programs were tried
    time_seconds: float  # Wall clock time


@dataclass
class Frontier:
    """
    A frontier of candidate programs for a task.

    Stores the best programs found so far, sorted by total score
    (description length + log likelihood).
    """
    task_name: str
    request_type: Type
    entries: List[EnumerationResult] = field(default_factory=list)
    max_size: int = 10

    def add(self, result: EnumerationResult) -> bool:
        """Add a result if it improves the frontier. Returns True if added."""
        # Check if this program is already in the frontier
        for e in self.entries:
            if e.program == result.program:
                return False

        self.entries.append(result)
        # Sort by total score (lower is better)
        self.entries.sort(key=lambda e: e.description_length - e.log_likelihood)
        # Keep only the best
        if len(self.entries) > self.max_size:
            self.entries = self.entries[:self.max_size]
            return result in self.entries
        return True

    @property
    def best(self) -> Optional[EnumerationResult]:
        """Return the best program found."""
        return self.entries[0] if self.entries else None

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0


@dataclass(order=True)
class PriorityItem:
    """Item in the enumeration priority queue."""
    priority: float  # Negative log probability (lower = higher priority)
    program: Program = field(compare=False)
    tp: Type = field(compare=False)


class Enumerator:
    """
    Best-first program enumerator.

    Enumerates programs in order of their description length (grammar probability),
    respecting type constraints.

    This is a simplified version of DreamCoder's enumeration that:
    1. Uses Python instead of OCaml for clarity
    2. Tracks enumeration metrics for cognitive analysis
    3. Supports timeout and budget constraints
    """

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int = 6,
        max_programs: int = 100000
    ):
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

        # Statistics
        self.programs_enumerated = 0
        self.programs_by_type: Dict[str, int] = defaultdict(int)

    def enumerate(
        self,
        request_type: Type,
        max_cost: float = float('inf'),
        timeout_seconds: float = float('inf'),
        env: List[Type] = None
    ) -> Generator[Tuple[Program, float], None, None]:
        """
        Enumerate programs of the given type in order of description length.

        Yields (program, log_probability) pairs.

        Args:
            request_type: The type of programs to enumerate
            max_cost: Maximum description length (bits)
            timeout_seconds: Wall clock timeout
            env: Type environment for bound variables
        """
        if env is None:
            env = []

        start_time = time.time()
        self.programs_enumerated = 0

        # Priority queue: (cost, program, type)
        # We enumerate partial programs and complete them
        pq: List[PriorityItem] = []

        # Initialize with all ways to start a program of request_type
        self._initialize_queue(pq, request_type, env)

        seen: Set[str] = set()

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

    def _initialize_queue(
        self,
        pq: List[PriorityItem],
        request_type: Type,
        env: List[Type]
    ) -> None:
        """Initialize the priority queue with starting programs."""

        ctx = TypeContext()

        # If request type is an arrow, we need to produce a lambda
        if isinstance(request_type, Arrow):
            # Start with (λ. ?body?) where body has type request_type.ret
            # But the body can use $0 which has type request_type.arg
            body_type = request_type.ret
            arg_type = request_type.arg

            # Recursively initialize for the body type
            body_pq: List[PriorityItem] = []
            self._initialize_queue(body_pq, body_type, [arg_type] + env)

            for item in body_pq:
                lambda_prog = Abstraction(item.program)
                heapq.heappush(pq, PriorityItem(item.priority, lambda_prog, request_type))
            return

        # Add primitives that produce request_type
        candidates = self.grammar.candidates_for_type(request_type, ctx, env)
        for prod, inst_type, log_prob in candidates:
            cost = -log_prob

            # If the primitive needs arguments, start building applications
            if isinstance(inst_type, Arrow):
                # Need to apply arguments
                n_args = type_arity(inst_type)
                arg_types = inst_type.arguments

                # Create program with "holes" for arguments
                # We represent this by pushing partial applications
                self._add_partial_applications(pq, prod.program, inst_type, cost, env, request_type)
            else:
                # No arguments needed - this is a complete program
                heapq.heappush(pq, PriorityItem(cost, prod.program, request_type))

        # Add variables from environment
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
        """Add partial applications of a function to the queue."""

        if depth > self.max_depth:
            return

        ctx = TypeContext()
        arg_type = ctx.apply(func_type.arg)
        ret_type = ctx.apply(func_type.ret)

        # Get all possible arguments
        arg_candidates = self.grammar.candidates_for_type(arg_type, ctx, env)
        var_candidates = self.grammar.variable_candidates(arg_type, ctx, env)

        # For each possible first argument
        for prod, inst_type, log_prob in arg_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            if isinstance(inst_type, Arrow):
                # Argument needs its own arguments - recurse to build it
                # Build all possible complete arguments of this type
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
                    # Need more arguments
                    self._add_partial_applications(pq, new_prog, ret_type, total_cost, env, final_type, depth + 1)
                else:
                    # Complete!
                    heapq.heappush(pq, PriorityItem(total_cost, new_prog, final_type))

        # Variable arguments
        for idx, log_prob in var_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            new_prog = Application(func, Index(idx))
            if isinstance(ret_type, Arrow):
                self._add_partial_applications(pq, new_prog, ret_type, total_cost, env, final_type, depth + 1)
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
        """Build complete arguments that need their own arguments."""

        if depth > self.max_depth:
            return

        ctx = TypeContext()
        arg_type = ctx.apply(func_type.arg)
        ret_type = ctx.apply(func_type.ret)

        # Get argument candidates
        arg_candidates = self.grammar.candidates_for_type(arg_type, ctx, env)
        var_candidates = self.grammar.variable_candidates(arg_type, ctx, env)

        # Variable arguments (simplest)
        for idx, log_prob in var_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost
            new_prog = Application(func, Index(idx))

            if isinstance(ret_type, Arrow):
                # Need more arguments for this
                for complete, complete_cost in self._build_complete_arg(
                    new_prog, ret_type, total_cost, env, depth + 1
                ):
                    yield (complete, complete_cost)
            else:
                yield (new_prog, total_cost)

        # Primitive arguments
        for prod, inst_type, log_prob in arg_candidates:
            arg_cost = -log_prob
            total_cost = base_cost + arg_cost

            if isinstance(inst_type, Arrow):
                # This arg also needs args - recurse
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

    def _is_complete(self, program: Program) -> bool:
        """Check if a program has no holes (is complete)."""
        # In our representation, all programs from the queue are complete
        # (we don't use explicit holes)
        return True

    def _expand_program(
        self,
        pq: List[PriorityItem],
        program: Program,
        tp: Type,
        cost: float,
        env: List[Type]
    ) -> None:
        """Expand a partial program by filling in a hole."""
        # Not used in current implementation - we build complete programs directly
        pass


def enumerate_for_task(
    grammar: Grammar,
    examples: List[Tuple[Any, Any]],  # [(input, output), ...]
    request_type: Type,
    eval_fn: Callable[[Program, Any], Any],  # How to evaluate program on input
    max_cost: float = 20.0,
    timeout_seconds: float = 60.0,
    max_programs: int = 100000
) -> Frontier:
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
    frontier = Frontier(task_name="task", request_type=request_type)
    enumerator = Enumerator(grammar, max_programs=max_programs)

    start_time = time.time()
    programs_tried = 0

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

            result = EnumerationResult(
                program=program,
                log_probability=log_prob,
                log_likelihood=log_likelihood,
                description_length=-log_prob / math.log(2),
                programs_enumerated=programs_tried,
                time_seconds=time.time() - start_time
            )

            # Add to frontier if it's a perfect solution
            if correct == len(examples):
                frontier.add(result)
                # Early exit if we found a solution
                break

        except Exception as e:
            # Program crashed - skip it
            pass

    return frontier


# ============================================================================
# SIMPLE ENUMERATION WITH PROPER DEPTH HANDLING
# ============================================================================

def enumerate_simple(
    grammar: Grammar,
    request_type: Type,
    max_depth: int = 4,
    env: List[Type] = None,
    seen: Set[str] = None
) -> Generator[Tuple[Program, float], None, None]:
    """
    Enumerate programs of a given type, up to max_depth.

    Uses iterative deepening to find shorter programs first.
    """
    if seen is None:
        seen = set()
    if env is None:
        env = []

    # Iterative deepening: try depth 1, then 2, etc.
    for depth in range(1, max_depth + 1):
        for prog, log_prob in _enumerate_at_depth(grammar, request_type, depth, env, seen):
            yield (prog, log_prob)


def _enumerate_at_depth(
    grammar: Grammar,
    request_type: Type,
    depth: int,
    env: List[Type],
    seen: Set[str]
) -> Generator[Tuple[Program, float], None, None]:
    """Enumerate programs at exactly the given depth."""

    ctx = TypeContext()

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
        for idx, log_prob in var_candidates:
            prog = Index(idx)
            prog_str = str(prog)
            if prog_str not in seen:
                seen.add(prog_str)
                yield (prog, log_prob)

        # Primitives with no arguments
        candidates = grammar.candidates_for_type(request_type, ctx, env)
        for prod, inst_type, log_prob in candidates:
            if not isinstance(inst_type, Arrow):
                prog_str = str(prod.program)
                if prog_str not in seen:
                    seen.add(prog_str)
                    yield (prod.program, log_prob)
        return

    # Depth > 1: primitives with arguments
    candidates = grammar.candidates_for_type(request_type, ctx, env)
    for prod, inst_type, log_prob in candidates:
        if isinstance(inst_type, Arrow):
            # Need to fill arguments - distribute remaining depth among args
            for filled, arg_log_prob in _fill_arguments_depth(
                grammar, prod.program, inst_type, depth - 1, env, seen
            ):
                prog_str = str(filled)
                if prog_str not in seen:
                    seen.add(prog_str)
                    yield (filled, log_prob + arg_log_prob)


def _fill_arguments_depth(
    grammar: Grammar,
    func: Program,
    func_type: Arrow,
    remaining_depth: int,
    env: List[Type],
    seen: Set[str]
) -> Generator[Tuple[Program, float], None, None]:
    """Fill in arguments for a function with depth budget."""

    if remaining_depth < 1:
        return

    arg_type = func_type.arg
    ret_type = func_type.ret

    # Count how many arguments we need
    n_remaining_args = 1 + (type_arity(ret_type) if isinstance(ret_type, Arrow) else 0)

    # For this arg, try all depths from 1 to remaining_depth - (n_remaining_args - 1)
    # to leave at least 1 depth for each remaining arg
    max_arg_depth = remaining_depth - (n_remaining_args - 1)

    # Handle polymorphic types by trying only the most useful concrete types
    # to avoid combinatorial explosion
    concrete_types = _get_useful_concrete_types(arg_type)

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


def _get_useful_concrete_types(t: Type) -> List[Type]:
    """
    For a type variable, return the most useful concrete types.
    Prioritize types that are commonly compared in card games.
    """
    if isinstance(t, TypeVariable):
        # Prioritize card game relevant types first
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


def _substitute_type_var(t: Type, var: Type, replacement: Type) -> Type:
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
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Enumeration Tests ===\n")

    from .grammar import uniform_grammar

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

    # Enumerate INT -> INT programs (functions)
    print("\n=== Enumerating INT -> INT programs ===")
    count = 0
    for prog, log_prob in enumerate_simple(g, arrow(INT, INT), max_depth=3):
        print(f"  {prog} : log_p = {log_prob:.2f}")
        # Test the function
        fn = prog.evaluate([])
        print(f"    f(3) = {fn(3)}")
        count += 1
        if count >= 10:
            print("  ...")
            break

    print("\n=== Enumeration Tests OK ===")
