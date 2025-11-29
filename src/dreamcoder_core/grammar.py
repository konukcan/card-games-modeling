"""
Probabilistic Context-Free Grammar (PCFG) for Program Synthesis

This module implements DreamCoder's grammar representation:
- Each production has a log-probability
- Programs are scored by their description length (negative log-probability)
- The grammar can be updated via inside-outside algorithm or compression

The grammar generates programs top-down, making type-directed choices.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import math
import random
from collections import defaultdict

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow,
    canonical_type, type_arity
)
from .program import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args, multi_lambda
)


@dataclass
class Production:
    """
    A production in the grammar.

    Each production associates a primitive/invented with:
    - Its type
    - A log-probability weight
    """
    program: Union[Primitive, Invented]
    tp: Type
    log_probability: float

    def __str__(self) -> str:
        return f"{self.program} : {self.tp} [log_p={self.log_probability:.2f}]"


class Grammar:
    """
    A probabilistic context-free grammar for program synthesis.

    The grammar consists of:
    - A set of typed productions (primitives and invented abstractions)
    - A log-probability for using a bound variable (log_variable)

    Programs are generated top-down:
    1. Given a target type, enumerate all productions that could produce it
    2. Choose one according to probabilities
    3. Recursively fill in argument types

    Description length = -log P(program | grammar)
    """

    def __init__(
        self,
        productions: List[Production],
        log_variable: float = -1.0,
        continuation_type: Optional[Type] = None
    ):
        """
        Initialize a grammar.

        Args:
            productions: List of typed productions with log-probabilities
            log_variable: Log-probability of using a bound variable
            continuation_type: For imperative programs (not used here)
        """
        self.productions = list(productions)
        self.log_variable = log_variable
        self.continuation_type = continuation_type

        # Build index by primitive/invented for fast lookup
        self._by_name: Dict[str, Production] = {}
        for p in self.productions:
            key = str(p.program)
            self._by_name[key] = p

    def __str__(self) -> str:
        lines = ["Grammar:"]
        lines.append(f"  log_variable = {self.log_variable:.2f}")
        lines.append(f"  {len(self.productions)} productions:")
        for p in sorted(self.productions, key=lambda p: -p.log_probability):
            lines.append(f"    {p}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.productions)

    def primitives(self) -> List[Union[Primitive, Invented]]:
        """Return all primitives/invented in the grammar."""
        return [p.program for p in self.productions]

    def get_production(self, program: Union[Primitive, Invented]) -> Optional[Production]:
        """Look up a production by its program."""
        key = str(program)
        return self._by_name.get(key)

    def log_probability(self, program: Union[Primitive, Invented]) -> float:
        """Get the log-probability of a production."""
        prod = self.get_production(program)
        if prod is None:
            return float('-inf')
        return prod.log_probability

    def candidates_for_type(
        self,
        target_type: Type,
        ctx: TypeContext,
        env: List[Type]
    ) -> List[Tuple[Production, Type, float]]:
        """
        Find all productions that could produce the target type.

        Returns list of (production, instantiated_type, log_probability).
        """
        candidates = []

        for prod in self.productions:
            # Try to unify the production's return type with target
            try:
                # Create fresh type variables for this production
                fresh_ctx = TypeContext()
                inst_type = fresh_ctx.instantiate(prod.tp)

                # Get the return type
                ret_type = inst_type.returns

                # Try to unify
                fresh_ctx.unify(ret_type, target_type)
                final_type = fresh_ctx.apply(inst_type)

                candidates.append((prod, final_type, prod.log_probability))
            except UnificationError:
                continue

        return candidates

    def variable_candidates(
        self,
        target_type: Type,
        ctx: TypeContext,
        env: List[Type]
    ) -> List[Tuple[int, float]]:
        """
        Find all bound variables that match the target type.

        Returns list of (de_bruijn_index, log_probability).
        """
        candidates = []

        for i, var_type in enumerate(env):
            try:
                fresh_ctx = TypeContext()
                inst_var_type = fresh_ctx.instantiate(var_type)
                fresh_ctx.unify(inst_var_type, target_type)
                candidates.append((i, self.log_variable))
            except UnificationError:
                continue

        return candidates

    def program_log_likelihood(
        self,
        program: Program,
        request_type: Type,
        env: List[Type] = None
    ) -> float:
        """
        Compute log P(program | grammar, request_type).

        This is the description length (lower = shorter = better).
        """
        if env is None:
            env = []

        ctx = TypeContext()

        return self._ll_helper(program, request_type, ctx, env)

    def _ll_helper(
        self,
        program: Program,
        target_type: Type,
        ctx: TypeContext,
        env: List[Type]
    ) -> float:
        """Recursive helper for computing log-likelihood."""

        # Index (variable)
        if isinstance(program, Index):
            if program.i >= len(env):
                return float('-inf')

            try:
                ctx.unify(env[program.i], target_type)
                # Count how many variables of this type
                n_same_type = sum(
                    1 for i, t in enumerate(env)
                    if self._types_compatible(t, target_type)
                )
                return self.log_variable - math.log(max(1, n_same_type))
            except UnificationError:
                return float('-inf')

        # Abstraction
        if isinstance(program, Abstraction):
            if not isinstance(target_type, Arrow):
                return float('-inf')

            new_env = [target_type.arg] + env
            return self._ll_helper(program.body, target_type.ret, ctx, new_env)

        # Primitive or Invented
        if isinstance(program, (Primitive, Invented)):
            prod = self.get_production(program)
            if prod is None:
                return float('-inf')

            try:
                inst_type = ctx.instantiate(prod.tp)
                # Unify with target type directly
                ctx.unify(inst_type, target_type)
                return prod.log_probability
            except UnificationError:
                return float('-inf')

        # Application
        if isinstance(program, Application):
            # Infer the function type
            arg_type = ctx.fresh_type_variable()
            func_type = Arrow(arg_type, target_type)

            ll_func = self._ll_helper(program.f, func_type, ctx, env)
            if ll_func == float('-inf'):
                return float('-inf')

            ll_arg = self._ll_helper(program.x, ctx.apply(arg_type), ctx, env)
            return ll_func + ll_arg

        return float('-inf')

    def _types_compatible(self, t1: Type, t2: Type) -> bool:
        """Check if two types can be unified."""
        try:
            ctx = TypeContext()
            ctx.unify(t1, t2)
            return True
        except UnificationError:
            return False

    def description_length(
        self,
        program: Program,
        request_type: Type
    ) -> float:
        """
        Compute the description length (bits) of a program.

        This is -log_2(P(program)) / log(2).
        """
        ll = self.program_log_likelihood(program, request_type)
        if ll == float('-inf'):
            return float('inf')
        return -ll / math.log(2)

    # ========================================================================
    # GRAMMAR UPDATES (for wake-sleep learning)
    # ========================================================================

    def with_production(self, production: Production) -> 'Grammar':
        """Return a new grammar with an added production."""
        new_productions = self.productions + [production]
        return Grammar(new_productions, self.log_variable, self.continuation_type)

    def with_invented(
        self,
        invented: Invented,
        log_probability: float = 0.0
    ) -> 'Grammar':
        """Add an invented abstraction to the grammar."""
        ctx = TypeContext()
        tp = invented.infer_type(ctx, [])
        prod = Production(invented, tp, log_probability)
        return self.with_production(prod)

    def normalize_probabilities(self) -> 'Grammar':
        """
        Normalize log-probabilities so they sum to 1 (in probability space).

        This ensures the grammar is a proper probability distribution.
        """
        if not self.productions:
            return self

        # Use log-sum-exp for numerical stability
        log_probs = [p.log_probability for p in self.productions] + [self.log_variable]
        max_lp = max(log_probs)
        log_sum = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

        new_productions = [
            Production(p.program, p.tp, p.log_probability - log_sum)
            for p in self.productions
        ]
        new_log_variable = self.log_variable - log_sum

        return Grammar(new_productions, new_log_variable, self.continuation_type)

    def inside_outside_update(
        self,
        frontiers: List[List[Tuple[Program, float]]],
        pseudo_counts: float = 0.1
    ) -> 'Grammar':
        """
        Update grammar weights using inside-outside algorithm on frontiers.

        Args:
            frontiers: List of frontiers, each is [(program, log_likelihood), ...]
            pseudo_counts: Laplace smoothing for unseen productions

        Returns:
            Updated grammar with new weights
        """
        # Count production uses across all programs
        counts: Dict[str, float] = defaultdict(lambda: pseudo_counts)
        variable_count = pseudo_counts

        for frontier in frontiers:
            for program, ll in frontier:
                self._count_uses(program, counts, variable_count)

        # Convert counts to log-probabilities
        total = sum(counts.values()) + variable_count
        new_productions = []
        for prod in self.productions:
            key = str(prod.program)
            count = counts.get(key, pseudo_counts)
            new_lp = math.log(count / total)
            new_productions.append(Production(prod.program, prod.tp, new_lp))

        new_log_variable = math.log(variable_count / total)

        return Grammar(new_productions, new_log_variable, self.continuation_type)

    def _count_uses(
        self,
        program: Program,
        counts: Dict[str, float],
        variable_count: float
    ) -> float:
        """Count production uses in a program (helper for inside-outside)."""
        if isinstance(program, Index):
            variable_count += 1
            return variable_count

        if isinstance(program, (Primitive, Invented)):
            key = str(program)
            counts[key] = counts.get(key, 0) + 1
            return variable_count

        if isinstance(program, Abstraction):
            return self._count_uses(program.body, counts, variable_count)

        if isinstance(program, Application):
            variable_count = self._count_uses(program.f, counts, variable_count)
            variable_count = self._count_uses(program.x, counts, variable_count)
            return variable_count

        return variable_count


# ============================================================================
# GRAMMAR CONSTRUCTION HELPERS
# ============================================================================

def make_grammar(
    primitives: List[Primitive],
    log_probability: float = 0.0,
    log_variable: float = -1.0
) -> Grammar:
    """
    Create a grammar from a list of primitives.

    All primitives start with equal log-probability.
    """
    productions = [
        Production(p, p.tp, log_probability)
        for p in primitives
    ]
    return Grammar(productions, log_variable)


def uniform_grammar(primitives: List[Primitive]) -> Grammar:
    """Create a grammar with uniform probabilities over primitives."""
    n = len(primitives) + 1  # +1 for variable
    log_p = -math.log(n)
    return make_grammar(primitives, log_probability=log_p, log_variable=log_p)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Grammar Tests ===\n")

    # Create simple primitives
    add = Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y)
    zero = Primitive('0', INT, 0)
    one = Primitive('1', INT, 1)
    succ = Primitive('succ', arrow(INT, INT), lambda x: x + 1)

    # Create grammar
    g = uniform_grammar([add, mul, zero, one, succ])
    print(g)

    # Test program likelihood
    # Program: (+ 1 0)
    p1 = Application(Application(add, one), zero)
    print(f"\nProgram: {p1}")
    ll = g.program_log_likelihood(p1, INT)
    print(f"Log-likelihood: {ll:.4f}")
    print(f"Description length: {g.description_length(p1, INT):.2f} bits")

    # Program: (λ (+ $0 1))  -- increment function
    p2 = Abstraction(Application(Application(add, Index(0)), one))
    print(f"\nProgram: {p2}")
    ll2 = g.program_log_likelihood(p2, arrow(INT, INT))
    print(f"Log-likelihood: {ll2:.4f}")
    print(f"Description length: {g.description_length(p2, arrow(INT, INT)):.2f} bits")

    # Test candidates
    print("\n=== Candidates for INT ===")
    ctx = TypeContext()
    candidates = g.candidates_for_type(INT, ctx, [])
    for prod, tp, lp in candidates:
        print(f"  {prod.program}: {tp} [log_p={lp:.2f}]")

    print("\n=== Candidates for INT -> INT ===")
    candidates = g.candidates_for_type(arrow(INT, INT), ctx, [])
    for prod, tp, lp in candidates:
        print(f"  {prod.program}: {tp} [log_p={lp:.2f}]")

    print("\n=== Grammar Tests OK ===")
