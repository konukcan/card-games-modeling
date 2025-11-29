# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cythonized Probabilistic Context-Free Grammar (PCFG) for Program Synthesis

Optimized version of grammar.py with:
- cdef classes for speed
- Typed attributes and local variables
- Disabled bounds checking for performance

The grammar generates programs top-down, making type-directed choices.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
from collections import defaultdict

# Import from Cython modules
from .type_system_cy import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow,
    canonical_type, type_arity
)
from .program_cy import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args, multi_lambda
)


# ============================================================================
# PRODUCTION (cdef class for performance)
# ============================================================================

cdef class Production:
    """
    A production in the grammar.

    Each production associates a primitive/invented with:
    - Its type
    - A log-probability weight
    """
    cdef readonly object program  # Primitive or Invented
    cdef readonly object tp       # Type
    cdef readonly double log_probability

    def __init__(self, object program, object tp, double log_probability):
        self.program = program
        self.tp = tp
        self.log_probability = log_probability

    def __str__(self) -> str:
        return f"{self.program} : {self.tp} [log_p={self.log_probability:.2f}]"


# ============================================================================
# GRAMMAR (cdef class for performance)
# ============================================================================

cdef class Grammar:
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
    cdef readonly list productions
    cdef readonly double log_variable
    cdef readonly object continuation_type
    cdef dict _by_name

    def __init__(
        self,
        list productions,
        double log_variable = -1.0,
        object continuation_type = None
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
        self._by_name = {}
        cdef Production p
        cdef str key
        for p in self.productions:
            key = str(p.program)
            self._by_name[key] = p

    def __str__(self) -> str:
        cdef list lines = ["Grammar:"]
        lines.append(f"  log_variable = {self.log_variable:.2f}")
        lines.append(f"  {len(self.productions)} productions:")
        cdef Production p
        for p in sorted(self.productions, key=lambda p: -p.log_probability):
            lines.append(f"    {p}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.productions)

    cpdef list primitives(self):
        """Return all primitives/invented in the grammar."""
        cdef list result = []
        cdef Production p
        for p in self.productions:
            result.append(p.program)
        return result

    cpdef Production get_production(self, object program):
        """Look up a production by its program."""
        cdef str key = str(program)
        return self._by_name.get(key)

    cpdef double get_log_probability(self, object program):
        """Get the log-probability of a production."""
        cdef Production prod = self.get_production(program)
        if prod is None:
            return float('-inf')
        return prod.log_probability

    cpdef list candidates_for_type(self, object target_type, object ctx, list env):
        """
        Find all productions that could produce the target type.

        Returns list of (production, instantiated_type, log_probability).
        """
        cdef list candidates = []
        cdef Production prod
        cdef object fresh_ctx
        cdef object inst_type
        cdef object ret_type
        cdef object final_type

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

    cpdef list variable_candidates(self, object target_type, object ctx, list env):
        """
        Find all bound variables that match the target type.

        Returns list of (de_bruijn_index, log_probability).
        """
        cdef list candidates = []
        cdef int i
        cdef object var_type
        cdef object fresh_ctx
        cdef object inst_var_type

        for i, var_type in enumerate(env):
            try:
                fresh_ctx = TypeContext()
                inst_var_type = fresh_ctx.instantiate(var_type)
                fresh_ctx.unify(inst_var_type, target_type)
                candidates.append((i, self.log_variable))
            except UnificationError:
                continue

        return candidates

    cpdef double program_log_likelihood(self, object program, object request_type, list env=None):
        """
        Compute log P(program | grammar, request_type).

        This is the description length (lower = shorter = better).
        """
        if env is None:
            env = []

        cdef object ctx = TypeContext()
        return self._ll_helper(program, request_type, ctx, env)

    cdef double _ll_helper(self, object program, object target_type, object ctx, list env):
        """Recursive helper for computing log-likelihood."""
        cdef double ll_func, ll_arg
        cdef int n_same_type, i
        cdef object t
        cdef list new_env
        cdef Production prod
        cdef object inst_type
        cdef object arg_type
        cdef object func_type

        # Index (variable)
        if isinstance(program, Index):
            if program.i >= len(env):
                return float('-inf')

            try:
                ctx.unify(env[program.i], target_type)
                # Count how many variables of this type
                n_same_type = 0
                for i, t in enumerate(env):
                    if self._types_compatible(t, target_type):
                        n_same_type += 1
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
        if isinstance(program, Primitive) or isinstance(program, Invented):
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

    cdef bint _types_compatible(self, object t1, object t2):
        """Check if two types can be unified."""
        try:
            ctx = TypeContext()
            ctx.unify(t1, t2)
            return True
        except UnificationError:
            return False

    cpdef double description_length(self, object program, object request_type):
        """
        Compute the description length (bits) of a program.

        This is -log_2(P(program)) / log(2).
        """
        cdef double ll = self.program_log_likelihood(program, request_type)
        if ll == float('-inf'):
            return float('inf')
        return -ll / math.log(2)

    # ========================================================================
    # GRAMMAR UPDATES (for wake-sleep learning)
    # ========================================================================

    cpdef Grammar with_production(self, Production production):
        """Return a new grammar with an added production."""
        cdef list new_productions = self.productions + [production]
        return Grammar(new_productions, self.log_variable, self.continuation_type)

    cpdef Grammar with_invented(self, object invented, double log_probability=0.0):
        """Add an invented abstraction to the grammar."""
        cdef object ctx = TypeContext()
        cdef object tp = invented.infer_type(ctx, [])
        cdef Production prod = Production(invented, tp, log_probability)
        return self.with_production(prod)

    cpdef Grammar normalize_probabilities(self):
        """
        Normalize log-probabilities so they sum to 1 (in probability space).

        This ensures the grammar is a proper probability distribution.
        """
        if not self.productions:
            return self

        # Use log-sum-exp for numerical stability
        cdef list log_probs = []
        cdef Production p
        for p in self.productions:
            log_probs.append(p.log_probability)
        log_probs.append(self.log_variable)

        cdef double max_lp = max(log_probs)
        # Compute sum without generator expression (not allowed in cpdef)
        cdef double exp_sum = 0.0
        cdef double lp
        for lp in log_probs:
            exp_sum += math.exp(lp - max_lp)
        cdef double log_sum = max_lp + math.log(exp_sum)

        cdef list new_productions = []
        for p in self.productions:
            new_productions.append(Production(p.program, p.tp, p.log_probability - log_sum))

        cdef double new_log_variable = self.log_variable - log_sum

        return Grammar(new_productions, new_log_variable, self.continuation_type)

    cpdef Grammar inside_outside_update(self, list frontiers, double pseudo_counts=0.1):
        """
        Update grammar weights using inside-outside algorithm on frontiers.

        Args:
            frontiers: List of frontiers, each is [(program, log_likelihood), ...]
            pseudo_counts: Laplace smoothing for unseen productions

        Returns:
            Updated grammar with new weights
        """
        # Count production uses across all programs
        # Use a regular dict instead of defaultdict to avoid lambda
        cdef dict counts = {}
        cdef double variable_count = pseudo_counts
        cdef list frontier
        cdef object program
        cdef double ll

        for frontier in frontiers:
            for program, ll in frontier:
                variable_count = self._count_uses(program, counts, variable_count)

        # Convert counts to log-probabilities
        cdef double total_counts = 0.0
        cdef double v
        for v in counts.values():
            total_counts += v
        cdef double total = total_counts + variable_count
        cdef list new_productions = []
        cdef Production prod
        cdef str key
        cdef double count
        cdef double new_lp

        for prod in self.productions:
            key = str(prod.program)
            count = counts.get(key, pseudo_counts)
            new_lp = math.log(count / total)
            new_productions.append(Production(prod.program, prod.tp, new_lp))

        cdef double new_log_variable = math.log(variable_count / total)

        return Grammar(new_productions, new_log_variable, self.continuation_type)

    cdef double _count_uses(self, object program, dict counts, double variable_count):
        """Count production uses in a program (helper for inside-outside)."""
        cdef str key

        if isinstance(program, Index):
            variable_count += 1
            return variable_count

        if isinstance(program, Primitive) or isinstance(program, Invented):
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

def make_grammar(list primitives, double log_probability=0.0, double log_variable=-1.0):
    """
    Create a grammar from a list of primitives.

    All primitives start with equal log-probability.
    """
    cdef list productions = []
    cdef object p
    for p in primitives:
        productions.append(Production(p, p.tp, log_probability))
    return Grammar(productions, log_variable)


def uniform_grammar(list primitives):
    """Create a grammar with uniform probabilities over primitives."""
    cdef int n = len(primitives) + 1  # +1 for variable
    cdef double log_p = -math.log(n)
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

    print("\n=== Grammar Tests OK ===")
