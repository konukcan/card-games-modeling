"""
=============================================================================
Probabilistic Context-Free Grammar (PCFG) for Program Synthesis
=============================================================================

This module implements DreamCoder's grammar representation:
- Each production has a log-probability
- Programs are scored by their description length (negative log-probability)
- The grammar can be updated via inside-outside algorithm or compression
- Type-indexed normalization: probabilities normalized per return type

The grammar generates programs top-down, making type-directed choices.

ARCHITECTURE COMPARISON:
------------------------
ORIGINAL DREAMCODER (OCaml):
    - Grammar is a record type with logVariable and productions list
    - Separate "Library" type for invented abstractions
    - Type-indexed normalization (probabilities normalized per-return-type)
    - Uses Dirichlet-Categorical conjugate prior (α = 1)

OUR IMPLEMENTATION (Python):
    - Grammar class with log_variable and productions list
    - Primitives and Invented abstractions unified in same productions list
    - Type-indexed normalization (matches original DreamCoder)
    - Inside-outside with Dirichlet prior (default α=1, configurable)

NOTE ON LIBRARY TYPE:
    Original DreamCoder separates "base primitives" from "invented abstractions"
    (the Library). We use a unified approach where both are Production objects
    containing either Primitive or Invented programs. This simplifies the code
    without losing functionality.

KEY CONCEPTS:
- Type-indexed normalization: When filling a hole of type T, only T-returning
  primitives compete. This matches the original DreamCoder.
- Dirichlet prior: Grammar updates use Bayesian smoothing with configurable
  α parameter (default α=1, Laplace smoothing).

See docs/on_dirichlet_priors.md for detailed explanation of the Dirichlet prior.
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


# =============================================================================
# PRODUCTION CLASS
# =============================================================================

@dataclass
class Production:
    """
    A production rule in the probabilistic grammar.

    FIELDS:
    -------
    program: Union[Primitive, Invented]
        The actual primitive or learned abstraction.
        - Primitive: Built-in function like '+', 'map', 'filter'
        - Invented: Learned abstraction like '#(λ all_same_color $0)'

    tp: Type
        The polymorphic type of this production.
        Examples:
            INT                          -- A constant like '0' or '1'
            Arrow(INT, INT)              -- A unary function like 'succ'
            Arrow(INT, Arrow(INT, INT))  -- Binary function like '+'
            Arrow(TypeVariable('a'), TypeVariable('a'))  -- Polymorphic identity

    log_probability: float
        The natural log of the probability: log(P(choose this production))
        - Negative values (e.g., -2.3) mean probability < 1
        - More negative = less likely
        - log(0.1) ≈ -2.3, log(0.01) ≈ -4.6
    """
    program: Union[Primitive, Invented]
    tp: Type
    log_probability: float

    def __str__(self) -> str:
        return f"{self.program} : {self.tp} [log_p={self.log_probability:.2f}]"


# =============================================================================
# GRAMMAR CLASS
# =============================================================================

class Grammar:
    """
    A Probabilistic Context-Free Grammar (PCFG) for program synthesis.

    WHAT IS A PCFG?
    ---------------
    A PCFG assigns probabilities to derivations. In our case:
    - Non-terminals are TYPES (e.g., INT, BOOL, HAND → BOOL)
    - Productions are PRIMITIVES that produce each type
    - A program's probability is the product of all production probabilities used

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
            log_variable: Log-probability of using a BOUND VARIABLE (de Bruijn index).
                          Default -1.0 means P(use variable) ≈ 0.37 (= e^-1).
                          Variables aren't primitives - they're references to
                          lambda-bound values. In (λ $0), the $0 refers to the
                          lambda's argument.
            continuation_type: For imperative programs (not used here)
        """
        self.productions = list(productions)
        self.log_variable = log_variable
        self.continuation_type = continuation_type

        # Build index by primitive/invented for fast lookup
        # Key: string representation of primitive (e.g., "+", "map", "#invented_0")
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

    # =========================================================================
    # CANDIDATE LOOKUP - Which primitives can fill a hole of type T?
    # =========================================================================

    def candidates_for_type(
        self,
        target_type: Type,
        ctx: TypeContext,
        env: List[Type],
        normalize: bool = True
    ) -> List[Tuple[Production, Type, float]]:
        """
        Find all productions that could produce the target type.

        Given a "hole" of type T (e.g., ?:INT or ?:HAND→BOOL), find all
        primitives whose return type unifies with T.

        Args:
            target_type: The type we need to produce
            ctx: Type context for fresh variables
            env: Type environment (types of bound variables)
            normalize: If True, apply type-indexed normalization so that
                       probabilities of returned candidates sum to 1.
                       This matches original DreamCoder behavior.

        Returns:
            List of (production, instantiated_type, log_probability).
            If normalize=True, log_probabilities are adjusted so exp(sum) = 1.

        TYPE-INDEXED NORMALIZATION:
            When filling a hole of type T, only T-returning primitives compete.
            Their probabilities are normalized within this subset.
            Primitives returning other types don't affect the distribution.
        """
        candidates = []

        for prod in self.productions:
            # Try to unify the production's return type with target
            try:
                # Create fresh type variables for this production
                # This avoids type variable capture between productions
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

        # Type-indexed normalization: normalize probabilities within candidates
        if normalize and candidates:
            candidates = self._normalize_candidates(candidates)

        return candidates

    def _normalize_candidates(
        self,
        candidates: List[Tuple[Production, Type, float]]
    ) -> List[Tuple[Production, Type, float]]:
        """
        Normalize log-probabilities of candidates so they sum to 1.

        This implements type-indexed normalization: when choosing among
        primitives that can produce type T, we normalize within that set.

        Uses log-sum-exp for numerical stability:
            log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ - max(x)))
        """
        if not candidates:
            return candidates

        # Extract log-probs
        log_probs = [lp for _, _, lp in candidates]

        # Log-sum-exp: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
        max_lp = max(log_probs)
        log_sum = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

        # Normalize: new_lp = old_lp - log_sum
        return [
            (prod, tp, lp - log_sum)
            for prod, tp, lp in candidates
        ]

    def variable_candidates(
        self,
        target_type: Type,
        ctx: TypeContext,
        env: List[Type]
    ) -> List[Tuple[int, float]]:
        """
        Find all bound variables that match the target type.

        In (λ ... $0 ...), $0 refers to the lambda's argument.
        The 'env' tracks types: env[i] = type of $i.

        Args:
            target_type: The type we need
            ctx: For type inference
            env: Types of bound variables. env[i] = type of $i.

        Returns:
            List of (de_bruijn_index, log_probability).
            All variables get the same log_probability (self.log_variable).
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

    # =========================================================================
    # PROGRAM LIKELIHOOD - P(program | grammar)
    # =========================================================================

    def program_log_likelihood(
        self,
        program: Program,
        request_type: Type,
        env: List[Type] = None
    ) -> float:
        """
        Compute log P(program | grammar, request_type).

        This is the DESCRIPTION LENGTH in nats (natural log units).
        To convert to bits: bits = -log_likelihood / log(2)

        The probability that this grammar would generate exactly this program
        when asked to produce the request_type.

        Formula: P(program) = ∏ P(each choice made during generation)
                 log P(program) = Σ log P(each choice)
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
        """
        Recursive helper for computing log-likelihood.

        CASES BY PROGRAM TYPE:
        ----------------------
        1. INDEX (variable): Check type matches, return log_variable - log(#same-type-vars)
        2. ABSTRACTION (lambda): Target must be Arrow, recurse on body
        3. PRIMITIVE/INVENTED: Look up in grammar, return log_probability
        4. APPLICATION: Sum log-likelihoods of function and argument
        """

        # Index (variable)
        if isinstance(program, Index):
            if program.i >= len(env):
                return float('-inf')

            try:
                ctx.unify(env[program.i], target_type)
                # Count how many variables of this type
                # We divide probability among them
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

            # Extend environment: new arg becomes $0
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

        DL(program) = -log₂ P(program) = -log P(program) / log(2)

        Lower DL = simpler/more natural program given this grammar.

        MDL PRINCIPLE:
            Minimum Description Length says:
            Best grammar = minimizes DL(grammar) + Σ DL(program_i | grammar)
            This is Occam's Razor formalized.
        """
        ll = self.program_log_likelihood(program, request_type)
        if ll == float('-inf'):
            return float('inf')
        return -ll / math.log(2)

    # =========================================================================
    # GRAMMAR DESCRIPTION LENGTH (for MDL scoring)
    # =========================================================================

    def grammar_description_length(self) -> float:
        """
        Compute the description length of the grammar itself.

        This is crucial for MDL-based compression. The total MDL is:
            MDL = DL(grammar) + Σ DL(program_i | grammar)

        A grammar with many complex abstractions has high DL, which
        must be offset by savings in program description lengths.

        FORMULA:
            DL(grammar) = Σ DL(production_i)

        Each production's DL includes:
            1. Type complexity (number of type constructors)
            2. Body complexity (for Invented: size of body in AST nodes)
            3. Primitives have fixed body cost of 1

        Returns:
            Description length in bits (using log base 2)

        WHY THIS MATTERS:
            Without grammar DL penalty, compression would add every
            possible abstraction. The grammar DL penalizes complexity,
            ensuring we only keep abstractions that earn their keep.

        COMPARISON TO ORIGINAL DREAMCODER:
            Original uses similar scoring with:
            - Type encoding using rational numbers
            - Body encoding using grammar probabilities
            Our simplified version uses AST size as proxy.
        """
        total_dl = 0.0

        for prod in self.productions:
            # Type complexity: count type constructors
            type_dl = self._type_description_length(prod.tp)

            # Body complexity
            if isinstance(prod.program, Invented):
                # Invented: encode the body structure
                # Use AST size as proxy for encoding cost
                body_dl = prod.program.body.size()
            else:
                # Primitives: fixed cost (they're atomic symbols)
                body_dl = 1.0

            # Each production also costs 1 bit to "announce"
            production_cost = 1.0

            total_dl += type_dl + body_dl + production_cost

        return total_dl

    def _type_description_length(self, tp: Type) -> float:
        """
        Compute description length of a type.

        Counts type constructors as proxy for encoding complexity.
        More complex types (deeply nested arrows, lists) cost more.

        TYPE ENCODING COSTS:
            BaseType (INT, BOOL, etc.):  1 bit
            Arrow (→):                    1 bit + DL(arg) + DL(ret)
            ListType ([]):                1 bit + DL(elem)
            TypeVariable (α, β):          0.5 bits (free/polymorphic)

        The 0.5 cost for type variables reflects that polymorphic
        primitives are "simpler" than monomorphic equivalents.
        """
        if isinstance(tp, BaseType):
            return 1.0

        if isinstance(tp, Arrow):
            # Arrow costs 1 to encode the constructor, plus child types
            return 1.0 + self._type_description_length(tp.arg) + \
                        self._type_description_length(tp.ret)

        if isinstance(tp, ListType):
            return 1.0 + self._type_description_length(tp.elem)

        if isinstance(tp, TypeVariable):
            # Type variables are "free" (polymorphism is cheap)
            return 0.5

        # Unknown type: conservative estimate
        return 1.0

    def invented_count(self) -> int:
        """Count the number of invented abstractions in the grammar."""
        return sum(1 for p in self.productions if isinstance(p.program, Invented))

    def primitive_count(self) -> int:
        """Count the number of base primitives in the grammar."""
        return sum(1 for p in self.productions if isinstance(p.program, Primitive))

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

        Uses log-sum-exp for numerical stability:
            log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ - max(x)))
        """
        if not self.productions:
            return self

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
        alpha: float = 1.0
    ) -> 'Grammar':
        """
        Update grammar weights using inside-outside algorithm on frontiers.

        This implements Bayesian updating with a symmetric Dirichlet prior.

        Args:
            frontiers: List of frontiers, each is [(program, log_likelihood), ...]
            alpha: Dirichlet concentration parameter (default 1.0 = Laplace smoothing)
                   - α < 1: Sparse prior, unused primitives get very low probability
                   - α = 1: Laplace smoothing (uniform Dirichlet), balanced
                   - α > 1: Dense prior, probabilities spread across all primitives

                   Original DreamCoder uses α = 1.

        Returns:
            Updated grammar with new weights

        BAYESIAN INTERPRETATION:
            Prior:     P(θ) = Dirichlet(α, α, ..., α)
            Posterior: P(θ|data) = Dirichlet(α + count₁, α + count₂, ...)
            Estimate:  P(prim_i) = (count_i + α) / (total + K*α)

        EFFECT OF α ON LEARNING DYNAMICS:
            α = 0.1 (Sparse): Unused primitives → probability ≈ 0.01
            α = 1.0 (Laplace): Unused primitives → probability ≈ 0.05
            α = 10 (Dense): Unused primitives → probability ≈ 0.10

        See docs/on_dirichlet_priors.md for detailed explanation.
        """
        # Count production uses across all programs
        # Initialize with alpha (Dirichlet pseudo-counts)
        counts: Dict[str, float] = defaultdict(lambda: alpha)
        variable_count = alpha

        for frontier in frontiers:
            for program, ll in frontier:
                variable_count = self._count_uses(program, counts, variable_count)

        # Convert counts to log-probabilities
        # P(prim_i) = (count_i + α) / (total + K*α)
        # But since we initialized with α, counts already include it
        total = sum(counts.values()) + variable_count
        new_productions = []
        for prod in self.productions:
            key = str(prod.program)
            count = counts.get(key, alpha)
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
    """
    Create a grammar with uniform probabilities over primitives.

    Each primitive (and variable) has probability 1/(n+1).
    """
    n = len(primitives) + 1  # +1 for variable
    log_p = -math.log(n)     # log(1/n) = -log(n)
    return make_grammar(primitives, log_probability=log_p, log_variable=log_p)


# ============================================================================
# COMPARISON TO ORIGINAL DREAMCODER
# ============================================================================
"""
+------------------------+---------------------------+---------------------------+
| Component              | Original DreamCoder       | Our Implementation        |
+------------------------+---------------------------+---------------------------+
| Language               | OCaml/Haskell             | Python                    |
+------------------------+---------------------------+---------------------------+
| Production storage     | (program, type, log_p)    | Production dataclass      |
|                        | tuple in list             | in list                   |
+------------------------+---------------------------+---------------------------+
| Library vs Grammar     | Separate types            | Unified (both Production) |
|                        | (primitives vs invented)  | with Primitive/Invented   |
+------------------------+---------------------------+---------------------------+
| Type normalization     | Per-return-type           | Per-return-type ✓         |
|                        | (each type sums to 1)     | (matches original)        |
+------------------------+---------------------------+---------------------------+
| Dirichlet prior        | α = 1 (Laplace)           | α = 1 (Laplace) ✓         |
|                        | in full Bayesian update   | configurable via alpha    |
+------------------------+---------------------------+---------------------------+
| Inside-outside         | Full EM with type-aware   | Simplified counting       |
|                        | expectations              |                           |
+------------------------+---------------------------+---------------------------+
| Polymorphism           | Careful instantiation     | Basic instantiation       |
|                        | tracking                  |                           |
+------------------------+---------------------------+---------------------------+

KEY INSIGHT:
    The grammar is both a GENERATOR (defines what programs exist) and a
    SCORER (assigns probability/description-length to programs).

    During synthesis: Use grammar to prioritize which programs to explore.
    After synthesis: Update grammar to make successful patterns more likely.

    This is the WAKE-SLEEP loop:
        WAKE:  Use grammar to find programs
        SLEEP: Update grammar based on what worked
"""


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
