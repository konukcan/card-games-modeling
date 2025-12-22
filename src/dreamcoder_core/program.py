"""
Lambda Calculus Program Representation
======================================

PURPOSE:
    This module represents programs as Abstract Syntax Trees (ASTs) in the
    lambda calculus. Programs are the synthesized solutions to tasks.

COMPARISON WITH ORIGINAL DREAMCODER (Ellis et al.):
    - Original has 8 AST node types; we have 6 (Primitive, Index, Application,
      Abstraction, Invented, Hole)
    - Original uses global Primitive.GLOBALS registry; we pass primitives explicitly
    - Original has multiple visitor classes; we use walk() and ProgramTransformer
    - Both use de Bruijn indices for variable representation

KEY CONCEPT - DE BRUIJN INDICES:
    Instead of named variables (λx. λy. x + y), we use numeric indices:
        λ. λ. $1 + $0

    $0 = the innermost bound variable (most recent λ)
    $1 = one λ level up
    $2 = two λ levels up
    etc.

    WHY? Named variables have "alpha equivalence" problem:
        λx. x  and  λy. y  are the same function but look different.
    De Bruijn indices make structurally identical programs look identical.

    EXAMPLE TRANSLATIONS:
        λx. x           →  λ. $0           (identity)
        λx. λy. x       →  λ. λ. $1        (first of two args)
        λx. λy. y       →  λ. λ. $0        (second of two args)
        λx. λy. x + y   →  λ. λ. $1 + $0   (add two args)
        λf. λx. f x     →  λ. λ. $1 $0     (apply f to x)

ARCHITECTURE:

    Program (Abstract Base)
    ├── Primitive(name, tp, value)     # Built-in operations: +, suit, rank, all, any...
    ├── Index(i)                       # Variable reference: $0, $1, $2...
    ├── Application(f, x)              # Function application: (f x)
    ├── Abstraction(body)              # Lambda: λ. body
    ├── Invented(body, name)           # Learned abstraction: #name
    └── Hole(tp, id)                   # Placeholder for top-down synthesis: ?id:tp

    EVALUATION MODEL:
    Programs evaluate in an environment (list of values).
    env[i] = value of $i
    When entering a λ, the argument becomes the new $0.

TOP-DOWN SYNTHESIS SUPPORT:
    The Hole node enables top-down enumeration:
    1. Start with ?:request_type
    2. Replace holes with grammar productions
    3. Continue until no holes remain

    Holes are typed placeholders that track what needs to be filled.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Counter
from abc import ABC, abstractmethod
from collections import Counter as CounterClass
import math

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)


# ============================================================================
# PROGRAM BASE CLASS
# ============================================================================

class Program(ABC):
    """
    Abstract base class for all program expressions.

    Every program node must implement:
    - __str__: Human-readable representation
    - __eq__, __hash__: For deduplication and caching
    - evaluate(env): Execute the program
    - infer_type(ctx, env): Type inference
    - size(), depth(): Complexity metrics
    - free_indices(): Which de Bruijn indices are unbound?
    - shift(), substitute(): For beta reduction
    - walk(): Tree traversal

    PYTHON SYNTAX NOTES:
    - ABC = Abstract Base Class
    - @abstractmethod = subclasses MUST implement
    - List[Any] = list containing any type
    - -> Type means return type annotation
    """

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable string representation."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Structural equality check."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        pass

    @abstractmethod
    def evaluate(self, env: List[Any]) -> Any:
        """
        Execute the program in an environment.

        The environment is a list where env[i] is the value of $i.
        When we enter a lambda and apply it to an argument,
        the argument becomes env[0] and everything else shifts.

        EXAMPLE:
            Program: (λ. $0 + 1)
            evaluate([])  →  returns a closure (function)
            When closure is called with arg=5:
                evaluate([5])  →  $0=5, so $0+1 = 6

        RAISES:
            RuntimeError if program contains Holes (incomplete programs
            cannot be evaluated)
        """
        pass

    @abstractmethod
    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        """
        Infer the type of this program.

        Args:
            ctx: Type context for fresh variables and unification
            env: Type environment, env[i] is the type of $i

        Returns:
            The inferred type of this expression
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Return the AST size (number of nodes).

        Used for description length / complexity metrics.
        Smaller programs are preferred (Occam's razor).
        """
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return the AST depth (longest path from root to leaf)."""
        pass

    @abstractmethod
    def free_indices(self, depth: int = 0) -> Set[int]:
        """
        Return free (unbound) de Bruijn indices.

        A free index is one not bound by any enclosing λ.

        The depth parameter tracks how many λs we've descended into.
        An index i is free at depth d if i >= d.

        EXAMPLE:
            In (λ. $0 + $1):
            - $0 at depth=1 is NOT free (bound by the λ)
            - $1 at depth=1 IS free (refers to outer context)
            free_indices() returns {0} (the adjusted free index)
        """
        pass

    @abstractmethod
    def shift(self, amount: int, cutoff: int = 0) -> 'Program':
        """
        Shift de Bruijn indices >= cutoff by amount.

        This is crucial for correct substitution during beta reduction.

        WHEN NEEDED:
        When substituting a term into a lambda body, indices in the
        substituted term must be adjusted so they still refer to the
        same variables.

        EXAMPLE:
            shift(1, 0) on $0  →  $1  (increase by 1)
            shift(1, 1) on $0  →  $0  (below cutoff, unchanged)
            shift(-1, 0) on $1 →  $0  (decrease by 1)
        """
        pass

    @abstractmethod
    def substitute(self, index: int, replacement: 'Program') -> 'Program':
        """
        Substitute a program for a de Bruijn index.

        This is the core of beta reduction:
        (λ. body) arg  →  body with $0 replaced by arg

        EXAMPLE:
            ($0 + $1).substitute(0, five) = (five + $1)
        """
        pass

    def is_closed(self) -> bool:
        """Check if this program has no free variables."""
        return len(self.free_indices()) == 0

    def beta_reduce(self) -> 'Program':
        """
        Perform one step of beta reduction if possible.

        Beta reduction: (λ. body) arg  →  body[0 := arg]

        Returns self if no reduction is possible.
        Override in Application to perform actual reduction.
        """
        return self

    def normalize(self) -> 'Program':
        """
        Fully normalize (beta-reduce until fixed point).

        Reduces repeatedly until no more reductions are possible.
        Has a safety limit of 1000 steps to prevent infinite loops.
        """
        current = self
        for _ in range(1000):  # Prevent infinite loops
            reduced = current.beta_reduce()
            if reduced == current:
                return current
            current = reduced
        return current

    @abstractmethod
    def walk(self, f: Callable[['Program'], None]) -> None:
        """
        Walk the AST, calling f on each node.

        Traverses all subexpressions in the tree.
        Used for collecting subprograms, counting primitives, etc.
        """
        pass

    def subprograms(self) -> List['Program']:
        """Return all subprograms (including self)."""
        result = []
        self.walk(lambda p: result.append(p))
        return result

    @abstractmethod
    def clone(self) -> 'Program':
        """Deep copy the program."""
        pass

    def is_complete(self) -> bool:
        """
        Check if program has no holes (is complete).

        A complete program can be evaluated.
        An incomplete program (with holes) is a partial program
        used during top-down synthesis.
        """
        return not has_holes(self)


# ============================================================================
# PRIMITIVE
# ============================================================================

@dataclass(frozen=True)
class Primitive(Program):
    """
    A built-in primitive operation.

    Primitives are the atomic building blocks of our DSL:
    - Arithmetic: +, -, *, /, >, <, ==
    - Card accessors: suit, rank, value
    - List operations: map, filter, all, any, length
    - Constants: true, false, 0, 1, clubs, hearts, ...

    FIELDS:
        name: Identifier string (e.g., 'suit', 'all', '+')
        tp: The type of the primitive (e.g., card -> suit)
        value: The Python implementation (function or constant)

    TYPE EXAMPLE:
        map : ('a -> 'b) -> list('a) -> list('b)
        This is a polymorphic type - 'a and 'b are type variables.

    VALUE EXAMPLE:
        Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)

        Note the curried form: + takes one int, returns a function
        that takes another int and returns their sum.

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Original: Global Primitive.GLOBALS[name] registry
    - Ours: Primitives passed explicitly (no global state)
    """
    name: str       # Identifier (e.g., 'suit', 'rank', 'all')
    tp: Type        # Type signature
    value: Any      # Python implementation

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        # Two primitives are equal if they have the same name
        return isinstance(other, Primitive) and self.name == other.name

    def __hash__(self) -> int:
        return hash(('primitive', self.name))

    def evaluate(self, env: List[Any]) -> Any:
        # Primitives evaluate to their implementation
        # The environment is ignored (primitives don't reference variables)
        return self.value

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        # Instantiate the type (fresh type variables for polymorphism)
        # This ensures each use of a polymorphic primitive gets fresh vars
        # Example: using 'map' twice gives different 'a, 'b each time
        return ctx.instantiate(self.tp)

    def size(self) -> int:
        return 1  # A primitive is one node

    def depth(self) -> int:
        return 1

    def free_indices(self, depth: int = 0) -> Set[int]:
        return set()  # Primitives have no variables

    def shift(self, amount: int, cutoff: int = 0) -> 'Program':
        return self  # Nothing to shift

    def substitute(self, index: int, replacement: Program) -> Program:
        return self  # Nothing to substitute

    def walk(self, f: Callable[[Program], None]) -> None:
        f(self)  # Visit self

    def clone(self) -> Program:
        return self  # Primitives are immutable, no need to copy


# ============================================================================
# INDEX (de Bruijn variable)
# ============================================================================

@dataclass(frozen=True)
class Index(Program):
    """
    A de Bruijn index variable.

    De Bruijn indices refer to lambda-bound variables by their
    distance from the binding site:

        $0 = innermost (most recently bound)
        $1 = one lambda out
        $2 = two lambdas out
        etc.

    EXAMPLE:
        λ. λ. $1 + $0
             │    │
             │    └── $0 refers to inner λ's variable
             └─────── $1 refers to outer λ's variable

        In named form: λx. λy. x + y

    WHY DE BRUIJN INDICES?
        Named variables have alpha-equivalence issues:
            λx. x  and  λy. y  are the same function.
        De Bruijn indices make identical programs look identical:
            λ. $0  and  λ. $0  are obviously the same.

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Same concept and implementation
    - Both use Index class with integer field
    """
    i: int  # The index value

    def __str__(self) -> str:
        return f"${self.i}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Index) and self.i == other.i

    def __hash__(self) -> int:
        return hash(('index', self.i))

    def evaluate(self, env: List[Any]) -> Any:
        """
        Look up the value in the environment.

        env[0] = value of $0
        env[1] = value of $1
        etc.
        """
        if self.i >= len(env):
            raise RuntimeError(f"Index ${self.i} not in environment of size {len(env)}")
        return env[self.i]

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        """Look up the type in the type environment."""
        if self.i >= len(env):
            raise UnificationError(f"Index ${self.i} not in type environment")
        return env[self.i]

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def free_indices(self, depth: int = 0) -> Set[int]:
        """
        Return free indices, adjusted for current depth.

        An index i is free at depth d if i >= d (not bound by any enclosing λ).
        The returned index is adjusted: i - d (relative to outer context).

        EXAMPLE:
            In (λ. $0 + $1):
            - At depth 1: $0 is NOT free (0 < 1)
            - At depth 1: $1 IS free (1 >= 1), returned as {0} (1-1=0)
        """
        if self.i >= depth:
            return {self.i - depth}  # Adjusted to outer context
        return set()  # Bound by enclosing lambda

    def shift(self, amount: int, cutoff: int = 0) -> Program:
        """
        Shift index if >= cutoff.

        Used during substitution to adjust indices.
        """
        if self.i >= cutoff:
            return Index(self.i + amount)
        return self

    def substitute(self, index: int, replacement: Program) -> Program:
        """Replace this index with replacement if it matches."""
        if self.i == index:
            return replacement
        return self

    def walk(self, f: Callable[[Program], None]) -> None:
        f(self)

    def clone(self) -> Program:
        return self  # Immutable


# ============================================================================
# APPLICATION
# ============================================================================

@dataclass(frozen=True)
class Application(Program):
    """
    Function application: (f x)

    Represents applying function f to argument x.

    Applications are LEFT-ASSOCIATIVE (curried):
        (f x y z) = (((f x) y) z)

    This matches the curried function types:
        f : A -> B -> C -> D
        f x : B -> C -> D      (partial application)
        f x y : C -> D
        f x y z : D

    EXAMPLE:
        (+ 3 4)  means  ((+ 3) 4)
        First + is applied to 3, giving "add 3" function.
        Then "add 3" is applied to 4, giving 7.

    BETA REDUCTION:
        (λ. body) arg  →  body[0 := arg]
        Apply lambda to argument by substituting arg for $0 in body.

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Same structure
    - Both have f and x (or f and arg) fields
    """
    f: Program  # The function
    x: Program  # The argument

    def __str__(self) -> str:
        f_str = str(self.f)
        x_str = str(self.x)

        # Parenthesize arguments that are applications (for readability)
        if isinstance(self.x, Application):
            x_str = f"({x_str})"

        # Parenthesize lambdas
        if isinstance(self.f, Abstraction):
            f_str = f"({f_str})"
        if isinstance(self.x, Abstraction):
            x_str = f"({x_str})"

        return f"{f_str} {x_str}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Application) and self.f == other.f and self.x == other.x

    def __hash__(self) -> int:
        return hash(('app', self.f, self.x))

    def evaluate(self, env: List[Any]) -> Any:
        """
        Evaluate by applying function to argument.

        1. Evaluate f to get a function value
        2. Evaluate x to get an argument value
        3. Call the function with the argument
        """
        func = self.f.evaluate(env)
        arg = self.x.evaluate(env)

        if callable(func):
            return func(arg)
        else:
            raise RuntimeError(f"Cannot apply non-function: {func}")

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        """
        Type inference for application.

        If f : A -> B and x : A, then (f x) : B

        We use unification:
        1. Infer type of f
        2. Infer type of x
        3. f's type must unify with (x's type -> fresh variable)
        4. Return the fresh variable (now unified to actual return type)
        """
        f_type = self.f.infer_type(ctx, env)
        x_type = self.x.infer_type(ctx, env)

        # f must be an arrow type: x_type -> ret_type
        ret_type = ctx.fresh_type_variable()
        expected_f_type = Arrow(x_type, ret_type)

        ctx.unify(f_type, expected_f_type)
        return ctx.apply(ret_type)

    def size(self) -> int:
        return 1 + self.f.size() + self.x.size()

    def depth(self) -> int:
        return 1 + max(self.f.depth(), self.x.depth())

    def free_indices(self, depth: int = 0) -> Set[int]:
        return self.f.free_indices(depth) | self.x.free_indices(depth)

    def shift(self, amount: int, cutoff: int = 0) -> Program:
        return Application(
            self.f.shift(amount, cutoff),
            self.x.shift(amount, cutoff)
        )

    def substitute(self, index: int, replacement: Program) -> Program:
        return Application(
            self.f.substitute(index, replacement),
            self.x.substitute(index, replacement)
        )

    def beta_reduce(self) -> Program:
        """
        Perform beta reduction: (λ. body) arg → body[0 := arg]

        This is the heart of lambda calculus computation.

        STEPS:
        1. If f is an Abstraction (lambda), perform reduction:
           a. Shift arg's indices up (prepare for entering lambda scope)
           b. Substitute shifted arg for $0 in body
           c. Shift result's indices down (exit lambda scope)

        2. Otherwise, try to reduce f or x

        WHY THE SHIFTING?
        When arg enters the lambda's scope, its free variables need
        adjustment so they still refer to the same outer variables.

        EXAMPLE:
            (λ. $0 + 1) 5
            1. shifted_x = 5 (primitives don't change)
            2. body[$0 := 5] = (5 + 1)
            3. shift down = (5 + 1)
            Result: 5 + 1
        """
        # If f is a lambda, perform beta reduction
        if isinstance(self.f, Abstraction):
            # (λ. body) x  -->  body[0 := x]

            # Step 1: Prepare x for substitution
            # When x enters lambda scope, its free indices must be shifted up
            shifted_x = self.x.shift(1, 0)

            # Step 2: Substitute x for $0 in body
            result = self.f.body.substitute(0, shifted_x)

            # Step 3: Shift result indices back down (we've exited the lambda)
            return result.shift(-1, 0)

        # Otherwise, try to reduce subterms
        f_reduced = self.f.beta_reduce()
        if f_reduced != self.f:
            return Application(f_reduced, self.x)

        x_reduced = self.x.beta_reduce()
        if x_reduced != self.x:
            return Application(self.f, x_reduced)

        return self  # No reduction possible

    def walk(self, func: Callable[[Program], None]) -> None:
        func(self)
        self.f.walk(func)
        self.x.walk(func)

    def clone(self) -> Program:
        return Application(self.f.clone(), self.x.clone())


# ============================================================================
# ABSTRACTION
# ============================================================================

@dataclass(frozen=True)
class Abstraction(Program):
    """
    Lambda abstraction: λ. body

    Creates a function that binds a new variable in the body.
    The bound variable is accessed as $0 in the body.

    EXAMPLES:
        λ. $0           Identity function (λx. x)
        λ. λ. $1        First of two args (λx. λy. x)
        λ. λ. $0        Second of two args (λx. λy. y)
        λ. $0 + 1       Increment (λx. x + 1)

    EVALUATION:
        Evaluating a lambda creates a closure - a function that
        captures the current environment.

        (λ. $0 + $1).evaluate([outer_val])
        Returns a function that when called with arg:
            evaluates body in [arg, outer_val]

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Same structure (body field)
    - Both use de Bruijn indices
    """
    body: Program  # The lambda body

    def __str__(self) -> str:
        return f"(λ {self.body})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Abstraction) and self.body == other.body

    def __hash__(self) -> int:
        return hash(('abs', self.body))

    def evaluate(self, env: List[Any]) -> Any:
        """
        Return a closure that captures the environment.

        When the closure is called with an argument:
        1. Prepend the argument to the environment (becomes $0)
        2. Evaluate the body in this extended environment
        """
        def closure(arg):
            new_env = [arg] + env  # arg becomes $0, old $0 becomes $1, etc.
            return self.body.evaluate(new_env)
        return closure

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        """
        Infer type of lambda.

        λ. body has type (arg_type -> body_type)
        where body_type is inferred with arg_type as $0's type.
        """
        arg_type = ctx.fresh_type_variable()
        new_env = [arg_type] + env  # arg_type is $0's type
        ret_type = self.body.infer_type(ctx, new_env)
        return Arrow(ctx.apply(arg_type), ctx.apply(ret_type))

    def size(self) -> int:
        return 1 + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()

    def free_indices(self, depth: int = 0) -> Set[int]:
        # Inside lambda, depth increases (one more binding)
        return self.body.free_indices(depth + 1)

    def shift(self, amount: int, cutoff: int = 0) -> Program:
        # Inside lambda, cutoff increases (one more binding level)
        return Abstraction(self.body.shift(amount, cutoff + 1))

    def substitute(self, index: int, replacement: Program) -> Program:
        """
        Substitute in body, adjusting for the lambda's binding.

        Inside the lambda:
        - The target index shifts up by 1 (since there's a new $0)
        - The replacement must be shifted up (its free vars now have one more binder above)
        """
        shifted_replacement = replacement.shift(1, 0)
        return Abstraction(self.body.substitute(index + 1, shifted_replacement))

    def walk(self, func: Callable[[Program], None]) -> None:
        func(self)
        self.body.walk(func)

    def clone(self) -> Program:
        return Abstraction(self.body.clone())


# ============================================================================
# INVENTED (learned abstraction)
# ============================================================================

@dataclass(frozen=True)
class Invented(Program):
    """
    A learned abstraction discovered during compression.

    Invented primitives are reusable "library functions" extracted
    from successful programs. They reduce description length.

    EXAMPLE:
        If many programs use: (all (λ. is_red $0) hand)
        Compression extracts: #all_red = λ. all (λ. is_red $0) $0

        Now programs can use: #all_red hand  (shorter!)

    DESCRIPTION LENGTH:
        When counting program size, an Invented counts as 1
        (it's a library reference), not its full body size.

        This is the MDL (Minimum Description Length) principle:
        frequently-used patterns are worth abstracting.

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Same concept
    - Original: Can have a type field
    - Ours: Type is inferred from body
    """
    body: Program                 # The abstracted expression
    name: Optional[str] = None    # Optional human-readable name

    def __str__(self) -> str:
        if self.name:
            return f"#{self.name}"
        return f"#({self.body})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Invented) and self.body == other.body

    def __hash__(self) -> int:
        return hash(('invented', self.body))

    def evaluate(self, env: List[Any]) -> Any:
        # Evaluate the body in the current environment
        return self.body.evaluate(env)

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        return self.body.infer_type(ctx, env)

    def size(self) -> int:
        # When USED, an invented counts as 1 (it's a library reference)
        # This is key for MDL: factoring out common patterns saves description length
        return 1

    def depth(self) -> int:
        return 1

    def free_indices(self, depth: int = 0) -> Set[int]:
        return self.body.free_indices(depth)

    def shift(self, amount: int, cutoff: int = 0) -> Program:
        return Invented(self.body.shift(amount, cutoff), self.name)

    def substitute(self, index: int, replacement: Program) -> Program:
        return Invented(self.body.substitute(index, replacement), self.name)

    def walk(self, func: Callable[[Program], None]) -> None:
        func(self)
        self.body.walk(func)

    def clone(self) -> Program:
        return Invented(self.body.clone(), self.name)

    def inner_size(self) -> int:
        """The actual size of the body (for compression analysis)."""
        return self.body.size()


# ============================================================================
# HOLE (placeholder for top-down synthesis)
# ============================================================================

# Global counter for generating unique hole IDs
_hole_id_counter = 0


def _next_hole_id() -> int:
    """Generate the next unique hole ID."""
    global _hole_id_counter
    _hole_id_counter += 1
    return _hole_id_counter


def reset_hole_counter() -> None:
    """Reset the hole ID counter (useful for testing)."""
    global _hole_id_counter
    _hole_id_counter = 0


@dataclass(frozen=True)
class Hole(Program):
    """
    A placeholder for an unknown subprogram during top-down synthesis.

    Holes represent "what still needs to be filled" in a partial program.
    They are typed, so we know what kind of program should replace them.

    USAGE IN TOP-DOWN ENUMERATION:
        1. Start with: Hole(request_type)   e.g., ?0:int->bool
        2. Replace with a production:       filter ?1:(α->bool) ?2:list(α)
        3. Continue until no holes remain

    FIELDS:
        tp: The type that the hole must be filled with
        id: Unique identifier (for tracking which hole to fill)

    EXAMPLE:
        ?0:int              A hole expecting an int
        ?1:(int->bool)      A hole expecting a predicate
        ?2:list(card)       A hole expecting a list of cards

    INVARIANTS:
        - Holes cannot be evaluated (raises RuntimeError)
        - Holes have no free indices (they're placeholders, not variables)
        - Holes are immutable and can be compared by (tp, id)

    COMPARISON WITH ORIGINAL DREAMCODER:
        Original DreamCoder has Hole and FragmentVariable nodes.
        We combine these into a single Hole node with a type and ID.
    """
    tp: Type              # The type this hole must be filled with
    id: int = field(default_factory=_next_hole_id)  # Unique identifier

    def __str__(self) -> str:
        """
        String representation: ?id:type

        Examples:
            ?0:int
            ?1:(int -> bool)
            ?2:list(card)
        """
        return f"?{self.id}:{self.tp}"

    def __eq__(self, other) -> bool:
        """Two holes are equal if they have the same ID and type."""
        return isinstance(other, Hole) and self.id == other.id and self.tp == other.tp

    def __hash__(self) -> int:
        return hash(('hole', self.id, self.tp))

    def evaluate(self, env: List[Any]) -> Any:
        """
        Holes cannot be evaluated - they represent incomplete programs.

        Raises:
            RuntimeError: Always, because holes are placeholders.
        """
        raise RuntimeError(
            f"Cannot evaluate incomplete program: hole ?{self.id}:{self.tp} "
            f"has not been filled"
        )

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        """
        Return the type stored in the hole.

        Since holes are typed placeholders, we know their type statically.
        We instantiate to get fresh type variables if needed.
        """
        return ctx.instantiate(self.tp)

    def size(self) -> int:
        """A hole counts as 1 node."""
        return 1

    def depth(self) -> int:
        """A hole has depth 1."""
        return 1

    def free_indices(self, depth: int = 0) -> Set[int]:
        """Holes have no free indices (they're not variable references)."""
        return set()

    def shift(self, amount: int, cutoff: int = 0) -> 'Program':
        """Shifting doesn't affect holes (no indices to shift)."""
        return self

    def substitute(self, index: int, replacement: Program) -> Program:
        """Substitution doesn't affect holes (no indices to substitute)."""
        return self

    def walk(self, f: Callable[[Program], None]) -> None:
        """Visit this hole."""
        f(self)

    def clone(self) -> Program:
        """Holes are immutable, return self."""
        return self


# ============================================================================
# HOLE UTILITY FUNCTIONS
# ============================================================================

def has_holes(program: Program) -> bool:
    """
    Check if a program contains any holes.

    A program with holes is incomplete and cannot be evaluated.
    This is used to check if synthesis has finished.

    Args:
        program: The program to check

    Returns:
        True if the program contains at least one Hole

    Example:
        >>> has_holes(Primitive('+', INT, lambda x: x))
        False
        >>> has_holes(Hole(INT))
        True
        >>> has_holes(Application(Primitive('+', ...), Hole(INT)))
        True
    """
    found = [False]  # Use list to allow mutation in nested function

    def check(p: Program) -> None:
        if isinstance(p, Hole):
            found[0] = True

    program.walk(check)
    return found[0]


def collect_holes(program: Program) -> List[Hole]:
    """
    Collect all holes in a program.

    Returns holes in traversal order (pre-order: parent before children,
    left before right).

    Args:
        program: The program to search

    Returns:
        List of all Hole nodes in the program

    Example:
        >>> p = Application(Hole(INT, 1), Hole(BOOL, 2))
        >>> holes = collect_holes(p)
        >>> [h.id for h in holes]
        [1, 2]
    """
    holes = []

    def collect(p: Program) -> None:
        if isinstance(p, Hole):
            holes.append(p)

    program.walk(collect)
    return holes


def count_holes(program: Program) -> int:
    """
    Count the number of holes in a program.

    Args:
        program: The program to count holes in

    Returns:
        Number of holes
    """
    return len(collect_holes(program))


def find_first_hole(program: Program) -> Optional[Hole]:
    """
    Find the first (leftmost) hole in a program.

    The "first" hole is defined by pre-order traversal:
    - Visit node before children
    - Visit left child before right child

    This gives a deterministic order for hole-filling in top-down synthesis.

    Args:
        program: The program to search

    Returns:
        The first Hole found, or None if the program is complete

    Example:
        >>> p = Application(Hole(INT, 1), Hole(BOOL, 2))
        >>> find_first_hole(p).id
        1
    """
    # For efficiency, we don't use walk() here because we want to stop early
    if isinstance(program, Hole):
        return program

    if isinstance(program, (Primitive, Index)):
        return None

    if isinstance(program, Application):
        # Check function first (left), then argument (right)
        hole = find_first_hole(program.f)
        if hole is not None:
            return hole
        return find_first_hole(program.x)

    if isinstance(program, Abstraction):
        return find_first_hole(program.body)

    if isinstance(program, Invented):
        return find_first_hole(program.body)

    return None


def substitute_hole(program: Program, hole_id: int, replacement: Program) -> Program:
    """
    Replace a specific hole with a program.

    This is the core operation for top-down synthesis:
    take a partial program and fill in one of its holes.

    Args:
        program: The program containing the hole
        hole_id: The ID of the hole to replace
        replacement: The program to put in place of the hole

    Returns:
        A new program with the hole replaced

    Note:
        If the hole is not found, returns the original program unchanged.
        If replacement contains holes, the result is still partial.

    Example:
        >>> p = Application(Hole(INT, 1), Hole(BOOL, 2))
        >>> filled = substitute_hole(p, 1, Primitive('5', INT, 5))
        >>> str(filled)
        '5 ?2:bool'
    """
    if isinstance(program, Hole):
        if program.id == hole_id:
            return replacement
        return program

    if isinstance(program, (Primitive, Index)):
        return program

    if isinstance(program, Application):
        new_f = substitute_hole(program.f, hole_id, replacement)
        new_x = substitute_hole(program.x, hole_id, replacement)
        if new_f is program.f and new_x is program.x:
            return program  # No change, return original for efficiency
        return Application(new_f, new_x)

    if isinstance(program, Abstraction):
        new_body = substitute_hole(program.body, hole_id, replacement)
        if new_body is program.body:
            return program
        return Abstraction(new_body)

    if isinstance(program, Invented):
        new_body = substitute_hole(program.body, hole_id, replacement)
        if new_body is program.body:
            return program
        return Invented(new_body, program.name)

    return program


def max_hole_id(program: Program) -> int:
    """
    Find the maximum hole ID in a program.

    Useful for generating new unique hole IDs when expanding a partial program.

    Args:
        program: The program to search

    Returns:
        Maximum hole ID found, or -1 if no holes exist
    """
    holes = collect_holes(program)
    if not holes:
        return -1
    return max(h.id for h in holes)


def create_hole_for_type(tp: Type) -> Hole:
    """
    Create a new hole with a unique ID for the given type.

    Args:
        tp: The type the hole should have

    Returns:
        A new Hole with a fresh unique ID
    """
    return Hole(tp)  # Uses default_factory to generate ID


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_args(func: Program, *args: Program) -> Program:
    """
    Helper: apply multiple arguments to a function.

    apply_args(f, a, b, c) = ((f a) b) c

    This is left-associative application, matching curried functions.

    EXAMPLE:
        apply_args(add, Primitive('3'), Primitive('4'))
        = Application(Application(add, 3), 4)
        = (add 3 4) in string form
    """
    result = func
    for arg in args:
        result = Application(result, arg)
    return result


def multi_lambda(n: int, body: Program) -> Program:
    """
    Create n nested lambda abstractions.

    multi_lambda(2, body) = λ. λ. body

    EXAMPLE:
        multi_lambda(3, Index(2))
        = λ. λ. λ. $2
        = λx. λy. λz. x (first of three args)
    """
    result = body
    for _ in range(n):
        result = Abstraction(result)
    return result


def program_size(p: Program) -> int:
    """Alias for p.size()."""
    return p.size()


def program_depth(p: Program) -> int:
    """Alias for p.depth()."""
    return p.depth()


# ============================================================================
# AST UTILITIES - PRETTY PRINTING
# ============================================================================

def pretty_print(program: Program, indent: int = 0, indent_str: str = "  ") -> str:
    """
    Create a multi-line, indented representation of a program.

    This is useful for visualizing complex programs with deep nesting.
    Each level of nesting adds one level of indentation.

    Args:
        program: The program to format
        indent: Current indentation level (default 0)
        indent_str: String to use for each indentation level (default 2 spaces)

    Returns:
        A multi-line formatted string

    Example:
        >>> p = Application(Application(add, one), Application(mul, two))
        >>> print(pretty_print(p))
        (App
          (App
            +
            1)
          (App
            *
            2))
    """
    prefix = indent_str * indent

    if isinstance(program, Primitive):
        return f"{prefix}Prim:{program.name}"

    if isinstance(program, Index):
        return f"{prefix}Var:${program.i}"

    if isinstance(program, Hole):
        return f"{prefix}Hole:?{program.id}:{program.tp}"

    if isinstance(program, Application):
        f_str = pretty_print(program.f, indent + 1, indent_str)
        x_str = pretty_print(program.x, indent + 1, indent_str)
        return f"{prefix}(App\n{f_str}\n{x_str})"

    if isinstance(program, Abstraction):
        body_str = pretty_print(program.body, indent + 1, indent_str)
        return f"{prefix}(λ\n{body_str})"

    if isinstance(program, Invented):
        if program.name:
            return f"{prefix}Lib:#{program.name}"
        body_str = pretty_print(program.body, indent + 1, indent_str)
        return f"{prefix}(Invented\n{body_str})"

    return f"{prefix}Unknown:{program}"


def compact_str(program: Program) -> str:
    """
    Create a compact string representation without extra parentheses.

    This is more readable than __str__ for nested applications.

    Args:
        program: The program to format

    Returns:
        A compact string representation
    """
    if isinstance(program, Primitive):
        return program.name

    if isinstance(program, Index):
        return f"${program.i}"

    if isinstance(program, Hole):
        return f"?{program.id}"

    if isinstance(program, Abstraction):
        return f"λ.{compact_str(program.body)}"

    if isinstance(program, Application):
        # Collect all arguments for curried application
        func = program.f
        args = [program.x]
        while isinstance(func, Application):
            args.insert(0, func.x)
            func = func.f
        func_str = compact_str(func)
        args_str = " ".join(compact_str(a) for a in args)
        return f"({func_str} {args_str})"

    if isinstance(program, Invented):
        if program.name:
            return f"#{program.name}"
        return f"#[{compact_str(program.body)}]"

    return str(program)


# ============================================================================
# AST UTILITIES - PROGRAM TRANSFORMER
# ============================================================================

class ProgramTransformer:
    """
    Base class for transforming program ASTs.

    Subclass this and override specific transform_X methods to create
    custom transformations. The default behavior is to recursively
    transform children and rebuild the node.

    This is the "visitor pattern" for building new ASTs.

    USAGE:
        class DoubleConstants(ProgramTransformer):
            def transform_primitive(self, p: Primitive) -> Program:
                if isinstance(p.value, (int, float)):
                    return Primitive(p.name, p.tp, p.value * 2)
                return p

        transformer = DoubleConstants()
        new_program = transformer.transform(old_program)

    COMPARISON WITH walk():
        - walk() visits nodes without modification (read-only)
        - ProgramTransformer builds a new AST (possibly modified)
    """

    def transform(self, program: Program) -> Program:
        """
        Transform a program, dispatching to the appropriate method.

        This is the main entry point. It dispatches to transform_primitive,
        transform_index, etc. based on the program type.
        """
        if isinstance(program, Primitive):
            return self.transform_primitive(program)
        elif isinstance(program, Index):
            return self.transform_index(program)
        elif isinstance(program, Hole):
            return self.transform_hole(program)
        elif isinstance(program, Application):
            return self.transform_application(program)
        elif isinstance(program, Abstraction):
            return self.transform_abstraction(program)
        elif isinstance(program, Invented):
            return self.transform_invented(program)
        else:
            raise ValueError(f"Unknown program type: {type(program)}")

    def transform_primitive(self, program: Primitive) -> Program:
        """Transform a Primitive. Default: return unchanged."""
        return program

    def transform_index(self, program: Index) -> Program:
        """Transform an Index. Default: return unchanged."""
        return program

    def transform_hole(self, program: Hole) -> Program:
        """Transform a Hole. Default: return unchanged."""
        return program

    def transform_application(self, program: Application) -> Program:
        """
        Transform an Application.

        Default: transform children and rebuild.
        """
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program  # No change
        return Application(new_f, new_x)

    def transform_abstraction(self, program: Abstraction) -> Program:
        """
        Transform an Abstraction.

        Default: transform body and rebuild.
        """
        new_body = self.transform(program.body)
        if new_body is program.body:
            return program
        return Abstraction(new_body)

    def transform_invented(self, program: Invented) -> Program:
        """
        Transform an Invented.

        Default: transform body and rebuild.
        """
        new_body = self.transform(program.body)
        if new_body is program.body:
            return program
        return Invented(new_body, program.name)


class SubstitutePrimitive(ProgramTransformer):
    """
    Transformer that replaces one primitive with another program.

    Useful for inlining or replacing primitives.

    Example:
        >>> transformer = SubstitutePrimitive('old_name', new_program)
        >>> result = transformer.transform(program)
    """

    def __init__(self, name: str, replacement: Program):
        self.name = name
        self.replacement = replacement

    def transform_primitive(self, program: Primitive) -> Program:
        if program.name == self.name:
            return self.replacement
        return program


class InlineInvented(ProgramTransformer):
    """
    Transformer that inlines all Invented nodes (expands library references).

    Useful for getting the "full" program without library abstractions.
    """

    def transform_invented(self, program: Invented) -> Program:
        # First transform the body recursively
        return self.transform(program.body)


# ============================================================================
# AST UTILITIES - ANALYSIS FUNCTIONS
# ============================================================================

def collect_primitives(program: Program) -> Set[Primitive]:
    """
    Collect all primitives used in a program.

    Args:
        program: The program to analyze

    Returns:
        Set of Primitive objects used in the program

    Example:
        >>> p = Application(add, Application(mul, one))
        >>> prims = collect_primitives(p)
        >>> sorted(p.name for p in prims)
        ['*', '+', '1']
    """
    primitives = set()

    def collect(p: Program) -> None:
        if isinstance(p, Primitive):
            primitives.add(p)

    program.walk(collect)
    return primitives


def collect_primitive_names(program: Program) -> Set[str]:
    """
    Collect names of all primitives used in a program.

    Args:
        program: The program to analyze

    Returns:
        Set of primitive names (strings)
    """
    return {p.name for p in collect_primitives(program)}


def count_primitive_uses(program: Program) -> Dict[str, int]:
    """
    Count how many times each primitive is used.

    Args:
        program: The program to analyze

    Returns:
        Dictionary mapping primitive names to use counts

    Example:
        >>> p = Application(add, Application(add, one))
        >>> count_primitive_uses(p)
        {'+': 2, '1': 1}
    """
    counts: Dict[str, int] = {}

    def count(p: Program) -> None:
        if isinstance(p, Primitive):
            counts[p.name] = counts.get(p.name, 0) + 1

    program.walk(count)
    return counts


def find_shared_subexpressions(program: Program) -> List[Tuple[Program, int]]:
    """
    Find subexpressions that appear multiple times in a program.

    This is useful for compression: shared subexpressions are good
    candidates for abstraction.

    Args:
        program: The program to analyze

    Returns:
        List of (subprogram, count) tuples for subprograms appearing
        more than once, sorted by count (most frequent first)

    Note:
        Only considers subprograms of size > 1 (single nodes are too small
        to benefit from abstraction).

    Example:
        >>> # If (+ 1 1) appears 3 times in a program
        >>> shared = find_shared_subexpressions(program)
        >>> # Returns [(Application(+, Application(+, 1, 1)), 3), ...]
    """
    # Count all subprograms
    subprog_counts: Dict[str, Tuple[Program, int]] = {}

    def count_subprog(p: Program) -> None:
        # Only consider non-trivial subprograms
        if p.size() > 1:
            key = str(p)
            if key in subprog_counts:
                _, count = subprog_counts[key]
                subprog_counts[key] = (p, count + 1)
            else:
                subprog_counts[key] = (p, 1)

    program.walk(count_subprog)

    # Filter to those appearing more than once
    shared = [(p, count) for p, count in subprog_counts.values() if count > 1]

    # Sort by count descending, then by size descending (prefer larger shared subexprs)
    shared.sort(key=lambda x: (-x[1], -x[0].size()))

    return shared


def alpha_equivalent(p1: Program, p2: Program) -> bool:
    """
    Check if two programs are alpha-equivalent (structurally identical).

    For de Bruijn indexed programs, this is the same as structural equality,
    but we handle some special cases:
    - Invented nodes with different names but same body are equivalent
    - Holes with different IDs but same type are equivalent

    Args:
        p1: First program
        p2: Second program

    Returns:
        True if programs are alpha-equivalent

    Example:
        >>> alpha_equivalent(Abstraction(Index(0)), Abstraction(Index(0)))
        True
        >>> alpha_equivalent(Hole(INT, 1), Hole(INT, 2))
        True  # Same type, different ID
    """
    if type(p1) != type(p2):
        return False

    if isinstance(p1, Primitive):
        return p1.name == p2.name

    if isinstance(p1, Index):
        return p1.i == p2.i

    if isinstance(p1, Hole):
        # Holes are equivalent if they have the same type
        # (IDs are just for tracking during synthesis)
        return p1.tp == p2.tp

    if isinstance(p1, Application):
        return (alpha_equivalent(p1.f, p2.f) and
                alpha_equivalent(p1.x, p2.x))

    if isinstance(p1, Abstraction):
        return alpha_equivalent(p1.body, p2.body)

    if isinstance(p1, Invented):
        # Invented nodes are equivalent if their bodies are equivalent
        # (names are just for display)
        return alpha_equivalent(p1.body, p2.body)

    return False


def program_to_tree_dict(program: Program) -> Dict[str, Any]:
    """
    Convert a program to a dictionary representation (for JSON/visualization).

    Args:
        program: The program to convert

    Returns:
        Dictionary with 'type', 'value', and optionally 'children' keys

    Example:
        >>> p = Application(add, one)
        >>> program_to_tree_dict(p)
        {'type': 'Application', 'children': [
            {'type': 'Primitive', 'value': '+'},
            {'type': 'Primitive', 'value': '1'}
        ]}
    """
    if isinstance(program, Primitive):
        return {'type': 'Primitive', 'value': program.name}

    if isinstance(program, Index):
        return {'type': 'Index', 'value': program.i}

    if isinstance(program, Hole):
        return {'type': 'Hole', 'id': program.id, 'hole_type': str(program.tp)}

    if isinstance(program, Application):
        return {
            'type': 'Application',
            'children': [
                program_to_tree_dict(program.f),
                program_to_tree_dict(program.x)
            ]
        }

    if isinstance(program, Abstraction):
        return {
            'type': 'Abstraction',
            'children': [program_to_tree_dict(program.body)]
        }

    if isinstance(program, Invented):
        return {
            'type': 'Invented',
            'name': program.name,
            'children': [program_to_tree_dict(program.body)]
        }

    return {'type': 'Unknown', 'value': str(program)}


def lambda_depth(program: Program) -> int:
    """
    Count the maximum nesting depth of lambda abstractions.

    This is different from AST depth - it only counts λ nodes.

    Args:
        program: The program to analyze

    Returns:
        Maximum number of nested lambdas
    """
    if isinstance(program, Abstraction):
        return 1 + lambda_depth(program.body)
    if isinstance(program, Application):
        return max(lambda_depth(program.f), lambda_depth(program.x))
    if isinstance(program, Invented):
        return lambda_depth(program.body)
    return 0


def uses_variable(program: Program, index: int) -> bool:
    """
    Check if a program uses a specific de Bruijn index.

    Args:
        program: The program to check
        index: The de Bruijn index to look for

    Returns:
        True if the program contains Index(index)
    """
    found = [False]

    def check(p: Program) -> None:
        if isinstance(p, Index) and p.i == index:
            found[0] = True

    program.walk(check)
    return found[0]


def count_applications(program: Program) -> int:
    """
    Count the number of applications in a program.

    Useful for complexity analysis.
    """
    count = [0]

    def counter(p: Program) -> None:
        if isinstance(p, Application):
            count[0] += 1

    program.walk(counter)
    return count[0]


# ============================================================================
# PROGRAM PARSING (simple parser for debugging)
# ============================================================================

def _tokenize(s: str) -> List[str]:
    """
    Tokenize a program string into tokens.

    Returns list of tokens where each token is either:
    - A primitive name (alphanumeric, +, -, *, /, etc.)
    - $n (de Bruijn index)
    - ?n:type (hole)
    - λ
    - ( or )

    EXAMPLE:
        "(λ $0)"  →  ['(', 'λ', '$0', ')']
        "(+ 1 2)" →  ['(', '+', '1', '2', ')']
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        elif c in '()':
            tokens.append(c)
            i += 1
        elif c == 'λ':
            tokens.append('λ')
            i += 1
        elif c == '$':
            # De Bruijn index: $0, $1, $12, etc.
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c == '?':
            # Hole: ?0, ?1:type, etc.
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] == ':' or s[j].isalnum() or s[j] in '->()'):
                if s[j].isspace() or s[j] in '()':
                    break
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            # Primitive name - collect until whitespace or paren
            j = i
            while j < len(s) and not s[j].isspace() and s[j] not in '()':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_expr(tokens: List[str], pos: int, primitives: Dict[str, Primitive]) -> Tuple[Program, int]:
    """
    Parse an expression from tokens starting at pos.

    Returns (program, new_position).

    GRAMMAR (informal):
        expr ::= $n                    # Index
               | ?n                    # Hole (simple)
               | ?n:type               # Hole (with type)
               | name                  # Primitive
               | (λ expr+)            # Lambda with body
               | (expr expr*)          # Application

    EXAMPLE:
        "(λ $0)"  →  Abstraction(Index(0))
        "(+ 1 2)" →  Application(Application(+, 1), 2)
    """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of input")

    tok = tokens[pos]

    # De Bruijn index
    if tok.startswith('$'):
        return Index(int(tok[1:])), pos + 1

    # Hole
    if tok.startswith('?'):
        # Parse ?id or ?id:type
        if ':' in tok:
            id_part, type_part = tok[1:].split(':', 1)
            hole_id = int(id_part) if id_part else _next_hole_id()
            # For simplicity, treat type as a base type by name
            # A full implementation would parse the type properly
            hole_type = BaseType(type_part) if type_part else TypeVariable(0)
        else:
            hole_id = int(tok[1:]) if len(tok) > 1 else _next_hole_id()
            hole_type = TypeVariable(0)  # Unknown type
        return Hole(hole_type, hole_id), pos + 1

    # Primitive
    if tok in primitives:
        return primitives[tok], pos + 1

    # Parenthesized expression
    if tok == '(':
        pos += 1
        if pos >= len(tokens):
            raise ValueError("Unexpected end after '('")

        # Lambda: (λ body) where body can be multiple expressions forming an application
        if tokens[pos] == 'λ':
            pos += 1
            # Parse all expressions in the body until ')'
            body_exprs = []
            while pos < len(tokens) and tokens[pos] != ')':
                expr, pos = _parse_expr(tokens, pos, primitives)
                body_exprs.append(expr)

            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Expected ')' after lambda body")
            pos += 1

            if len(body_exprs) == 0:
                raise ValueError("Lambda body cannot be empty")
            elif len(body_exprs) == 1:
                body = body_exprs[0]
            else:
                # Multiple exprs = application: (λ f x y) -> (λ ((f x) y))
                body = body_exprs[0]
                for arg in body_exprs[1:]:
                    body = Application(body, arg)

            return Abstraction(body), pos

        # Application: (f x1 x2 ... xn) = (...((f x1) x2) ... xn)
        # Parse all expressions until we hit ')'
        exprs = []
        while pos < len(tokens) and tokens[pos] != ')':
            expr, pos = _parse_expr(tokens, pos, primitives)
            exprs.append(expr)

        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Expected ')' to close application")
        pos += 1

        if len(exprs) == 0:
            raise ValueError("Empty application")
        if len(exprs) == 1:
            return exprs[0], pos

        # Left-associate: (f x y z) -> (((f x) y) z)
        result = exprs[0]
        for arg in exprs[1:]:
            result = Application(result, arg)
        return result, pos

    raise ValueError(f"Unexpected token: {tok}")


def parse_program(s: str, primitives: Dict[str, Primitive]) -> Program:
    """
    Parse a program string into a Program object.

    SYNTAX:
        $n          -> Index(n)
        ?n          -> Hole with ID n
        ?n:type     -> Hole with ID n and type
        name        -> Primitive lookup
        (λ body)    -> Abstraction
        (f x y ...) -> Left-associated Application: ((f x) y) ...

    EXAMPLES:
        "true"        -> Primitive('true')
        "$0"          -> Index(0)
        "(λ $0)"      -> Abstraction(Index(0))
        "(map f xs)"  -> Application(Application(map, f), xs)
        "?0:int"      -> Hole(INT, 0)

    PARAMETERS:
        s: The program string to parse
        primitives: Dictionary mapping names to Primitive objects
    """
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("Empty program string")

    program, pos = _parse_expr(tokens, 0, primitives)
    if pos < len(tokens):
        raise ValueError(f"Unexpected tokens after expression: {tokens[pos:]}")
    return program


# ============================================================================
# TESTS (run with: python program.py)
# ============================================================================

if __name__ == "__main__":
    print("=== Program Representation Tests ===\n")

    # Reset hole counter for consistent test output
    reset_hole_counter()

    # Create some primitives
    add = Primitive('+', arrow(INT, INT, INT),
                    lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT),
                    lambda x: lambda y: x * y)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)

    # Test simple application: (+ 1 2)
    p1 = apply_args(add, one, two)
    print(f"Program: {p1}")
    print(f"Size: {p1.size()}, Depth: {p1.depth()}")
    print(f"Result: {p1.evaluate([])}")
    print(f"Is complete: {p1.is_complete()}")

    # Test lambda: (λ (+ $0 1))
    p2 = Abstraction(apply_args(add, Index(0), one))
    print(f"\nProgram: {p2}")
    print(f"Size: {p2.size()}, Depth: {p2.depth()}")

    # Type inference
    ctx = TypeContext()
    t2 = p2.infer_type(ctx, [])
    print(f"Type: {ctx.apply(t2)}")

    # Evaluate: apply to 5
    inc = p2.evaluate([])
    print(f"Applied to 5: {inc(5)}")

    # Test beta reduction
    # (λ (+ $0 1)) 5  -->  (+ 5 1)  -->  6
    five = Primitive('5', INT, 5)
    p3 = Application(p2, five)
    print(f"\nBefore reduction: {p3}")
    p3_reduced = p3.beta_reduce()
    print(f"After reduction: {p3_reduced}")
    print(f"Evaluated: {p3.evaluate([])}")

    # Test nested lambdas: (λ λ (+ $0 $1))  -- adds two numbers
    p4 = Abstraction(Abstraction(apply_args(add, Index(0), Index(1))))
    print(f"\nDouble lambda: {p4}")
    add_fn = p4.evaluate([])
    print(f"3 + 4 = {add_fn(3)(4)}")

    # Test free indices
    open_prog = apply_args(add, Index(0), Index(2))
    print(f"\nOpen program: {open_prog}")
    print(f"Free indices: {open_prog.free_indices()}")
    print(f"Is closed: {open_prog.is_closed()}")

    closed_prog = Abstraction(apply_args(add, Index(0), one))
    print(f"Closed program: {closed_prog}")
    print(f"Free indices: {closed_prog.free_indices()}")
    print(f"Is closed: {closed_prog.is_closed()}")

    # Test subprograms
    print(f"\nSubprograms of {p1}:")
    for sp in p1.subprograms():
        print(f"  {sp}")

    # ==========================================
    # Test Hole functionality
    # ==========================================
    print("\n=== Hole Tests ===\n")

    reset_hole_counter()

    # Create holes
    h1 = Hole(INT)
    h2 = Hole(arrow(INT, BOOL))
    print(f"Hole 1: {h1}")
    print(f"Hole 2: {h2}")

    # Test has_holes
    print(f"\nhas_holes(add): {has_holes(add)}")
    print(f"has_holes(h1): {has_holes(h1)}")

    # Create partial program: (+ ?1 ?2)
    partial = apply_args(add, Hole(INT), Hole(INT))
    print(f"\nPartial program: {partial}")
    print(f"has_holes: {has_holes(partial)}")
    print(f"is_complete: {partial.is_complete()}")
    print(f"count_holes: {count_holes(partial)}")

    # Collect holes
    holes = collect_holes(partial)
    print(f"Holes: {[str(h) for h in holes]}")

    # Find first hole
    first = find_first_hole(partial)
    print(f"First hole: {first}")

    # Substitute a hole
    filled = substitute_hole(partial, first.id, one)
    print(f"After filling first hole with 1: {filled}")
    print(f"Still has holes: {has_holes(filled)}")

    # Fill remaining hole
    remaining = find_first_hole(filled)
    fully_filled = substitute_hole(filled, remaining.id, two)
    print(f"Fully filled: {fully_filled}")
    print(f"Is complete: {fully_filled.is_complete()}")
    print(f"Evaluated: {fully_filled.evaluate([])}")

    # Test that holes can't be evaluated
    print("\nTrying to evaluate partial program...")
    try:
        partial.evaluate([])
    except RuntimeError as e:
        print(f"Error (expected): {e}")

    # ==========================================
    # Test AST Utilities
    # ==========================================
    print("\n=== AST Utility Tests ===\n")

    # Pretty print
    complex_prog = Abstraction(apply_args(add, apply_args(mul, Index(0), two), one))
    print("Pretty print of (λ (+ (* $0 2) 1)):")
    print(pretty_print(complex_prog))

    # Compact string
    print(f"\nCompact: {compact_str(complex_prog)}")

    # Collect primitives
    prims = collect_primitives(complex_prog)
    print(f"\nPrimitives used: {sorted(p.name for p in prims)}")

    # Count primitive uses
    repeated = apply_args(add, apply_args(add, one, one), one)
    counts = count_primitive_uses(repeated)
    print(f"Primitive counts in {repeated}: {counts}")

    # Shared subexpressions
    print(f"\nShared subexpressions in {repeated}:")
    shared = find_shared_subexpressions(repeated)
    for expr, count in shared:
        print(f"  {expr} appears {count} times")

    # Alpha equivalence
    p_a = Abstraction(Index(0))
    p_b = Abstraction(Index(0))
    print(f"\nalpha_equivalent((λ $0), (λ $0)): {alpha_equivalent(p_a, p_b)}")

    h_a = Hole(INT, 1)
    h_b = Hole(INT, 2)
    print(f"alpha_equivalent(?1:int, ?2:int): {alpha_equivalent(h_a, h_b)}")

    # Transformer example
    print("\n=== Transformer Test ===")

    class ReplaceOne(ProgramTransformer):
        """Replace all occurrences of '1' with '2'"""
        def transform_primitive(self, p: Primitive) -> Program:
            if p.name == '1':
                return two
            return p

    transformer = ReplaceOne()
    original = apply_args(add, one, one)
    transformed = transformer.transform(original)
    print(f"Original: {original}")
    print(f"After replacing 1->2: {transformed}")

    # Program to tree dict
    print("\n=== Tree Dict ===")
    tree = program_to_tree_dict(p1)
    import json
    print(json.dumps(tree, indent=2))

    print("\n=== All Program Tests OK ===")
