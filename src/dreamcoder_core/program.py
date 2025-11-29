"""
Lambda Calculus Program Representation

Implements programs using de Bruijn indices, following DreamCoder's approach.

Program types:
- Primitive: Built-in operations like +, map, filter
- Application: Function application (f x)
- Abstraction: Lambda abstraction (λ. body)
- Index: de Bruijn index variable ($0, $1, ...)
- Invented: Learned abstractions from compression

De Bruijn indices avoid variable naming:
- $0 refers to the innermost bound variable
- $1 refers to the next outer binding
- etc.

Example: λx. λy. x  is written as  λ. λ. $1
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
import math

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)


class Program(ABC):
    """Abstract base class for all program expressions."""

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable string representation."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def evaluate(self, env: List[Any]) -> Any:
        """
        Evaluate the program in an environment.

        The environment is a list where env[i] is the value of $i.
        """
        pass

    @abstractmethod
    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        """
        Infer the type of this program.

        Args:
            ctx: Type context for fresh variables and unification
            env: Type environment, env[i] is the type of $i
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the AST size (number of nodes)."""
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return the AST depth."""
        pass

    @abstractmethod
    def free_indices(self, depth: int = 0) -> Set[int]:
        """Return free de Bruijn indices (adjusted for nesting)."""
        pass

    @abstractmethod
    def shift(self, amount: int, cutoff: int = 0) -> 'Program':
        """Shift de Bruijn indices >= cutoff by amount."""
        pass

    @abstractmethod
    def substitute(self, index: int, replacement: 'Program') -> 'Program':
        """Substitute a program for a de Bruijn index."""
        pass

    def is_closed(self) -> bool:
        """Check if this program has no free variables."""
        return len(self.free_indices()) == 0

    def beta_reduce(self) -> 'Program':
        """Perform one step of beta reduction if possible."""
        return self

    def normalize(self) -> 'Program':
        """Fully normalize (beta-reduce until fixed point)."""
        current = self
        for _ in range(1000):  # Prevent infinite loops
            reduced = current.beta_reduce()
            if reduced == current:
                return current
            current = reduced
        return current

    @abstractmethod
    def walk(self, f: Callable[['Program'], None]) -> None:
        """Walk the AST, calling f on each node."""
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


# ============================================================================
# PRIMITIVE
# ============================================================================

@dataclass(frozen=True)
class Primitive(Program):
    """
    A built-in primitive operation.

    Attributes:
        name: Identifier for the primitive
        tp: The type of the primitive
        value: The implementation (a Python callable or value)
    """
    name: str
    tp: Type
    value: Any

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, Primitive) and self.name == other.name

    def __hash__(self) -> int:
        return hash(('primitive', self.name))

    def evaluate(self, env: List[Any]) -> Any:
        return self.value

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        # Instantiate the type (fresh type variables for polymorphism)
        return ctx.instantiate(self.tp)

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def free_indices(self, depth: int = 0) -> Set[int]:
        return set()

    def shift(self, amount: int, cutoff: int = 0) -> 'Program':
        return self

    def substitute(self, index: int, replacement: Program) -> Program:
        return self

    def walk(self, f: Callable[[Program], None]) -> None:
        f(self)

    def clone(self) -> Program:
        return self  # Primitives are immutable


# ============================================================================
# INDEX (de Bruijn variable)
# ============================================================================

@dataclass(frozen=True)
class Index(Program):
    """
    A de Bruijn index variable.

    $0 refers to the most recently bound variable,
    $1 to the one before that, etc.
    """
    i: int

    def __str__(self) -> str:
        return f"${self.i}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Index) and self.i == other.i

    def __hash__(self) -> int:
        return hash(('index', self.i))

    def evaluate(self, env: List[Any]) -> Any:
        if self.i >= len(env):
            raise RuntimeError(f"Index ${self.i} not in environment of size {len(env)}")
        return env[self.i]

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        if self.i >= len(env):
            raise UnificationError(f"Index ${self.i} not in type environment")
        return env[self.i]

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def free_indices(self, depth: int = 0) -> Set[int]:
        if self.i >= depth:
            return {self.i - depth}
        return set()

    def shift(self, amount: int, cutoff: int = 0) -> Program:
        if self.i >= cutoff:
            return Index(self.i + amount)
        return self

    def substitute(self, index: int, replacement: Program) -> Program:
        if self.i == index:
            return replacement
        return self

    def walk(self, f: Callable[[Program], None]) -> None:
        f(self)

    def clone(self) -> Program:
        return self


# ============================================================================
# APPLICATION
# ============================================================================

@dataclass(frozen=True)
class Application(Program):
    """
    Function application: (f x)

    Represents applying function f to argument x.
    """
    f: Program
    x: Program

    def __str__(self) -> str:
        f_str = str(self.f)
        x_str = str(self.x)

        # Parenthesize arguments that are applications
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
        func = self.f.evaluate(env)
        arg = self.x.evaluate(env)

        if callable(func):
            return func(arg)
        else:
            raise RuntimeError(f"Cannot apply non-function: {func}")

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        f_type = self.f.infer_type(ctx, env)
        x_type = self.x.infer_type(ctx, env)

        # f must be an arrow type
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
        # If f is a lambda, perform beta reduction
        if isinstance(self.f, Abstraction):
            # (λ. body) x  -->  body[0 := x]
            shifted_x = self.x.shift(1, 0)  # Prepare x for substitution
            result = self.f.body.substitute(0, shifted_x)
            return result.shift(-1, 0)  # Adjust indices back

        # Otherwise, try to reduce subterms
        f_reduced = self.f.beta_reduce()
        if f_reduced != self.f:
            return Application(f_reduced, self.x)

        x_reduced = self.x.beta_reduce()
        if x_reduced != self.x:
            return Application(self.f, x_reduced)

        return self

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

    Binds a new variable in the body.
    The bound variable is accessed as $0 in the body.
    """
    body: Program

    def __str__(self) -> str:
        return f"(λ {self.body})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Abstraction) and self.body == other.body

    def __hash__(self) -> int:
        return hash(('abs', self.body))

    def evaluate(self, env: List[Any]) -> Any:
        # Return a Python function that captures the environment
        def closure(arg):
            new_env = [arg] + env
            return self.body.evaluate(new_env)
        return closure

    def infer_type(self, ctx: TypeContext, env: List[Type]) -> Type:
        arg_type = ctx.fresh_type_variable()
        new_env = [arg_type] + env
        ret_type = self.body.infer_type(ctx, new_env)
        return Arrow(ctx.apply(arg_type), ctx.apply(ret_type))

    def size(self) -> int:
        return 1 + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()

    def free_indices(self, depth: int = 0) -> Set[int]:
        return self.body.free_indices(depth + 1)

    def shift(self, amount: int, cutoff: int = 0) -> Program:
        return Abstraction(self.body.shift(amount, cutoff + 1))

    def substitute(self, index: int, replacement: Program) -> Program:
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

    Invented primitives are lambda expressions that have been
    factored out as reusable library functions.
    """
    body: Program
    name: Optional[str] = None  # Optional human-readable name

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
        # When used, an invented counts as 1 (it's a library reference)
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
# HELPER FUNCTIONS
# ============================================================================

def apply_args(func: Program, *args: Program) -> Program:
    """Helper: apply multiple arguments to a function."""
    result = func
    for arg in args:
        result = Application(result, arg)
    return result


def multi_lambda(n: int, body: Program) -> Program:
    """Create n nested lambda abstractions."""
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
# PROGRAM PARSING (simple parser for debugging)
# ============================================================================

def _tokenize(s: str) -> List[str]:
    """
    Tokenize a program string into tokens.

    Returns list of tokens where each token is either:
    - A primitive name (alphanumeric, +, -, *, /, etc.)
    - $n (de Bruijn index)
    - λ
    - ( or )
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
            # De Bruijn index
            j = i + 1
            while j < len(s) and s[j].isdigit():
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
    """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of input")

    tok = tokens[pos]

    # De Bruijn index
    if tok.startswith('$'):
        return Index(int(tok[1:])), pos + 1

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

    Syntax:
        $n          -> Index(n)
        name        -> Primitive lookup
        (λ body)    -> Abstraction
        (f x y ...) -> Left-associated Application: ((f x) y) ...

    Examples:
        "true" -> Primitive('true')
        "$0" -> Index(0)
        "(λ $0)" -> Abstraction(Index(0))
        "(map f xs)" -> Application(Application(Primitive('map'), f), xs)
    """
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("Empty program string")

    program, pos = _parse_expr(tokens, 0, primitives)
    if pos < len(tokens):
        raise ValueError(f"Unexpected tokens after expression: {tokens[pos:]}")
    return program


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Program Representation Tests ===\n")

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

    print("\n=== Program Representation OK ===")
