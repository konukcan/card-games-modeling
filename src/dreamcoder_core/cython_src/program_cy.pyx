# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cythonized Lambda Calculus Program Representation

Optimized version of program.py with:
- cdef classes for speed
- Typed attributes and local variables
- Disabled bounds checking for performance

Program types:
- Primitive: Built-in operations like +, map, filter
- Application: Function application (f x)
- Abstraction: Lambda abstraction (λ. body)
- Index: de Bruijn index variable ($0, $1, ...)
- Invented: Learned abstractions from compression
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Import type system - we use 'object' type in signatures for cross-module compatibility
# but cast to proper types inside methods for performance
from .type_system_cy import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError, arrow
)


# ============================================================================
# HELPER FOR SUBPROGRAM COLLECTION (avoids closure in cpdef)
# ============================================================================

cdef void _collect_subprograms(Program p, list result):
    """Recursive helper to collect all subprograms."""
    result.append(p)
    if isinstance(p, Application):
        _collect_subprograms((<Application>p).f, result)
        _collect_subprograms((<Application>p).x, result)
    elif isinstance(p, Abstraction):
        _collect_subprograms((<Abstraction>p).body, result)
    elif isinstance(p, Invented):
        _collect_subprograms((<Invented>p).body, result)
    # Primitive and Index have no children


# ============================================================================
# CLOSURE CLASS (used by Abstraction.evaluate to avoid cpdef closure issue)
# ============================================================================

class _Closure:
    """A callable that represents a lambda closure."""
    __slots__ = ['body', 'env']

    def __init__(self, body, env):
        self.body = body
        self.env = env

    def __call__(self, arg):
        new_env = [arg] + self.env
        return self.body.evaluate(new_env)


# ============================================================================
# PROGRAM BASE CLASS (Extension Type)
# ============================================================================

cdef class Program:
    """Abstract base class for all program expressions - Cython extension type."""

    def __str__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other) -> bint:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    cpdef object evaluate(self, list env):
        """Evaluate the program in an environment."""
        raise NotImplementedError

    cpdef object infer_type(self, object ctx, list env):
        """Infer the type of this program."""
        raise NotImplementedError

    cpdef int size(self):
        """Return the AST size (number of nodes)."""
        raise NotImplementedError

    cpdef int depth(self):
        """Return the AST depth."""
        raise NotImplementedError

    cpdef set free_indices(self, int depth_level=0):
        """Return free de Bruijn indices (adjusted for nesting)."""
        raise NotImplementedError

    cpdef Program shift(self, int amount, int cutoff=0):
        """Shift de Bruijn indices >= cutoff by amount."""
        raise NotImplementedError

    cpdef Program substitute(self, int index, Program replacement):
        """Substitute a program for a de Bruijn index."""
        raise NotImplementedError

    cpdef bint is_closed(self):
        """Check if this program has no free variables."""
        return len(self.free_indices(0)) == 0

    cpdef Program beta_reduce(self):
        """Perform one step of beta reduction if possible."""
        return self

    cpdef Program normalize(self):
        """Fully normalize (beta-reduce until fixed point)."""
        cdef Program current = self
        cdef Program reduced
        cdef int i
        for i in range(1000):  # Prevent infinite loops
            reduced = current.beta_reduce()
            if reduced == current:
                return current
            current = reduced
        return current

    cpdef void walk(self, object f):
        """Walk the AST, calling f on each node."""
        raise NotImplementedError

    cpdef list subprograms(self):
        """Return all subprograms (including self)."""
        cdef list result = []
        _collect_subprograms(self, result)
        return result

    cpdef Program clone(self):
        """Deep copy the program."""
        raise NotImplementedError


# ============================================================================
# PRIMITIVE
# ============================================================================

cdef class Primitive(Program):
    """
    A built-in primitive operation.

    Attributes:
        name: Identifier for the primitive
        tp: The type of the primitive
        value: The implementation (a Python callable or value)
    """
    cdef readonly str name
    cdef readonly object tp  # Type
    cdef readonly object value
    cdef int _hash

    def __init__(self, str name, object tp, object value):
        self.name = name
        self.tp = tp
        self.value = value
        self._hash = hash(('primitive', name))

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bint:
        if not isinstance(other, Primitive):
            return False
        return self.name == (<Primitive>other).name

    def __hash__(self) -> int:
        return self._hash

    cpdef object evaluate(self, list env):
        return self.value

    cpdef object infer_type(self, object ctx, list env):
        # Instantiate the type (fresh type variables for polymorphism)
        return ctx.instantiate(self.tp)

    cpdef int size(self):
        return 1

    cpdef int depth(self):
        return 1

    cpdef set free_indices(self, int depth_level=0):
        return set()

    cpdef Program shift(self, int amount, int cutoff=0):
        return self

    cpdef Program substitute(self, int index, Program replacement):
        return self

    cpdef void walk(self, object f):
        f(self)

    cpdef Program clone(self):
        return self  # Primitives are immutable


# ============================================================================
# INDEX (de Bruijn variable)
# ============================================================================

cdef class Index(Program):
    """
    A de Bruijn index variable.

    $0 refers to the most recently bound variable,
    $1 to the one before that, etc.
    """
    cdef readonly int i
    cdef int _hash

    def __init__(self, int i):
        self.i = i
        self._hash = hash(('index', i))

    def __str__(self) -> str:
        return f"${self.i}"

    def __eq__(self, other) -> bint:
        if not isinstance(other, Index):
            return False
        return self.i == (<Index>other).i

    def __hash__(self) -> int:
        return self._hash

    cpdef object evaluate(self, list env):
        cdef int env_len = len(env)
        if self.i >= env_len:
            raise RuntimeError(f"Index ${self.i} not in environment of size {env_len}")
        return env[self.i]

    cpdef object infer_type(self, object ctx, list env):
        cdef int env_len = len(env)
        if self.i >= env_len:
            raise UnificationError(f"Index ${self.i} not in type environment")
        return env[self.i]

    cpdef int size(self):
        return 1

    cpdef int depth(self):
        return 1

    cpdef set free_indices(self, int depth_level=0):
        if self.i >= depth_level:
            return {self.i - depth_level}
        return set()

    cpdef Program shift(self, int amount, int cutoff=0):
        if self.i >= cutoff:
            return Index(self.i + amount)
        return self

    cpdef Program substitute(self, int index, Program replacement):
        if self.i == index:
            return replacement
        return self

    cpdef void walk(self, object f):
        f(self)

    cpdef Program clone(self):
        return self


# ============================================================================
# APPLICATION
# ============================================================================

cdef class Application(Program):
    """
    Function application: (f x)

    Represents applying function f to argument x.
    """
    cdef readonly Program f
    cdef readonly Program x
    cdef int _hash

    def __init__(self, Program f, Program x):
        self.f = f
        self.x = x
        self._hash = hash(('app', hash(f), hash(x)))

    def __str__(self) -> str:
        cdef str f_str = str(self.f)
        cdef str x_str = str(self.x)

        # Parenthesize arguments that are applications
        if isinstance(self.x, Application):
            x_str = f"({x_str})"

        # Parenthesize lambdas
        if isinstance(self.f, Abstraction):
            f_str = f"({f_str})"
        if isinstance(self.x, Abstraction):
            x_str = f"({x_str})"

        return f"{f_str} {x_str}"

    def __eq__(self, other) -> bint:
        if not isinstance(other, Application):
            return False
        cdef Application o = <Application>other
        return self.f == o.f and self.x == o.x

    def __hash__(self) -> int:
        return self._hash

    cpdef object evaluate(self, list env):
        cdef object func = self.f.evaluate(env)
        cdef object arg = self.x.evaluate(env)

        if callable(func):
            return func(arg)
        else:
            raise RuntimeError(f"Cannot apply non-function: {func}")

    cpdef object infer_type(self, object ctx, list env):
        f_type = self.f.infer_type(ctx, env)
        x_type = self.x.infer_type(ctx, env)

        # f must be an arrow type
        ret_type = ctx.fresh_type_variable()
        expected_f_type = Arrow(x_type, ret_type)

        ctx.unify(f_type, expected_f_type)
        return ctx.apply(ret_type)

    cpdef int size(self):
        return 1 + self.f.size() + self.x.size()

    cpdef int depth(self):
        cdef int f_depth = self.f.depth()
        cdef int x_depth = self.x.depth()
        return 1 + (f_depth if f_depth > x_depth else x_depth)

    cpdef set free_indices(self, int depth_level=0):
        return self.f.free_indices(depth_level) | self.x.free_indices(depth_level)

    cpdef Program shift(self, int amount, int cutoff=0):
        return Application(
            self.f.shift(amount, cutoff),
            self.x.shift(amount, cutoff)
        )

    cpdef Program substitute(self, int index, Program replacement):
        return Application(
            self.f.substitute(index, replacement),
            self.x.substitute(index, replacement)
        )

    cpdef Program beta_reduce(self):
        cdef Abstraction abs_f
        cdef Program shifted_x
        cdef Program result
        cdef Program f_reduced
        cdef Program x_reduced

        # If f is a lambda, perform beta reduction
        if isinstance(self.f, Abstraction):
            abs_f = <Abstraction>self.f
            # (λ. body) x  -->  body[0 := x]
            shifted_x = self.x.shift(1, 0)  # Prepare x for substitution
            result = abs_f.body.substitute(0, shifted_x)
            return result.shift(-1, 0)  # Adjust indices back

        # Otherwise, try to reduce subterms
        f_reduced = self.f.beta_reduce()
        if f_reduced != self.f:
            return Application(f_reduced, self.x)

        x_reduced = self.x.beta_reduce()
        if x_reduced != self.x:
            return Application(self.f, x_reduced)

        return self

    cpdef void walk(self, object func):
        func(self)
        self.f.walk(func)
        self.x.walk(func)

    cpdef Program clone(self):
        return Application(self.f.clone(), self.x.clone())


# ============================================================================
# ABSTRACTION
# ============================================================================

cdef class Abstraction(Program):
    """
    Lambda abstraction: λ. body

    Binds a new variable in the body.
    The bound variable is accessed as $0 in the body.
    """
    cdef readonly Program body
    cdef int _hash

    def __init__(self, Program body):
        self.body = body
        self._hash = hash(('abs', hash(body)))

    def __str__(self) -> str:
        return f"(λ {self.body})"

    def __eq__(self, other) -> bint:
        if not isinstance(other, Abstraction):
            return False
        return self.body == (<Abstraction>other).body

    def __hash__(self) -> int:
        return self._hash

    cpdef object evaluate(self, list env):
        # Return a callable that captures the environment
        # Use _Closure class to avoid closure-in-cpdef issue
        return _Closure(self.body, env)

    cpdef object infer_type(self, object ctx, list env):
        arg_type = ctx.fresh_type_variable()
        cdef list new_env = [arg_type] + env
        ret_type = self.body.infer_type(ctx, new_env)
        return Arrow(ctx.apply(arg_type), ctx.apply(ret_type))

    cpdef int size(self):
        return 1 + self.body.size()

    cpdef int depth(self):
        return 1 + self.body.depth()

    cpdef set free_indices(self, int depth_level=0):
        return self.body.free_indices(depth_level + 1)

    cpdef Program shift(self, int amount, int cutoff=0):
        return Abstraction(self.body.shift(amount, cutoff + 1))

    cpdef Program substitute(self, int index, Program replacement):
        cdef Program shifted_replacement = replacement.shift(1, 0)
        return Abstraction(self.body.substitute(index + 1, shifted_replacement))

    cpdef void walk(self, object func):
        func(self)
        self.body.walk(func)

    cpdef Program clone(self):
        return Abstraction(self.body.clone())


# ============================================================================
# INVENTED (learned abstraction)
# ============================================================================

cdef class Invented(Program):
    """
    A learned abstraction discovered during compression.

    Invented primitives are lambda expressions that have been
    factored out as reusable library functions.
    """
    cdef readonly Program body
    cdef readonly str name
    cdef int _hash

    def __init__(self, Program body, str name=None):
        self.body = body
        self.name = name
        self._hash = hash(('invented', hash(body)))

    def __str__(self) -> str:
        if self.name:
            return f"#{self.name}"
        return f"#({self.body})"

    def __eq__(self, other) -> bint:
        if not isinstance(other, Invented):
            return False
        return self.body == (<Invented>other).body

    def __hash__(self) -> int:
        return self._hash

    cpdef object evaluate(self, list env):
        # Evaluate the body in the current environment
        return self.body.evaluate(env)

    cpdef object infer_type(self, object ctx, list env):
        return self.body.infer_type(ctx, env)

    cpdef int size(self):
        # When used, an invented counts as 1 (it's a library reference)
        return 1

    cpdef int depth(self):
        return 1

    cpdef set free_indices(self, int depth_level=0):
        return self.body.free_indices(depth_level)

    cpdef Program shift(self, int amount, int cutoff=0):
        return Invented(self.body.shift(amount, cutoff), self.name)

    cpdef Program substitute(self, int index, Program replacement):
        return Invented(self.body.substitute(index, replacement), self.name)

    cpdef void walk(self, object func):
        func(self)
        self.body.walk(func)

    cpdef Program clone(self):
        return Invented(self.body.clone(), self.name)

    cpdef int inner_size(self):
        """The actual size of the body (for compression analysis)."""
        return self.body.size()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_args(Program func, *args):
    """Helper: apply multiple arguments to a function."""
    cdef Program result = func
    cdef Program arg
    for arg in args:
        result = Application(result, <Program>arg)
    return result


cpdef Program multi_lambda(int n, Program body):
    """Create n nested lambda abstractions."""
    cdef Program result = body
    cdef int i
    for i in range(n):
        result = Abstraction(result)
    return result


cpdef int program_size(Program p):
    """Alias for p.size()."""
    return p.size()


cpdef int program_depth(Program p):
    """Alias for p.depth()."""
    return p.depth()


# ============================================================================
# PROGRAM PARSING (simple parser for debugging)
# ============================================================================

def parse_program(str s, dict primitives):
    """
    Simple parser for programs.

    Syntax:
        $n          -> Index(n)
        name        -> Primitive lookup
        (λ body)    -> Abstraction
        (f x)       -> Application
    """
    cdef int depth_count = 0
    cdef int split = -1
    cdef int i
    cdef str c
    cdef str inner
    cdef str body_str
    cdef str f_str
    cdef str x_str

    s = s.strip()

    # Index
    if s.startswith('$'):
        return Index(int(s[1:]))

    # Primitive
    if s in primitives:
        return primitives[s]

    # Parenthesized expression
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1].strip()

        # Lambda
        if inner.startswith('λ'):
            body_str = inner[1:].strip()
            return Abstraction(parse_program(body_str, primitives))

        # Application - find the split point
        depth_count = 0
        split = -1
        for i, c in enumerate(inner):
            if c == '(':
                depth_count += 1
            elif c == ')':
                depth_count -= 1
            elif c == ' ' and depth_count == 0:
                split = i
                break

        if split > 0:
            f_str = inner[:split]
            x_str = inner[split+1:]
            return Application(
                parse_program(f_str, primitives),
                parse_program(x_str, primitives)
            )

    raise ValueError(f"Cannot parse: {s}")


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Program Representation Tests ===\n")

    from .type_system_cy import INT, BOOL

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

    print("\n=== Program Representation OK ===")
