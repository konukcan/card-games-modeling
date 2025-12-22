"""
Type System for Card Game DSL
=============================

PURPOSE:
    This module provides static typing for the Domain-Specific Language (DSL).
    It ensures programs are type-correct before execution and enables the
    enumeration engine to only generate well-typed programs.

COMPARISON WITH ORIGINAL DREAMCODER (Ellis et al.):
    - Original uses a single `TypeConstructor` class for ALL types (arrows, lists, base)
    - We use separate classes (BaseType, Arrow, ListType) for explicit pattern matching
    - Original has generic types (tint, tlist); we have card-specific (CARD, SUIT, RANK)
    - Original offers both mutable and immutable contexts; we use mutable only

KEY CONCEPTS:

    1. BASE TYPES: Ground types with no parameters
       Examples: bool, int, card, suit, rank

    2. ARROW TYPES: Function types (curried, right-associative)
       int -> int -> bool means "takes int, returns (int -> bool)"
       which means "takes two ints, returns bool"

    3. TYPE VARIABLES: For polymorphism (generic functions)
       'a -> 'a means "takes any type, returns same type" (identity)

    4. UNIFICATION: Finding if two types can be made equal
       Unifying 'a with int binds 'a := int
       Unifying int with bool FAILS (incompatible)

ARCHITECTURE:

    Type (Abstract Base)
    ├── BaseType(name)        # Ground types: bool, int, card, suit, rank
    ├── TypeVariable(id)      # Polymorphic variables: 'a, 'b, 'c...
    ├── Arrow(arg, ret)       # Function types: arg -> ret
    └── ListType(element)     # List types: list(element)

    TypeContext               # Manages unification and fresh variables
    └── _substitution: Dict   # Maps variable IDs to their bound types
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
import itertools


# ============================================================================
# TYPE BASE CLASS
# ============================================================================

class Type(ABC):
    """
    Abstract base class for all types.

    Every type must implement:
    - __str__: Human-readable representation (e.g., "int -> bool")
    - __eq__, __hash__: For using types in sets and as dict keys
    - free_type_variables: Which type variables are unbound?
    - apply_substitution: Replace type variables with their bindings

    PYTHON SYNTAX NOTES:
    - ABC = Abstract Base Class (from abc module)
    - @abstractmethod = subclasses MUST implement this method
    - -> after method means return type annotation
    - 'Type' in quotes = forward reference (Type isn't fully defined yet)
    """

    @abstractmethod
    def __str__(self) -> str:
        """Return human-readable string like 'int -> bool'."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check structural equality. Used by == operator."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """
        Return hash for use in sets/dicts.

        IMPORTANT: If two objects are equal (__eq__ returns True),
        they MUST have the same hash. This is a Python contract.
        """
        pass

    @abstractmethod
    def free_type_variables(self) -> Set[int]:
        """
        Return set of free (unbound) type variable IDs.

        Examples:
            int.free_type_variables() = {}           # No variables
            'a.free_type_variables() = {0}           # Variable 'a has id=0
            ('a -> int).free_type_variables() = {0}  # 'a is free
        """
        pass

    @abstractmethod
    def apply_substitution(self, subst: Dict[int, 'Type']) -> 'Type':
        """
        Replace type variables according to substitution mapping.

        Example:
            subst = {0: INT}  # 'a (id=0) maps to int
            ('a -> 'a).apply_substitution(subst) = (int -> int)

        PYTHON SYNTAX: Dict[int, 'Type'] means dictionary mapping ints to Types
        """
        pass

    def occurs_in(self, var_id: int) -> bool:
        """
        Check if type variable with given ID appears in this type.

        Used for the "occurs check" in unification to prevent infinite types.
        Example: Can't unify 'a with list('a) because 'a occurs in list('a)
        """
        return var_id in self.free_type_variables()

    def is_arrow(self) -> bool:
        """Check if this is a function type."""
        return isinstance(self, Arrow)

    @property
    def returns(self) -> 'Type':
        """
        Return the final return type (unwrapping nested arrows).

        PYTHON SYNTAX: @property makes this callable without parentheses
                       t.returns instead of t.returns()

        Examples:
            int.returns = int
            (int -> bool).returns = bool
            (int -> int -> bool).returns = bool  # Unwraps fully
        """
        if isinstance(self, Arrow):
            return self.ret.returns  # Recurse into return type
        return self

    @property
    def arguments(self) -> List['Type']:
        """
        Return list of argument types (for arrow types).

        Examples:
            int.arguments = []
            (int -> bool).arguments = [int]
            (int -> card -> bool).arguments = [int, card]
        """
        if isinstance(self, Arrow):
            return [self.arg] + self.ret.arguments  # Collect recursively
        return []


# ============================================================================
# BASE TYPE
# ============================================================================

@dataclass(frozen=True)
class BaseType(Type):
    """
    A ground type with no parameters.

    PYTHON SYNTAX:
    - @dataclass: Auto-generates __init__, __repr__ from fields
    - frozen=True: Makes instances immutable (hashable)
    - Fields listed below become constructor arguments

    Examples:
        BOOL = BaseType('bool')
        CARD = BaseType('card')

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Original: TypeConstructor('int', []) with empty argument list
    - Ours: Dedicated BaseType class, cleaner pattern matching
    """
    name: str  # The type name, e.g., 'bool', 'int', 'card'

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        # Check type first, then compare names
        return isinstance(other, BaseType) and self.name == other.name

    def __hash__(self) -> int:
        # Include 'base' to distinguish from other type kinds
        return hash(('base', self.name))

    def free_type_variables(self) -> Set[int]:
        # Base types have no type variables
        return set()

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        # Nothing to substitute in a base type
        return self


# ============================================================================
# TYPE VARIABLE
# ============================================================================

@dataclass(frozen=True)
class TypeVariable(Type):
    """
    A type variable for polymorphism (generics).

    Type variables are placeholders that can be unified with any type.
    They enable polymorphic functions like:
        map : ('a -> 'b) -> list('a) -> list('b)

    Each variable has a unique integer ID. When printed:
        id=0 -> 'a
        id=1 -> 'b
        id=25 -> 'z
        id=26 -> 't26

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Same concept, similar implementation
    - Original uses TypeVariable class too
    """
    id: int  # Unique identifier for this variable

    def __str__(self) -> str:
        # Use letters for readability: 'a, 'b, 'c, ...
        if self.id < 26:
            return f"'{chr(ord('a') + self.id)}"  # chr(97) = 'a'
        return f"'t{self.id}"  # Fall back to numbered form

    def __eq__(self, other) -> bool:
        return isinstance(other, TypeVariable) and self.id == other.id

    def __hash__(self) -> int:
        return hash(('var', self.id))

    def free_type_variables(self) -> Set[int]:
        # A type variable IS a free variable (itself)
        return {self.id}

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        """
        Follow the substitution chain with cycle detection.

        Example chain: 'a -> 'b -> int
            subst = {0: TypeVariable(1), 1: INT}
            TypeVariable(0).apply_substitution(subst) = INT

        Cycle detection prevents infinite loops if substitution is malformed.
        """
        visited = set()  # Track which variables we've seen
        current = self

        # Follow chain: if 'a -> 'b, and 'b -> int, get int
        while isinstance(current, TypeVariable) and current.id in subst:
            if current.id in visited:
                return current  # Cycle detected, break out
            visited.add(current.id)
            current = subst[current.id]

        # If we ended on a variable, return it
        if isinstance(current, TypeVariable):
            return current

        # Otherwise, recursively apply to the result (it might contain more variables)
        return current.apply_substitution(subst)


# ============================================================================
# ARROW TYPE (Function Type)
# ============================================================================

@dataclass(frozen=True)
class Arrow(Type):
    """
    Function type: arg -> ret

    Arrow types are RIGHT-ASSOCIATIVE and CURRIED:
        int -> int -> bool  means  int -> (int -> bool)
        This represents a function taking two ints, returning bool.

    In curried form, multi-argument functions are chains of single-argument functions:
        add : int -> int -> int
        add(3) : int -> int        # Partial application
        add(3)(4) : int            # Full application = 7

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Original: TypeConstructor('->', [arg, ret])
    - Ours: Dedicated Arrow class with arg and ret fields
    """
    arg: Type  # Argument type (left side of arrow)
    ret: Type  # Return type (right side of arrow)

    def __str__(self) -> str:
        # Parenthesize arg if it's also an arrow (for clarity)
        # (int -> int) -> bool  vs  int -> int -> bool
        arg_str = f"({self.arg})" if isinstance(self.arg, Arrow) else str(self.arg)
        return f"{arg_str} -> {self.ret}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Arrow) and self.arg == other.arg and self.ret == other.ret

    def __hash__(self) -> int:
        return hash(('arrow', self.arg, self.ret))

    def free_type_variables(self) -> Set[int]:
        # Union of free variables in arg and ret
        # PYTHON SYNTAX: | is set union operator
        return self.arg.free_type_variables() | self.ret.free_type_variables()

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        # Recursively apply to both components
        return Arrow(
            self.arg.apply_substitution(subst),
            self.ret.apply_substitution(subst)
        )


# ============================================================================
# LIST TYPE
# ============================================================================

@dataclass(frozen=True)
class ListType(Type):
    """
    Parameterized list type: list(element_type)

    Examples:
        list(int)  = ListType(INT)   # List of integers
        list(card) = ListType(CARD)  # List of cards = a hand

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Original: TypeConstructor('list', [element])
    - Ours: Dedicated ListType class
    """
    element: Type  # The type of elements in the list

    def __str__(self) -> str:
        return f"list({self.element})"

    def __eq__(self, other) -> bool:
        return isinstance(other, ListType) and self.element == other.element

    def __hash__(self) -> int:
        return hash(('list', self.element))

    def free_type_variables(self) -> Set[int]:
        return self.element.free_type_variables()

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        return ListType(self.element.apply_substitution(subst))


# ============================================================================
# STANDARD TYPES FOR CARD DOMAIN
# ============================================================================
# These are the concrete types used in our card game DSL.
# They are pre-instantiated for convenience.

# Base types - ground types specific to our domain
BOOL = BaseType('bool')   # Boolean: True or False
INT = BaseType('int')     # Integer: ..., -1, 0, 1, 2, ...
CARD = BaseType('card')   # A single playing card (suit + rank)
SUIT = BaseType('suit')   # Suit: Clubs, Diamonds, Hearts, Spades
RANK = BaseType('rank')   # Rank: 2, 3, 4, ..., 10, J, Q, K, A

# Common composite types
HAND = ListType(CARD)       # A hand is a list of cards
LIST_INT = ListType(INT)    # List of integers
LIST_BOOL = ListType(BOOL)  # List of booleans
LIST_SUIT = ListType(SUIT)  # List of suits
LIST_RANK = ListType(RANK)  # List of ranks


def arrow(*types: Type) -> Type:
    """
    Convenience function to create arrow types.

    USAGE:
        arrow(INT, BOOL)           = INT -> BOOL
        arrow(INT, INT, BOOL)      = INT -> INT -> BOOL
        arrow(CARD, SUIT)          = CARD -> SUIT
        arrow(HAND, BOOL)          = list(card) -> bool

    PYTHON SYNTAX: *types means "accept any number of arguments as tuple"

    IMPLEMENTATION:
        Builds right-associatively: arrow(A, B, C) = A -> (B -> C)
        Works backwards from the last type.
    """
    if len(types) < 2:
        raise ValueError("Arrow needs at least 2 types")

    # Start with the last type (the final return type)
    result = types[-1]

    # Work backwards, wrapping each in an Arrow
    # reversed(types[:-1]) = all but last, in reverse order
    for t in reversed(types[:-1]):
        result = Arrow(t, result)

    return result


# ============================================================================
# TYPE UNIFICATION
# ============================================================================

class UnificationError(Exception):
    """
    Raised when two types cannot be unified.

    Examples of unification failures:
    - unify(int, bool)  # Different base types
    - unify('a, list('a))  # Occurs check failure (infinite type)
    """
    pass


class TypeContext:
    """
    Manages type variables and unification during type inference.

    This is the workhorse of the type system. It:
    1. Generates fresh type variables (for instantiating polymorphic types)
    2. Maintains a substitution (mapping variables to their inferred types)
    3. Performs unification (making two types equal)

    USAGE EXAMPLE:
        ctx = TypeContext()
        a = ctx.fresh_type_variable()  # Creates 'a
        ctx.unify(a, INT)              # Now 'a = int
        ctx.apply(a)                   # Returns INT

    COMPARISON WITH ORIGINAL DREAMCODER:
    - Original has both Context (immutable) and MutableContext (mutable)
    - We use mutable only (simpler, sufficient for our needs)
    - Original stores substitution as list of pairs; we use dict
    """

    def __init__(self):
        self._next_var = 0                        # Counter for fresh variable IDs
        self._substitution: Dict[int, Type] = {}  # var_id -> bound_type

    def fresh_type_variable(self) -> TypeVariable:
        """
        Generate a fresh type variable with unique ID.

        Each call returns a new variable: 'a, 'b, 'c, ...
        """
        var = TypeVariable(self._next_var)
        self._next_var += 1
        return var

    def instantiate(self, t: Type) -> Type:
        """
        Replace all type variables with fresh ones.

        This is used when using a polymorphic primitive, so each use
        gets its own type variables.

        Example:
            id_type = 'a -> 'a
            ctx.instantiate(id_type)  # Returns 'b -> 'b (fresh variables)
            ctx.instantiate(id_type)  # Returns 'c -> 'c (fresh again)
        """
        free_vars = t.free_type_variables()
        if not free_vars:
            return t  # No variables to replace

        # Create fresh variables for each free variable
        fresh_subst = {v: self.fresh_type_variable() for v in free_vars}
        return t.apply_substitution(fresh_subst)

    def unify(self, t1: Type, t2: Type) -> None:
        """
        Unify two types, updating the substitution.

        Unification finds a substitution that makes t1 and t2 equal.
        Raises UnificationError if types are incompatible.

        ALGORITHM (Robinson's Unification):
        1. Apply current substitution to both types
        2. If equal, done
        3. If one is a variable, bind it (with occurs check)
        4. If both are compound (Arrow, List), unify components
        5. Otherwise, fail

        THE OCCURS CHECK:
        Prevents infinite types. Can't unify 'a with list('a)
        because that would make 'a = list(list(list(...))) infinitely.
        """
        # Apply current substitution first (resolve already-bound variables)
        t1 = self.apply(t1)
        t2 = self.apply(t2)

        # Already equal? Done!
        if t1 == t2:
            return

        # --- Type variable cases ---

        if isinstance(t1, TypeVariable):
            # OCCURS CHECK: t1 must not occur in t2
            # Example: can't unify 'a with list('a)
            if t1.id in t2.free_type_variables():
                raise UnificationError(f"Occurs check failed: {t1} in {t2}")
            # Bind t1 to t2
            self._substitution[t1.id] = t2
            return

        if isinstance(t2, TypeVariable):
            # Symmetric case
            if t2.id in t1.free_type_variables():
                raise UnificationError(f"Occurs check failed: {t2} in {t1}")
            self._substitution[t2.id] = t1
            return

        # --- Compound type cases ---

        if isinstance(t1, Arrow) and isinstance(t2, Arrow):
            # Unify argument types, then return types
            self.unify(t1.arg, t2.arg)
            self.unify(t1.ret, t2.ret)
            return

        if isinstance(t1, ListType) and isinstance(t2, ListType):
            # Unify element types
            self.unify(t1.element, t2.element)
            return

        # --- Base type case ---

        if isinstance(t1, BaseType) and isinstance(t2, BaseType):
            if t1.name == t2.name:
                return
            # Different base types can't unify

        # --- Failure ---
        raise UnificationError(f"Cannot unify {t1} with {t2}")

    def apply(self, t: Type) -> Type:
        """
        Apply the current substitution to a type, resolving all bound variables.

        Example:
            If _substitution = {0: INT, 1: BOOL}
            apply('a -> 'b) = int -> bool
        """
        return self._apply_recursive(t, set())

    def _apply_recursive(self, t: Type, visited: Set[int]) -> Type:
        """
        Apply substitution with cycle detection.

        The visited set prevents infinite loops if substitution
        somehow contains cycles (defensive programming).
        """
        if isinstance(t, TypeVariable):
            if t.id in visited:
                return t  # Break cycle
            if t.id in self._substitution:
                # Follow the substitution, tracking visited variables
                visited = visited | {t.id}  # New set with t.id added
                return self._apply_recursive(self._substitution[t.id], visited)
            return t  # Unbound variable
        elif isinstance(t, Arrow):
            return Arrow(
                self._apply_recursive(t.arg, visited),
                self._apply_recursive(t.ret, visited)
            )
        elif isinstance(t, ListType):
            return ListType(self._apply_recursive(t.element, visited))
        else:
            return t  # BaseType, nothing to do


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def canonical_type(t: Type) -> Type:
    """
    Normalize type variables to a canonical form.

    This renames variables to start from 'a in order of appearance.
    Useful for comparing types structurally regardless of variable names.

    Examples:
        'b -> 'c  becomes  'a -> 'b
        'x -> 'x  becomes  'a -> 'a
    """
    free_vars = sorted(t.free_type_variables())
    subst = {v: TypeVariable(i) for i, v in enumerate(free_vars)}
    return t.apply_substitution(subst)


def type_arity(t: Type) -> int:
    """
    Return the number of arguments a function type takes.

    Examples:
        type_arity(int) = 0
        type_arity(int -> bool) = 1
        type_arity(int -> int -> bool) = 2
        type_arity(card -> suit) = 1
    """
    if isinstance(t, Arrow):
        return 1 + type_arity(t.ret)
    return 0


# ============================================================================
# TESTS (run with: python type_system.py)
# ============================================================================

if __name__ == "__main__":
    # PYTHON SYNTAX: This block only runs when file is executed directly,
    # not when imported as a module.

    print("=== Type System Tests ===\n")

    # Basic types
    print(f"BOOL: {BOOL}")
    print(f"INT: {INT}")
    print(f"CARD: {CARD}")
    print(f"HAND: {HAND}")

    # Arrow types
    card_to_int = arrow(CARD, INT)
    print(f"\ncard -> int: {card_to_int}")

    hand_to_bool = arrow(HAND, BOOL)
    print(f"hand -> bool: {hand_to_bool}")

    # Higher-order type (function that takes a function)
    map_type = arrow(arrow(CARD, INT), HAND, LIST_INT)
    print(f"\nmap type: {map_type}")
    print(f"  arguments: {map_type.arguments}")
    print(f"  returns: {map_type.returns}")

    # Type unification
    print("\n=== Unification Tests ===\n")

    ctx = TypeContext()
    a = ctx.fresh_type_variable()
    b = ctx.fresh_type_variable()

    # Unify 'a with int
    ctx.unify(a, INT)
    print(f"After unifying {a} with int: {ctx.apply(a)}")

    # Unify 'b -> 'b with int -> int
    ctx2 = TypeContext()
    c = ctx2.fresh_type_variable()
    t1 = arrow(c, c)
    t2 = arrow(INT, INT)
    ctx2.unify(t1, t2)
    print(f"Unified {t1} with {t2}: {ctx2.apply(t1)}")

    print("\n=== Type System OK ===")
