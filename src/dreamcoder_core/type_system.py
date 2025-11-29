"""
Type System for Card Game DSL

Implements a simple type system with:
- Base types: bool, int, card, list(t)
- Arrow types: t1 -> t2
- Type variables for polymorphism

Following DreamCoder's type system but specialized for cards.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
import itertools


class Type(ABC):
    """Abstract base class for types."""

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def free_type_variables(self) -> Set[int]:
        """Return set of free type variable indices."""
        pass

    @abstractmethod
    def apply_substitution(self, subst: Dict[int, 'Type']) -> 'Type':
        """Apply a type substitution."""
        pass

    def occurs_in(self, var_id: int) -> bool:
        """Check if type variable occurs in this type."""
        return var_id in self.free_type_variables()

    def is_arrow(self) -> bool:
        return isinstance(self, Arrow)

    @property
    def returns(self) -> 'Type':
        """Return the final return type (for arrow types, unwrap)."""
        if isinstance(self, Arrow):
            return self.ret.returns
        return self

    @property
    def arguments(self) -> List['Type']:
        """Return the argument types (for arrow types)."""
        if isinstance(self, Arrow):
            return [self.arg] + self.ret.arguments
        return []


@dataclass(frozen=True)
class BaseType(Type):
    """A base type like bool, int, card."""
    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, BaseType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(('base', self.name))

    def free_type_variables(self) -> Set[int]:
        return set()

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        return self


@dataclass(frozen=True)
class TypeVariable(Type):
    """A type variable for polymorphism (e.g., 'a, 'b)."""
    id: int

    def __str__(self) -> str:
        # Use letters for readability
        if self.id < 26:
            return f"'{chr(ord('a') + self.id)}"
        return f"'t{self.id}"

    def __eq__(self, other) -> bool:
        return isinstance(other, TypeVariable) and self.id == other.id

    def __hash__(self) -> int:
        return hash(('var', self.id))

    def free_type_variables(self) -> Set[int]:
        return {self.id}

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        # Follow the substitution chain, but with cycle detection
        visited = set()
        current = self
        while isinstance(current, TypeVariable) and current.id in subst:
            if current.id in visited:
                return current  # Break cycle
            visited.add(current.id)
            current = subst[current.id]
        if isinstance(current, TypeVariable):
            return current
        return current.apply_substitution(subst)


@dataclass(frozen=True)
class Arrow(Type):
    """Function type: arg -> ret."""
    arg: Type
    ret: Type

    def __str__(self) -> str:
        arg_str = f"({self.arg})" if isinstance(self.arg, Arrow) else str(self.arg)
        return f"{arg_str} -> {self.ret}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Arrow) and self.arg == other.arg and self.ret == other.ret

    def __hash__(self) -> int:
        return hash(('arrow', self.arg, self.ret))

    def free_type_variables(self) -> Set[int]:
        return self.arg.free_type_variables() | self.ret.free_type_variables()

    def apply_substitution(self, subst: Dict[int, Type]) -> Type:
        return Arrow(
            self.arg.apply_substitution(subst),
            self.ret.apply_substitution(subst)
        )


@dataclass(frozen=True)
class ListType(Type):
    """List type: list(element_type)."""
    element: Type

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

# Base types
BOOL = BaseType('bool')
INT = BaseType('int')
CARD = BaseType('card')
SUIT = BaseType('suit')
RANK = BaseType('rank')

# Common composite types
HAND = ListType(CARD)  # A hand is a list of cards
LIST_INT = ListType(INT)
LIST_BOOL = ListType(BOOL)
LIST_SUIT = ListType(SUIT)
LIST_RANK = ListType(RANK)


def arrow(*types: Type) -> Type:
    """Convenience function: arrow(a, b, c) = a -> b -> c."""
    if len(types) < 2:
        raise ValueError("Arrow needs at least 2 types")
    result = types[-1]
    for t in reversed(types[:-1]):
        result = Arrow(t, result)
    return result


# ============================================================================
# TYPE UNIFICATION
# ============================================================================

class UnificationError(Exception):
    """Raised when types cannot be unified."""
    pass


class TypeContext:
    """Manages type variables and unification during type inference."""

    def __init__(self):
        self._next_var = 0
        self._substitution: Dict[int, Type] = {}

    def fresh_type_variable(self) -> TypeVariable:
        """Generate a fresh type variable."""
        var = TypeVariable(self._next_var)
        self._next_var += 1
        return var

    def instantiate(self, t: Type) -> Type:
        """Replace all type variables with fresh ones (for polymorphism)."""
        free_vars = t.free_type_variables()
        if not free_vars:
            return t

        # Create fresh variables for each free variable
        fresh_subst = {v: self.fresh_type_variable() for v in free_vars}
        return t.apply_substitution(fresh_subst)

    def unify(self, t1: Type, t2: Type) -> None:
        """
        Unify two types, updating the substitution.

        Raises UnificationError if types are incompatible.
        """
        # Apply current substitution first
        t1 = self.apply(t1)
        t2 = self.apply(t2)

        if t1 == t2:
            return

        # Type variable cases
        if isinstance(t1, TypeVariable):
            # Occurs check: t1 must not occur in t2
            if t1.id in t2.free_type_variables():
                raise UnificationError(f"Occurs check failed: {t1} in {t2}")
            self._substitution[t1.id] = t2
            return

        if isinstance(t2, TypeVariable):
            # Occurs check: t2 must not occur in t1
            if t2.id in t1.free_type_variables():
                raise UnificationError(f"Occurs check failed: {t2} in {t1}")
            self._substitution[t2.id] = t1
            return

        # Arrow types
        if isinstance(t1, Arrow) and isinstance(t2, Arrow):
            self.unify(t1.arg, t2.arg)
            self.unify(t1.ret, t2.ret)
            return

        # List types
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            self.unify(t1.element, t2.element)
            return

        # Base types
        if isinstance(t1, BaseType) and isinstance(t2, BaseType):
            if t1.name == t2.name:
                return

        raise UnificationError(f"Cannot unify {t1} with {t2}")

    def apply(self, t: Type) -> Type:
        """Apply the current substitution to a type, resolving chains."""
        return self._apply_recursive(t, set())

    def _apply_recursive(self, t: Type, visited: Set[int]) -> Type:
        """Apply substitution with cycle detection."""
        if isinstance(t, TypeVariable):
            if t.id in visited:
                return t  # Break cycle
            if t.id in self._substitution:
                visited = visited | {t.id}
                return self._apply_recursive(self._substitution[t.id], visited)
            return t
        elif isinstance(t, Arrow):
            return Arrow(
                self._apply_recursive(t.arg, visited),
                self._apply_recursive(t.ret, visited)
            )
        elif isinstance(t, ListType):
            return ListType(self._apply_recursive(t.element, visited))
        else:
            return t


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def canonical_type(t: Type) -> Type:
    """
    Normalize type variables to a canonical form.

    E.g., 'b -> 'c becomes 'a -> 'b
    """
    free_vars = sorted(t.free_type_variables())
    subst = {v: TypeVariable(i) for i, v in enumerate(free_vars)}
    return t.apply_substitution(subst)


def type_arity(t: Type) -> int:
    """Return the number of arguments a function type takes."""
    if isinstance(t, Arrow):
        return 1 + type_arity(t.ret)
    return 0


if __name__ == "__main__":
    # Test the type system
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

    # Higher-order type
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
