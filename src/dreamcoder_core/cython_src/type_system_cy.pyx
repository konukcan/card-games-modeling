# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cythonized Type System for Card Game DSL

Optimized version of type_system.py with:
- cdef classes for speed
- Typed attributes and local variables
- Disabled bounds checking for performance
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from cpython.object cimport PyObject

# ============================================================================
# TYPE BASE CLASS (Extension Type)
# ============================================================================

cdef class Type:
    """Abstract base class for types - Cython extension type."""

    def __str__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other) -> bint:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    cpdef set free_type_variables(self):
        """Return set of free type variable indices."""
        raise NotImplementedError

    cpdef Type apply_substitution(self, dict subst):
        """Apply a type substitution."""
        raise NotImplementedError

    cpdef bint occurs_in(self, int var_id):
        """Check if type variable occurs in this type."""
        return var_id in self.free_type_variables()

    cpdef bint is_arrow(self):
        return isinstance(self, Arrow)

    @property
    def returns(self):
        """Return the final return type (for arrow types, unwrap)."""
        if isinstance(self, Arrow):
            return (<Arrow>self).ret.returns
        return self

    @property
    def arguments(self):
        """Return the argument types (for arrow types)."""
        cdef list args = []
        cdef Type current = self
        while isinstance(current, Arrow):
            args.append((<Arrow>current).arg)
            current = (<Arrow>current).ret
        return args


# ============================================================================
# BASE TYPE
# ============================================================================

cdef class BaseType(Type):
    """A base type like bool, int, card."""
    cdef readonly str name
    cdef int _hash

    def __init__(self, str name):
        self.name = name
        self._hash = hash(('base', name))

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bint:
        if not isinstance(other, BaseType):
            return False
        return self.name == (<BaseType>other).name

    def __hash__(self) -> int:
        return self._hash

    cpdef set free_type_variables(self):
        return set()

    cpdef Type apply_substitution(self, dict subst):
        return self


# ============================================================================
# TYPE VARIABLE
# ============================================================================

cdef class TypeVariable(Type):
    """A type variable for polymorphism (e.g., 'a, 'b)."""
    cdef readonly int id
    cdef int _hash

    def __init__(self, int id):
        self.id = id
        self._hash = hash(('var', id))

    def __str__(self) -> str:
        if self.id < 26:
            return f"'{chr(ord('a') + self.id)}"
        return f"'t{self.id}"

    def __eq__(self, other) -> bint:
        if not isinstance(other, TypeVariable):
            return False
        return self.id == (<TypeVariable>other).id

    def __hash__(self) -> int:
        return self._hash

    cpdef set free_type_variables(self):
        return {self.id}

    cpdef Type apply_substitution(self, dict subst):
        # Follow the substitution chain with cycle detection
        cdef set visited = set()
        cdef Type current = self
        cdef TypeVariable tv

        while isinstance(current, TypeVariable):
            tv = <TypeVariable>current
            if tv.id not in subst:
                break
            if tv.id in visited:
                return current  # Break cycle
            visited.add(tv.id)
            current = subst[tv.id]

        if isinstance(current, TypeVariable):
            return current
        return current.apply_substitution(subst)


# ============================================================================
# ARROW TYPE
# ============================================================================

cdef class Arrow(Type):
    """Function type: arg -> ret."""
    cdef readonly Type arg
    cdef readonly Type ret
    cdef int _hash

    def __init__(self, Type arg, Type ret):
        self.arg = arg
        self.ret = ret
        self._hash = hash(('arrow', hash(arg), hash(ret)))

    def __str__(self) -> str:
        cdef str arg_str
        if isinstance(self.arg, Arrow):
            arg_str = f"({self.arg})"
        else:
            arg_str = str(self.arg)
        return f"{arg_str} -> {self.ret}"

    def __eq__(self, other) -> bint:
        if not isinstance(other, Arrow):
            return False
        cdef Arrow o = <Arrow>other
        return self.arg == o.arg and self.ret == o.ret

    def __hash__(self) -> int:
        return self._hash

    cpdef set free_type_variables(self):
        return self.arg.free_type_variables() | self.ret.free_type_variables()

    cpdef Type apply_substitution(self, dict subst):
        return Arrow(
            self.arg.apply_substitution(subst),
            self.ret.apply_substitution(subst)
        )


# ============================================================================
# LIST TYPE
# ============================================================================

cdef class ListType(Type):
    """List type: list(element_type)."""
    cdef readonly Type element
    cdef int _hash

    def __init__(self, Type element):
        self.element = element
        self._hash = hash(('list', hash(element)))

    def __str__(self) -> str:
        return f"list({self.element})"

    def __eq__(self, other) -> bint:
        if not isinstance(other, ListType):
            return False
        return self.element == (<ListType>other).element

    def __hash__(self) -> int:
        return self._hash

    cpdef set free_type_variables(self):
        return self.element.free_type_variables()

    cpdef Type apply_substitution(self, dict subst):
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
HAND = ListType(CARD)
LIST_INT = ListType(INT)
LIST_BOOL = ListType(BOOL)
LIST_SUIT = ListType(SUIT)
LIST_RANK = ListType(RANK)


def arrow(*types):
    """Convenience function: arrow(a, b, c) = a -> b -> c."""
    cdef int n = len(types)
    if n < 2:
        raise ValueError("Arrow needs at least 2 types")

    cdef Type result = <Type>types[n - 1]
    cdef int i
    for i in range(n - 2, -1, -1):
        result = Arrow(<Type>types[i], result)
    return result


# ============================================================================
# TYPE UNIFICATION
# ============================================================================

class UnificationError(Exception):
    """Raised when types cannot be unified."""
    pass


cdef class TypeContext:
    """Manages type variables and unification during type inference."""
    cdef int _next_var
    cdef dict _substitution

    def __init__(self):
        self._next_var = 0
        self._substitution = {}

    cpdef TypeVariable fresh_type_variable(self):
        """Generate a fresh type variable."""
        cdef TypeVariable var = TypeVariable(self._next_var)
        self._next_var += 1
        return var

    cpdef Type instantiate(self, Type t):
        """Replace all type variables with fresh ones (for polymorphism)."""
        cdef set free_vars = t.free_type_variables()
        if not free_vars:
            return t

        cdef dict fresh_subst = {}
        cdef int v
        for v in free_vars:
            fresh_subst[v] = self.fresh_type_variable()
        return t.apply_substitution(fresh_subst)

    cpdef void unify(self, Type t1, Type t2) except *:
        """Unify two types, updating the substitution."""
        t1 = self.apply(t1)
        t2 = self.apply(t2)

        if t1 == t2:
            return

        cdef TypeVariable tv1, tv2
        cdef Arrow a1, a2
        cdef ListType l1, l2
        cdef BaseType b1, b2

        # Type variable cases
        if isinstance(t1, TypeVariable):
            tv1 = <TypeVariable>t1
            if tv1.id in t2.free_type_variables():
                raise UnificationError(f"Occurs check failed: {t1} in {t2}")
            self._substitution[tv1.id] = t2
            return

        if isinstance(t2, TypeVariable):
            tv2 = <TypeVariable>t2
            if tv2.id in t1.free_type_variables():
                raise UnificationError(f"Occurs check failed: {t2} in {t1}")
            self._substitution[tv2.id] = t1
            return

        # Arrow types
        if isinstance(t1, Arrow) and isinstance(t2, Arrow):
            a1 = <Arrow>t1
            a2 = <Arrow>t2
            self.unify(a1.arg, a2.arg)
            self.unify(a1.ret, a2.ret)
            return

        # List types
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            l1 = <ListType>t1
            l2 = <ListType>t2
            self.unify(l1.element, l2.element)
            return

        # Base types
        if isinstance(t1, BaseType) and isinstance(t2, BaseType):
            b1 = <BaseType>t1
            b2 = <BaseType>t2
            if b1.name == b2.name:
                return

        raise UnificationError(f"Cannot unify {t1} with {t2}")

    cpdef Type apply(self, Type t):
        """Apply the current substitution to a type, resolving chains."""
        return self._apply_recursive(t, set())

    cdef Type _apply_recursive(self, Type t, set visited):
        """Apply substitution with cycle detection."""
        cdef TypeVariable tv
        cdef Arrow arr
        cdef ListType lst

        if isinstance(t, TypeVariable):
            tv = <TypeVariable>t
            if tv.id in visited:
                return t  # Break cycle
            if tv.id in self._substitution:
                return self._apply_recursive(self._substitution[tv.id], visited | {tv.id})
            return t
        elif isinstance(t, Arrow):
            arr = <Arrow>t
            return Arrow(
                self._apply_recursive(arr.arg, visited),
                self._apply_recursive(arr.ret, visited)
            )
        elif isinstance(t, ListType):
            lst = <ListType>t
            return ListType(self._apply_recursive(lst.element, visited))
        else:
            return t


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

cpdef Type canonical_type(Type t):
    """Normalize type variables to a canonical form."""
    cdef list free_vars = sorted(t.free_type_variables())
    cdef dict subst = {}
    cdef int i, v
    for i, v in enumerate(free_vars):
        subst[v] = TypeVariable(i)
    return t.apply_substitution(subst)


cpdef int type_arity(Type t):
    """Return the number of arguments a function type takes."""
    cdef int count = 0
    while isinstance(t, Arrow):
        count += 1
        t = (<Arrow>t).ret
    return count
