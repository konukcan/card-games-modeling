"""
New primitives for the grammar-comparison experiment.

Each primitive is a plain Python function designed to be more expressive
(higher arity / more general) than the current DSL primitives, allowing
shorter programs for many card-game rules.

These are defined in isolation — they do NOT modify the main DSL in src/.
"""

from collections import Counter
from typing import Any, Callable, List, TypeVar

T = TypeVar("T")
K = TypeVar("K")


def prim_slice(i: int, j: int, xs: List[T]) -> List[T]:
    """Return xs[i:j].

    A single primitive that subsumes several existing ones:
      - take(n, xs)       = prim_slice(0, n, xs)
      - drop(n, xs)       = prim_slice(n, len(xs), xs)
      - first_half(xs)    = prim_slice(0, len(xs)//2, xs)
      - second_half(xs)   = prim_slice(len(xs)//2, len(xs), xs)

    Args:
        i: Start index (inclusive).
        j: End index (exclusive).
        xs: The list to slice.

    Returns:
        The sub-list xs[i:j].
    """
    return xs[i:j]


def prim_shifted_match(k: int, pred: Callable[[Any, Any], bool], xs: List[T]) -> bool:
    """Check that pred(xs[i], xs[i+k]) holds for every valid index i.

    Generalises pairwise-adjacent checks (k=1) to any fixed offset.
    For example:
      - k=1: "each card matches its neighbour" (ascending run, same-suit pairs)
      - k=2: "every-other card matches" (ABAB patterns)

    Returns True vacuously when there are no valid (i, i+k) pairs
    (i.e. when len(xs) <= k).

    Args:
        k: The offset between the two elements being compared.
        pred: A binary predicate (a, b) -> bool.
        xs: The list to check.

    Returns:
        True iff pred(xs[i], xs[i+k]) for all 0 <= i < len(xs) - k.
    """
    return all(pred(xs[i], xs[i + k]) for i in range(len(xs) - k))


def prim_stride(k: int, xs: List[T]) -> List[T]:
    """Return every k-th element: xs[::k].

    Useful for extracting odd-positioned or even-positioned cards,
    or any regular sub-sequence.

    Args:
        k: Step size (must be >= 1).
        xs: The list to stride over.

    Returns:
        Elements at indices 0, k, 2k, ...
    """
    return xs[::k]


def prim_count_where(pred: Callable[[T], bool], xs: List[T]) -> int:
    """Count elements of xs for which pred returns True.

    Subsumes specialised counters like n_red, n_high, n_face, etc.

    Args:
        pred: A unary predicate.
        xs: The list to scan.

    Returns:
        Number of elements satisfying pred.
    """
    return sum(1 for x in xs if pred(x))


def prim_sorted_counts(key_fn: Callable[[T], K], xs: List[T]) -> List[int]:
    """Group xs by key_fn, count each group, return counts sorted descending.

    This single primitive captures the "shape" of a hand's distribution
    and subsumes many existing checks:
      - n_unique(key, xs)       = len(prim_sorted_counts(key, xs))
      - is_flush(xs)            = prim_sorted_counts(suit, xs) == [n]
      - has_pair(xs)            = prim_sorted_counts(rank, xs)[0] >= 2
      - most_common_count(xs)   = prim_sorted_counts(key, xs)[0]

    Args:
        key_fn: Function mapping each element to a grouping key.
        xs: The list to group.

    Returns:
        List of group sizes, sorted from largest to smallest.
    """
    if not xs:
        return []
    counts = Counter(key_fn(x) for x in xs)
    return sorted(counts.values(), reverse=True)
