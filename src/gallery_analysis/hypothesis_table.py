"""
Hypothesis table: fingerprinting, equivalence classes, and hit tracking.

Each enumerated program (hypothesis) is:
1. Evaluated on a probe set to get a boolean fingerprint
2. Grouped into equivalence classes by fingerprint
3. Evaluated on exemplar hands to get a hit vector

Equivalence classes track:
- The canonical (shortest/most probable) program
- All alternative expressions
- The shared fingerprint, hit vector, and extension size estimate
"""
import hashlib
import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank


def compute_fingerprint(predicate: Callable[[Hand], bool], probes: List[Hand]) -> str:
    """
    Compute a fingerprint for a predicate by evaluating it on probe hands.

    Returns a hex hash of the boolean output vector. Two predicates with the
    same hash are treated as extensionally equivalent.
    """
    bits = []
    for hand in probes:
        try:
            result = predicate(hand)
            bits.append("1" if result else "0")
        except Exception:
            bits.append("E")

    bit_string = "".join(bits)
    return hashlib.sha256(bit_string.encode()).hexdigest()


def estimate_extension_size(
    predicate: Callable[[Hand], bool],
    n_samples: int = 100_000,
    hand_size: int = 6,
    seed: int = 123
) -> Tuple[int, float]:
    """
    Estimate |extension(h)| via Monte Carlo sampling.

    Samples random hands from the full deck and counts how many satisfy
    the predicate. Scales up to the total number of possible hands C(52,6).

    Returns:
        (estimated_extension_size, base_rate)
    """
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]

    hits = 0
    for _ in range(n_samples):
        hand = rng.sample(deck, hand_size)
        try:
            if predicate(hand):
                hits += 1
        except Exception:
            pass

    base_rate = hits / n_samples
    # C(52, 6) = 20,358,520
    total_hands = 20_358_520
    estimated_size = int(base_rate * total_hands)

    return estimated_size, base_rate


@dataclass
class HypothesisEntry:
    """A single enumerated program/hypothesis."""
    program_str: str
    predicate: Callable[[Hand], bool]
    log_prior: float
    fingerprint: str
    hit_vector: Optional[List[bool]] = None
    n_hits: int = 0
    n_misses: int = 0


class HypothesisTable:
    """
    Manages all enumerated hypotheses, grouped by observational equivalence.

    Usage:
        table = HypothesisTable(probe_hands)
        table.add("program_str", predicate_fn, log_prior=-5.0, exemplar_hands=hands)
        ...
        classes = table.get_equivalence_classes()
    """

    def __init__(self, probes: List[Hand]):
        self.probes = probes
        # fingerprint -> list of HypothesisEntry
        self._classes: Dict[str, List[HypothesisEntry]] = {}
        self._total_added = 0
        self._total_deduplicated = 0

    def add(
        self,
        program_str: str,
        predicate: Callable[[Hand], bool],
        log_prior: float,
        exemplar_hands: Optional[List[Hand]] = None
    ) -> bool:
        """
        Add a hypothesis to the table.

        Returns True if this is a new equivalence class, False if deduplicated.
        """
        self._total_added += 1

        fp = compute_fingerprint(predicate, self.probes)

        hit_vector = None
        n_hits = 0
        n_misses = 0
        if exemplar_hands:
            hit_vector = []
            for hand in exemplar_hands:
                try:
                    result = predicate(hand)
                    hit_vector.append(result)
                    if result:
                        n_hits += 1
                    else:
                        n_misses += 1
                except Exception:
                    hit_vector.append(False)
                    n_misses += 1

        entry = HypothesisEntry(
            program_str=program_str,
            predicate=predicate,
            log_prior=log_prior,
            fingerprint=fp,
            hit_vector=hit_vector,
            n_hits=n_hits,
            n_misses=n_misses,
        )

        is_new = fp not in self._classes
        if is_new:
            self._classes[fp] = [entry]
        else:
            self._classes[fp].append(entry)
            self._total_deduplicated += 1

        return is_new

    def get_equivalence_classes(self) -> List[Dict[str, Any]]:
        """
        Return all equivalence classes, each with canonical program and statistics.

        Classes are sorted by canonical prior (most probable first).
        """
        classes = []
        for fp, entries in self._classes.items():
            # Canonical = highest prior (least negative log_prior)
            entries_sorted = sorted(entries, key=lambda e: -e.log_prior)
            canonical = entries_sorted[0]

            summed_prior = math.log(sum(math.exp(e.log_prior) for e in entries))

            classes.append({
                "canonical_program": canonical.program_str,
                "canonical_prior": canonical.log_prior,
                "summed_prior": summed_prior,
                "n_expressions": len(entries),
                "all_programs": [e.program_str for e in entries_sorted],
                "fingerprint": fp,
                "hit_vector": canonical.hit_vector,
                "n_hits": canonical.n_hits,
                "n_misses": canonical.n_misses,
                "predicate": canonical.predicate,
            })

        classes.sort(key=lambda c: -c["canonical_prior"])
        return classes

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_added": self._total_added,
            "total_deduplicated": self._total_deduplicated,
            "n_equivalence_classes": len(self._classes),
        }
