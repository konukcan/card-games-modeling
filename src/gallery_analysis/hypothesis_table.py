"""
Hypothesis table: fingerprinting, equivalence classes, and hit tracking.

Pipeline (informed by empirical exploration — see explore_trivial_vs_rare.py):

  1. TRIVIAL FILTER: Evaluate each program on all 360 curated exemplar hands
     (6 per rule × 60 rules). Programs that are constant (all-True or all-False)
     across ALL 360 hands are discarded as trivially equivalent to `true`/`false`.
     Empirically, this removes ~96% of enumerated programs.

  2. FINGERPRINT: Evaluate survivors on a probe set of random hands and hash
     the boolean output vector. Programs with identical fingerprints are grouped
     into equivalence classes. A modest probe set (1K-5K) suffices since the
     trivial filter already removed the bulk of collisions.

  3. HIT VECTOR: Evaluate each equivalence class on the target rule's exemplar
     hands to determine how many it covers (needed for Bayesian scoring).

BIAS DOCUMENTATION:
  The trivial filter uses curated exemplar hands from the 60 gallery rules.
  This introduces a bias: a rare hypothesis that is non-trivial but happens
  not to fire on ANY of the 360 curated exemplars will be incorrectly
  discarded as trivial. However, such a hypothesis would also receive zero
  likelihood in the Bayesian analysis (it fails to explain any rule's data),
  so this bias does not affect posterior computations for the 60 gallery rules.
  If the analysis were extended to rules beyond the gallery set, the exemplar
  pool would need to be expanded accordingly.

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


def is_trivial(
    predicate: Callable[[Hand], bool],
    all_exemplar_hands: List[Hand],
) -> bool:
    """
    Check if a predicate is trivially constant (always-True or always-False)
    by evaluating it on the curated exemplar hands from all 60 gallery rules.

    A predicate that produces the same boolean on all 360 curated exemplar hands
    is treated as trivial, since those hands cover a wide range of structural
    patterns (color, suit, rank, positional, sequential, etc.).

    BIAS: A non-trivial predicate that happens not to fire on any of the 360
    curated hands will be incorrectly classified as trivial. See module docstring
    for why this is acceptable for the gallery analysis.

    Args:
        predicate: The hypothesis function (Hand -> bool)
        all_exemplar_hands: All curated exemplar hands (typically 360 = 6 × 60 rules)

    Returns:
        True if the predicate appears trivially constant, False otherwise.
    """
    if not all_exemplar_hands:
        return False

    # Evaluate on first hand to establish baseline
    try:
        first_result = predicate(all_exemplar_hands[0])
    except Exception:
        return True  # errors on curated hands → treat as trivial

    # Check remaining hands — any disagreement means non-trivial
    for hand in all_exemplar_hands[1:]:
        try:
            if predicate(hand) != first_result:
                return False
        except Exception:
            # An error after a successful evaluation is a form of disagreement
            return False

    return True


def filter_trivial(
    programs: List[Tuple[str, Callable, float]],
    all_exemplar_hands: List[Hand],
) -> Tuple[List[Tuple[str, Callable, float]], Dict[str, int]]:
    """
    Remove trivially constant programs using the curated exemplar filter.

    Args:
        programs: List of (program_str, predicate_fn, log_prior)
        all_exemplar_hands: All 360 curated exemplar hands

    Returns:
        (surviving_programs, filter_stats)
        where filter_stats has keys: total, trivial_true, trivial_false, survivors
    """
    survivors = []
    n_trivial_true = 0
    n_trivial_false = 0

    for prog_str, pred_fn, log_prior in programs:
        if is_trivial(pred_fn, all_exemplar_hands):
            # Classify which kind of trivial
            try:
                if pred_fn(all_exemplar_hands[0]):
                    n_trivial_true += 1
                else:
                    n_trivial_false += 1
            except Exception:
                n_trivial_false += 1
        else:
            survivors.append((prog_str, pred_fn, log_prior))

    stats = {
        "total": len(programs),
        "trivial_true": n_trivial_true,
        "trivial_false": n_trivial_false,
        "survivors": len(survivors),
    }
    return survivors, stats


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
    # P(52, 6) = 14,658,134,400 (ordered, without replacement)
    total_hands = 14_658_134_400
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


def _safe_eval(predicate: Callable[[Hand], bool], hand: Hand) -> bool:
    """Evaluate predicate, returning False on any exception."""
    try:
        return bool(predicate(hand))
    except Exception:
        return False


def refine_rare_classes(
    equivalence_classes: List[Dict[str, Any]],
    main_probes: List[Hand],
    hit_threshold: int = 5,
    n_refinement_probes: int = 2000,
    refinement_seed: int = 4242,
) -> List[Dict[str, Any]]:
    """Second-pass refinement for rare equivalence classes.

    Classes whose canonical predicate fires on <= hit_threshold of the
    main probes are re-fingerprinted with additional refinement probes.
    If members diverge, the class is split.

    Args:
        equivalence_classes: Output of HypothesisTable.get_equivalence_classes()
            or a prior refinement pass.  Each dict must have at least:
            canonical_program, canonical_prior, summed_prior, n_expressions,
            all_programs, fingerprint, predicate.
            For splitting to work on multi-program classes, the dict should
            also carry _all_predicates and _all_priors.
        main_probes: The probe hands used for the original fingerprinting.
        hit_threshold: Classes whose canonical predicate fires on at most
            this many main probes are considered "rare" and eligible for
            refinement.
        n_refinement_probes: How many extra random hands to generate for
            the second-pass fingerprinting.
        refinement_seed: RNG seed for the refinement probe set (must differ
            from the main probe seed to add information).

    Returns:
        A new list of equivalence-class dicts.  Non-rare and single-program
        classes are passed through unchanged; rare multi-program classes may
        be split into several sub-classes.
    """
    from gallery_analysis.exemplars import generate_probe_set

    refinement_probes = generate_probe_set(n_refinement_probes, seed=refinement_seed)

    result: List[Dict[str, Any]] = []
    n_split = 0

    for cls in equivalence_classes:
        pred = cls["predicate"]
        hits = sum(1 for p in main_probes if _safe_eval(pred, p))

        # Non-rare classes or single-program classes cannot be split
        if hits > hit_threshold or cls["n_expressions"] <= 1:
            result.append(cls)
            continue

        all_predicates = cls.get("_all_predicates")
        all_priors = cls.get("_all_priors")
        if not all_predicates or len(all_predicates) <= 1:
            result.append(cls)
            continue

        # Re-fingerprint each member on the refinement probes
        sub_groups: Dict[str, List[int]] = {}
        for prog_idx, prog_pred in enumerate(all_predicates):
            refined_fp = compute_fingerprint(prog_pred, refinement_probes)
            sub_groups.setdefault(refined_fp, []).append(prog_idx)

        if len(sub_groups) <= 1:
            # All members still agree — no split needed
            result.append(cls)
            continue

        # Split into sub-classes
        all_programs = cls["all_programs"]
        for refined_fp, indices in sub_groups.items():
            sub_priors = [all_priors[i] for i in indices]
            best_idx = indices[sub_priors.index(max(sub_priors))]
            summed = math.log(sum(math.exp(all_priors[i]) for i in indices))

            # Combined fingerprint uses both main and refinement probes
            combined_fp = compute_fingerprint(
                all_predicates[best_idx],
                main_probes + refinement_probes,
            )

            sub_class: Dict[str, Any] = {
                "canonical_program": all_programs[best_idx],
                "canonical_prior": all_priors[best_idx],
                "summed_prior": summed,
                "n_expressions": len(indices),
                "all_programs": [all_programs[i] for i in indices],
                "fingerprint": combined_fp,
                "predicate": all_predicates[best_idx],
            }
            # Carry through any extra keys from the original class
            for key in cls:
                if key not in sub_class and key not in ("_all_predicates", "_all_priors"):
                    sub_class[key] = cls[key]

            result.append(sub_class)
        n_split += 1

    if n_split > 0:
        print(
            f"  Fingerprint refinement: {n_split} rare classes split, "
            f"total classes: {len(result)} (was {len(equivalence_classes)})",
            flush=True,
        )

    return result
