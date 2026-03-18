"""
Hypothesis trajectory analysis for MCMC chains.

This module extracts process-level data from MCMC chains for cognitive modeling.
The key idea is that MCMC sampling produces not just a posterior distribution,
but a *trajectory* of hypothesis consideration that maps onto human cognition:

- **Dwelling times**: How long the chain stays at a hypothesis reflects how
  "sticky" or compelling that hypothesis is. Longer dwelling predicts greater
  subjective confidence and slower disengagement.

- **Visit frequency**: The number of times a hypothesis is visited approximates
  its empirical posterior weight. Higher visit count = stronger belief.

- **First-passage time**: The step at which a hypothesis is first discovered
  predicts the *order* of hypothesis consideration by a learner. Earlier
  first-passage = more accessible or salient hypothesis.

Together these make testable predictions about human learning dynamics:
learners should consider hypotheses in roughly first-passage order, dwell
longer on high-posterior hypotheses, and show concentration effects when
one hypothesis dominates the chain.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.mcmc_search import MCMCResult


@dataclass
class HypothesisTrajectory:
    """A single hypothesis and its trajectory statistics.

    Attributes:
        program_str:     The program string representation.
        visit_count:     Total number of times the chain visited this hypothesis.
        first_seen_step: The MCMC step at which this hypothesis was first visited.
        dwelling_time:   Same as visit_count for now (no step-by-step trajectory yet).
                         When step-by-step recording is added (Task 2.2), this will
                         track consecutive dwelling instead.
    """
    program_str: str
    visit_count: int
    first_seen_step: int
    dwelling_time: int  # Same as visit_count for now (no step-by-step trajectory yet)


class TrajectoryAnalyzer:
    """Analyzes MCMC chain results to extract cognitive-process-level data.

    Given an MCMCResult from a completed chain, this class provides methods
    to compute dwelling times, frequency rankings, first-passage orderings,
    and summary statistics that can be compared against human behavioral data.

    Args:
        result: An MCMCResult from run_parallel_chains or run_mcmc_chain.
    """

    def __init__(self, result: MCMCResult) -> None:
        self.result = result

    def dwelling_times(self) -> Dict[str, int]:
        """Return visit counts as a dwelling time proxy.

        Currently, dwelling time equals visit count because the chain only
        records aggregate visit counts, not the step-by-step trajectory.
        Step-by-step recording (Task 2.2) will enable computing consecutive
        dwelling times (how many steps in a row the chain stays at a hypothesis).

        Returns:
            Dict mapping program string to its dwelling time (= visit count).
        """
        return dict(self.result.visit_counts)

    def frequency_ranking(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """Rank hypotheses by visit frequency (empirical posterior weight).

        Visit frequency approximates the posterior probability of each hypothesis.
        A hypothesis visited 100 times out of 1000 steps has an empirical
        posterior of ~0.10.

        Args:
            top_k: Maximum number of hypotheses to return, sorted by visit count
                   descending. Use len(result.visit_counts) to get all.

        Returns:
            List of dicts, each with keys:
              - 'program': the program string
              - 'visit_count': total visits
              - 'empirical_posterior': visit_count / n_steps
              - 'first_step': step at which hypothesis was first seen (-1 if unknown)
        """
        total = max(1, self.result.n_steps)
        sorted_hyps = sorted(
            self.result.visit_counts.items(), key=lambda x: -x[1]
        )[:top_k]
        return [
            {
                'program': prog,
                'visit_count': count,
                'empirical_posterior': count / total,
                'first_step': self.result.first_passage.get(prog, -1),
            }
            for prog, count in sorted_hyps
        ]

    def frequency_ranking_filtered(
        self, top_k: int = 50, max_ext_fraction: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Rank hypotheses by visit frequency, excluding near-tautologies.

        Like frequency_ranking() but filters out programs whose extension
        fraction exceeds max_ext_fraction. Extension fraction is the proportion
        of random probe hands that the program accepts -- a value of 0.5 means
        the program accepts half of all hands, making it too vague to be a
        useful hypothesis.

        Args:
            top_k:            Maximum number of hypotheses to return.
            max_ext_fraction: Programs with ext_fraction > this are excluded.
                              Default 0.5 removes programs accepting >50% of hands.

        Returns:
            List of dicts sorted by visit count descending, each with keys:
              - 'program': the program string
              - 'visit_count': total visits
              - 'empirical_posterior': visit_count / n_steps
              - 'first_step': step at which hypothesis was first seen (-1 if unknown)
              - 'ext_fraction': fraction of probe hands accepted by the program
        """
        total = max(1, self.result.n_steps)
        sorted_hyps = sorted(
            self.result.visit_counts.items(), key=lambda x: -x[1]
        )
        filtered = []
        for prog, count in sorted_hyps:
            ext_frac = self.result.ext_fractions.get(prog, 0.0)
            if ext_frac <= max_ext_fraction:
                filtered.append({
                    'program': prog,
                    'visit_count': count,
                    'empirical_posterior': count / total,
                    'first_step': self.result.first_passage.get(prog, -1),
                    'ext_fraction': ext_frac,
                })
                if len(filtered) >= top_k:
                    break
        return filtered

    def first_passage_ordering(self) -> List[Dict[str, Any]]:
        """Order hypotheses by first-passage time (when first discovered).

        First-passage time predicts the order in which a learner considers
        hypotheses: earlier discovery in the chain corresponds to more
        accessible or salient rules.

        Returns:
            List of dicts sorted by first_step ascending, each with keys:
              - 'program': the program string
              - 'first_step': step at which hypothesis was first seen
              - 'visit_count': total visits for this hypothesis
        """
        sorted_by_fp = sorted(
            self.result.first_passage.items(), key=lambda x: x[1]
        )
        return [
            {
                'program': prog,
                'first_step': step,
                'visit_count': self.result.visit_counts.get(prog, 0),
            }
            for prog, step in sorted_by_fp
        ]

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics for the chain.

        Returns a dict with:
          - n_steps: total MCMC steps
          - n_unique: number of distinct programs visited
          - acceptance_rate: fraction of proposals accepted
          - entropy_bits: Shannon entropy of the visit distribution (bits),
            measuring how spread out the chain's attention is
          - top_1_program: the most-visited program string (or None)
          - top_1_visits: visit count of the most-visited program
          - concentration: fraction of total visits at the mode (top program),
            measuring how peaked the empirical posterior is
        """
        visits = list(self.result.visit_counts.values())
        total = sum(visits)

        # Shannon entropy of the visit distribution in bits
        entropy = 0.0
        for v in visits:
            p = v / max(1, total)
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            'n_steps': self.result.n_steps,
            'n_unique': self.result.n_unique,
            'acceptance_rate': self.result.acceptance_rate,
            'entropy_bits': entropy,
            'top_1_program': self.result.best_program,
            'top_1_visits': max(visits) if visits else 0,
            'concentration': max(visits) / max(1, total),
        }

    def transition_counts(self) -> Dict[Tuple[str, str], int]:
        """Count transitions between hypotheses in the trajectory.

        A transition (A -> B) means the chain moved from A to B in one step.
        Self-transitions (A -> A) are counted when a proposal is rejected.

        Returns:
            Dict mapping (from_prog, to_prog) -> count.
        """
        if not self.result.trajectory:
            return {}
        counts: Dict[Tuple[str, str], int] = {}
        for i in range(len(self.result.trajectory) - 1):
            pair = (self.result.trajectory[i], self.result.trajectory[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def consecutive_dwelling_times(self) -> Dict[str, List[int]]:
        """Compute consecutive dwelling times (run lengths) per hypothesis.

        For each hypothesis, collect the lengths of consecutive runs
        (how many steps the chain stayed before moving away).

        Example: trajectory [A, A, A, B, B, A, A] gives:
            A: [3, 2]  (two runs of length 3 and 2)
            B: [2]     (one run of length 2)

        The mean run length is the expected "stickiness" of a hypothesis.
        """
        if not self.result.trajectory:
            return {}

        dwellings: Dict[str, List[int]] = {}
        current = self.result.trajectory[0]
        run_length = 1

        for i in range(1, len(self.result.trajectory)):
            if self.result.trajectory[i] == current:
                run_length += 1
            else:
                dwellings.setdefault(current, []).append(run_length)
                current = self.result.trajectory[i]
                run_length = 1
        # Don't forget the last run
        dwellings.setdefault(current, []).append(run_length)

        return dwellings
