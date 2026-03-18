"""
Tests for the MCMC hypothesis trajectory analyzer.

Verifies that TrajectoryAnalyzer correctly computes dwelling times,
frequency rankings (including filtered rankings), first-passage orderings,
and summary statistics from MCMCResult objects.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from dreamcoder_core.type_system import Arrow, BOOL, HAND
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.mcmc_search import (
    MCMCConfig, MCMCChain, MCMCResult, run_parallel_chains,
)
from gallery_analysis.mcmc_hypothesis_collector import TrajectoryAnalyzer


@pytest.fixture
def grammar():
    """Build the gallery grammar."""
    return build_gallery_grammar()


@pytest.fixture
def exemplars():
    """Load frozen exemplar hands."""
    return load_exemplars()


@pytest.fixture
def chain_result(grammar, exemplars):
    """Run a short chain and return the result."""
    config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    return MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )


def test_frequency_ranking(chain_result):
    """frequency_ranking should return sorted results with correct fields."""
    analyzer = TrajectoryAnalyzer(chain_result)
    ranking = analyzer.frequency_ranking(top_k=10)
    assert len(ranking) > 0
    for entry in ranking:
        assert 'program' in entry
        assert 'visit_count' in entry
        assert 'empirical_posterior' in entry
        assert 'first_step' in entry
    # Should be sorted by visit count descending
    counts = [e['visit_count'] for e in ranking]
    assert counts == sorted(counts, reverse=True)


def test_frequency_ranking_filtered(chain_result):
    """frequency_ranking_filtered should exclude high-extension programs."""
    analyzer = TrajectoryAnalyzer(chain_result)

    # Filtered ranking should have ext_fraction field
    filtered = analyzer.frequency_ranking_filtered(top_k=10, max_ext_fraction=0.5)
    for entry in filtered:
        assert 'ext_fraction' in entry
        assert entry['ext_fraction'] <= 0.5, (
            f"Program {entry['program'][:60]} has ext_fraction={entry['ext_fraction']} > 0.5"
        )

    # Unfiltered ranking may include programs that filtered excludes
    unfiltered = analyzer.frequency_ranking(top_k=50)
    assert len(filtered) <= len(unfiltered)


def test_frequency_ranking_filtered_strict(chain_result):
    """With max_ext_fraction=0.0, only programs with 0 extension should pass."""
    analyzer = TrajectoryAnalyzer(chain_result)
    strict = analyzer.frequency_ranking_filtered(top_k=50, max_ext_fraction=0.0)
    for entry in strict:
        assert entry['ext_fraction'] == 0.0


def test_dwelling_times(chain_result):
    """dwelling_times should return positive counts for visited programs."""
    analyzer = TrajectoryAnalyzer(chain_result)
    dwellings = analyzer.dwelling_times()
    assert len(dwellings) > 0
    for prog, time in dwellings.items():
        assert time > 0


def test_first_passage_ordering(chain_result):
    """first_passage_ordering should be sorted by step ascending."""
    analyzer = TrajectoryAnalyzer(chain_result)
    ordering = analyzer.first_passage_ordering()
    if len(ordering) > 1:
        steps = [e['first_step'] for e in ordering]
        assert steps == sorted(steps)


def test_summary(chain_result):
    """summary should return required fields with valid values."""
    analyzer = TrajectoryAnalyzer(chain_result)
    s = analyzer.summary()
    assert s['n_steps'] == chain_result.n_steps
    assert s['n_unique'] == chain_result.n_unique
    assert 0.0 <= s['acceptance_rate'] <= 1.0
    assert s['entropy_bits'] >= 0.0
    assert 0.0 <= s['concentration'] <= 1.0


def test_consecutive_dwelling_times(chain_result):
    """consecutive_dwelling_times should produce valid run lengths."""
    analyzer = TrajectoryAnalyzer(chain_result)
    dwellings = analyzer.consecutive_dwelling_times()
    if dwellings:
        for prog, runs in dwellings.items():
            assert all(r > 0 for r in runs), f"Invalid run lengths for {prog[:60]}"
            # Sum of run lengths across all programs should equal trajectory length
        total_runs = sum(sum(runs) for runs in dwellings.values())
        assert total_runs == len(chain_result.trajectory)
