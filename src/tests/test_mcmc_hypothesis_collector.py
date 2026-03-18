"""Tests for MCMC hypothesis trajectory collection."""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dreamcoder_core.type_system import Arrow, BOOL, HAND
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.mcmc_search import MCMCConfig, run_parallel_chains
from gallery_analysis.mcmc_hypothesis_collector import TrajectoryAnalyzer, HypothesisTrajectory


@pytest.fixture(scope="module")
def mcmc_result():
    """Run MCMC once and reuse result across tests."""
    grammar = build_gallery_grammar()
    exemplars = load_exemplars()
    config = MCMCConfig(n_steps=2000, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    return run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=2,
    )


def test_dwelling_times(mcmc_result):
    analyzer = TrajectoryAnalyzer(mcmc_result)
    dwellings = analyzer.dwelling_times()
    assert len(dwellings) > 0
    for prog, time in dwellings.items():
        assert time > 0


def test_frequency_ranking_sorted(mcmc_result):
    analyzer = TrajectoryAnalyzer(mcmc_result)
    ranking = analyzer.frequency_ranking(top_k=10)
    assert len(ranking) > 0
    assert len(ranking) <= 10
    for i in range(len(ranking) - 1):
        assert ranking[i]['visit_count'] >= ranking[i + 1]['visit_count']


def test_frequency_ranking_has_fields(mcmc_result):
    analyzer = TrajectoryAnalyzer(mcmc_result)
    ranking = analyzer.frequency_ranking(top_k=5)
    for entry in ranking:
        assert 'program' in entry
        assert 'visit_count' in entry
        assert 'empirical_posterior' in entry
        assert 'first_step' in entry
        assert 0 <= entry['empirical_posterior'] <= 1


def test_first_passage_ordering_sorted(mcmc_result):
    analyzer = TrajectoryAnalyzer(mcmc_result)
    ordering = analyzer.first_passage_ordering()
    assert len(ordering) > 0
    for i in range(len(ordering) - 1):
        assert ordering[i]['first_step'] <= ordering[i + 1]['first_step']


def test_summary_has_keys(mcmc_result):
    analyzer = TrajectoryAnalyzer(mcmc_result)
    s = analyzer.summary()
    assert 'n_steps' in s
    assert 'n_unique' in s
    assert 'entropy_bits' in s
    assert 'concentration' in s
    assert s['entropy_bits'] >= 0
    assert 0 <= s['concentration'] <= 1


def test_empirical_posteriors_sum_to_one(mcmc_result):
    analyzer = TrajectoryAnalyzer(mcmc_result)
    # Get ALL hypotheses (no top_k limit)
    ranking = analyzer.frequency_ranking(top_k=len(mcmc_result.visit_counts))
    total_post = sum(e['empirical_posterior'] for e in ranking)
    assert abs(total_post - 1.0) < 0.01, f"Posteriors sum to {total_post}, not ~1.0"
