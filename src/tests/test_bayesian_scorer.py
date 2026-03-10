"""Tests for Bayesian scorer: likelihood, posteriors, and difficulty metrics."""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.bayesian_scorer import (
    compute_log_likelihood_strict,
    compute_log_likelihood_noisy,
    score_hypotheses,
    normalize_posteriors,
    compute_rule_difficulty,
    TOTAL_HANDS,
)


# =========================================================================
# Strict likelihood tests
# =========================================================================

def test_strict_likelihood_all_hits():
    """With all exemplars hitting, likelihood = (1/|ext|)^n."""
    # 6 exemplars, extension size 1000
    ll = compute_log_likelihood_strict(n_hits=6, n_exemplars=6, extension_size=1000)
    expected = -6 * math.log(1000)
    assert abs(ll - expected) < 1e-10

def test_strict_likelihood_any_miss_is_zero():
    """If any exemplar misses, strict likelihood is -inf."""
    ll = compute_log_likelihood_strict(n_hits=5, n_exemplars=6, extension_size=1000)
    assert ll == float('-inf')

def test_strict_likelihood_empty_extension():
    """Empty extension (no hands satisfy h) gives -inf."""
    ll = compute_log_likelihood_strict(n_hits=6, n_exemplars=6, extension_size=0)
    assert ll == float('-inf')

def test_strict_likelihood_smaller_extension_is_better():
    """Size principle: smaller extension → higher likelihood."""
    ll_small = compute_log_likelihood_strict(6, 6, extension_size=100)
    ll_large = compute_log_likelihood_strict(6, 6, extension_size=10000)
    assert ll_small > ll_large  # smaller extension = higher likelihood


# =========================================================================
# Noisy likelihood tests
# =========================================================================

def test_noisy_likelihood_all_hits():
    """With all hits, noisy likelihood should be close to strict (for small ε)."""
    ll_strict = compute_log_likelihood_strict(6, 6, 1000)
    ll_noisy = compute_log_likelihood_noisy(6, 6, 1000, epsilon=0.001)
    # Should be close but slightly higher (noise adds a small uniform component)
    assert abs(ll_strict - ll_noisy) < 1.0  # within 1 nat

def test_noisy_likelihood_with_misses_is_finite():
    """Unlike strict, noisy likelihood is finite even with misses."""
    ll = compute_log_likelihood_noisy(n_hits=4, n_exemplars=6, extension_size=1000, epsilon=0.01)
    assert ll > float('-inf')
    assert ll < 0  # should be negative (log probability)

def test_noisy_likelihood_more_misses_is_worse():
    """More misses should give lower noisy likelihood."""
    ll_few_miss = compute_log_likelihood_noisy(5, 6, 1000, epsilon=0.01)
    ll_many_miss = compute_log_likelihood_noisy(3, 6, 1000, epsilon=0.01)
    assert ll_few_miss > ll_many_miss


# =========================================================================
# Normalization tests
# =========================================================================

def test_normalize_posteriors_sum_to_one():
    """Normalized posteriors should sum to 1."""
    from gallery_analysis.bayesian_scorer import ScoredHypothesis

    # Create mock scored hypotheses with different posteriors
    scored = []
    for i, lp in enumerate([-5.0, -10.0, -15.0]):
        sh = ScoredHypothesis(
            canonical_program=f"prog_{i}",
            n_expressions=1,
            log_prior_canonical=-3.0,
            log_prior_summed=-3.0,
            hit_vector=[True] * 6,
            n_hits=6,
            n_exemplars=6,
            extension_size=1000,
            base_rate=0.0001,
            log_likelihood_strict=lp + 3.0,
            log_likelihood_noisy=lp + 3.0,
            log_posterior_strict=lp,
            log_posterior_noisy=lp,
            fingerprint=f"fp_{i}",
            all_programs=[f"prog_{i}"],
        )
        scored.append(sh)

    normalized = normalize_posteriors(scored, mode="noisy")
    total_prob = sum(p for _, p in normalized)
    assert abs(total_prob - 1.0) < 1e-10

def test_normalize_posteriors_ordering():
    """Higher posterior should get higher probability."""
    from gallery_analysis.bayesian_scorer import ScoredHypothesis

    scored = []
    for i, lp in enumerate([-5.0, -10.0]):
        sh = ScoredHypothesis(
            canonical_program=f"prog_{i}",
            n_expressions=1,
            log_prior_canonical=-3.0,
            log_prior_summed=-3.0,
            hit_vector=[True] * 6,
            n_hits=6,
            n_exemplars=6,
            extension_size=1000,
            base_rate=0.0001,
            log_likelihood_strict=lp + 3.0,
            log_likelihood_noisy=lp + 3.0,
            log_posterior_strict=lp,
            log_posterior_noisy=lp,
            fingerprint=f"fp_{i}",
            all_programs=[f"prog_{i}"],
        )
        scored.append(sh)

    normalized = normalize_posteriors(scored, mode="noisy")
    # First should have higher probability (less negative posterior)
    assert normalized[0][1] > normalized[1][1]


# =========================================================================
# Difficulty metric tests
# =========================================================================

def test_difficulty_peaked_posterior_is_easy():
    """A posterior dominated by one hypothesis should have low entropy."""
    from gallery_analysis.bayesian_scorer import ScoredHypothesis

    sh = ScoredHypothesis(
        canonical_program="true_rule",
        n_expressions=1,
        log_prior_canonical=-3.0,
        log_prior_summed=-3.0,
        hit_vector=[True] * 6,
        n_hits=6,
        n_exemplars=6,
        extension_size=100,
        base_rate=0.00001,
        log_likelihood_strict=-30.0,
        log_likelihood_noisy=-30.0,
        log_posterior_strict=-33.0,
        log_posterior_noisy=-33.0,
        fingerprint="fp_0",
        all_programs=["true_rule"],
    )

    # One dominant hypothesis
    normalized = [(sh, 0.99)] + [(sh, 0.01 / 9)] * 9
    difficulty = compute_rule_difficulty(normalized, true_rule_program="true_rule")

    assert difficulty["top1_probability"] > 0.9
    assert difficulty["n_effective_hypotheses"] < 3.0
    assert difficulty["true_rule_rank"] == 1

def test_difficulty_flat_posterior_is_hard():
    """A flat posterior should have high entropy."""
    from gallery_analysis.bayesian_scorer import ScoredHypothesis

    # 10 equally likely hypotheses
    normalized = []
    for i in range(10):
        sh = ScoredHypothesis(
            canonical_program=f"prog_{i}",
            n_expressions=1,
            log_prior_canonical=-3.0,
            log_prior_summed=-3.0,
            hit_vector=[True] * 6,
            n_hits=6,
            n_exemplars=6,
            extension_size=1000,
            base_rate=0.0001,
            log_likelihood_strict=-40.0,
            log_likelihood_noisy=-40.0,
            log_posterior_strict=-43.0,
            log_posterior_noisy=-43.0,
            fingerprint=f"fp_{i}",
            all_programs=[f"prog_{i}"],
        )
        normalized.append((sh, 0.1))

    difficulty = compute_rule_difficulty(normalized)
    assert difficulty["top1_probability"] < 0.15
    assert difficulty["n_effective_hypotheses"] > 8.0
    assert difficulty["posterior_entropy"] > 2.0  # log(10) ≈ 2.3
