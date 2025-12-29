#!/usr/bin/env python3
"""
Description Generator Evaluation Harness

This module provides a comprehensive evaluation framework for testing the
description generator on synthetic card game rules. It implements:

1. Task generation and train/test splitting
2. Multiple evaluation metrics (accuracy, consistency, discrimination)
3. Baseline comparisons
4. Detailed reporting and visualization

Usage:
    python evaluation_harness.py [--quick] [--seed 42] [--output results.json]

Author: Description Generator Development
Date: December 2024
"""

import sys
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import statistics

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, sample_hand, hand_to_string
from description_generator.synthetic_tasks import (
    SyntheticTaskGenerator, SyntheticRule, generate_synthetic_rules
)
from description_generator.description_generator import (
    DescriptionGenerator, Description, FeatureExtractor, FeatureType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SEMANTIC CATEGORY MAPPING
# ============================================================================

# Maps rule categories to acceptable feature types for semantic correctness
# If a generated description's feature matches ANY in this set, it's semantically correct
CATEGORY_TO_FEATURES: Dict[str, Set[FeatureType]] = {
    # Uniform properties - all same X
    "uniform": {FeatureType.IS_FLUSH, FeatureType.IS_UNIFORM_COLOR,
                FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_COLORS, FeatureType.UNIQUE_RANKS},

    # Position-based features
    "position_first": {FeatureType.FIRST_CARD},
    "position_last": {FeatureType.LAST_CARD},
    "position_second": {FeatureType.FIRST_CARD},  # Close enough
    "position_middle": {FeatureType.FIRST_CARD, FeatureType.LAST_CARD},

    # Terminal/endpoint features
    "terminals": {FeatureType.ENDS_SAME_SUIT, FeatureType.ENDS_SAME_COLOR,
                  FeatureType.FIRST_CARD, FeatureType.LAST_CARD},
    "terminals_rank": {FeatureType.FIRST_CARD, FeatureType.LAST_CARD, FeatureType.RANK_SPREAD},

    # Sum-based features
    "sum": {FeatureType.SUM_RANKS},
    "sum_divisibility": {FeatureType.SUM_RANKS},

    # Pair/duplicate features
    "has": {FeatureType.HAS_PAIR, FeatureType.HAS_TRIPLE, FeatureType.UNIQUE_RANKS},
    "no": {FeatureType.HAS_PAIR, FeatureType.HAS_TRIPLE, FeatureType.UNIQUE_RANKS},
    "duplicates": {FeatureType.HAS_PAIR, FeatureType.HAS_TRIPLE, FeatureType.UNIQUE_RANKS},

    # Half-based features
    "halves_property": {FeatureType.HALVES_SAME_COLOR, FeatureType.IS_UNIFORM_COLOR},
    "halves_set": {FeatureType.HALVES_SAME_COLOR, FeatureType.UNIQUE_SUITS},
    "halves_sum": {FeatureType.SUM_RANKS},
    "halves_copy": {FeatureType.HALVES_SAME_COLOR, FeatureType.IS_PALINDROME_SUITS},

    # Count features
    "unique_count": {FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_RANKS, FeatureType.UNIQUE_COLORS},
    "unique_bound": {FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_RANKS, FeatureType.UNIQUE_COLORS},
    "count_specific": {FeatureType.SUIT_COUNT, FeatureType.COLOR_COUNT, FeatureType.RANK_COUNT},
    "count_compare": {FeatureType.SUIT_COUNT, FeatureType.COLOR_COUNT},
    "count_parity": {FeatureType.SUIT_COUNT, FeatureType.COLOR_COUNT},
    "majority": {FeatureType.COLOR_COUNT, FeatureType.SUIT_COUNT, FeatureType.IS_UNIFORM_COLOR},

    # Pattern features
    "sorted": {FeatureType.IS_SORTED},
    "monotonic": {FeatureType.IS_SORTED, FeatureType.RANK_SPREAD},
    "alternating": {FeatureType.UNIQUE_COLORS, FeatureType.IS_UNIFORM_COLOR},
    "palindrome": {FeatureType.IS_PALINDROME_SUITS, FeatureType.IS_PALINDROME_COLORS},
    "periodic": {FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_COLORS},

    # Range/spread features
    "range": {FeatureType.RANK_SPREAD, FeatureType.MAX_RANK, FeatureType.MIN_RANK},
    "spread": {FeatureType.RANK_SPREAD, FeatureType.MAX_RANK, FeatureType.MIN_RANK},

    # Adjacent/sequence features
    "adjacent": {FeatureType.IS_SORTED, FeatureType.RANK_SPREAD},
    "adjacent_constraint": {FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_COLORS, FeatureType.IS_SORTED},
    "runs": {FeatureType.IS_SORTED, FeatureType.RANK_SPREAD},
    "skip": {FeatureType.IS_SORTED, FeatureType.RANK_SPREAD},

    # Positional pairs
    "position_pair": {FeatureType.ENDS_SAME_SUIT, FeatureType.ENDS_SAME_COLOR,
                      FeatureType.FIRST_CARD, FeatureType.LAST_CARD},

    # Shape features (histogram-based)
    "shape": {FeatureType.HAS_PAIR, FeatureType.HAS_TRIPLE, FeatureType.UNIQUE_RANKS},

    # Compositional (Level 5) - accept any relevant feature
    "and_combination": set(FeatureType),  # Any feature could be part of the combo
    "or_combination": set(FeatureType),
    "conditional": set(FeatureType),
    "complex": set(FeatureType),
    "xor": set(FeatureType),
    "triple": set(FeatureType),  # Three-way combinations
    "negation": set(FeatureType),
    "biconditional": set(FeatureType),
}

# Keywords in descriptions that map to features (for text-based matching)
TEXT_TO_FEATURES: Dict[str, Set[FeatureType]] = {
    "same suit": {FeatureType.IS_FLUSH, FeatureType.UNIQUE_SUITS},
    "same color": {FeatureType.IS_UNIFORM_COLOR, FeatureType.UNIQUE_COLORS},
    "pair": {FeatureType.HAS_PAIR},
    "triple": {FeatureType.HAS_TRIPLE},
    "three of a kind": {FeatureType.HAS_TRIPLE},
    "sorted": {FeatureType.IS_SORTED},
    "increasing order": {FeatureType.IS_SORTED},
    "decreasing": {FeatureType.IS_SORTED},
    "first card": {FeatureType.FIRST_CARD},
    "last card": {FeatureType.LAST_CARD},
    "starts with": {FeatureType.FIRST_CARD},
    "ends with": {FeatureType.LAST_CARD},
    "first and last": {FeatureType.ENDS_SAME_SUIT, FeatureType.ENDS_SAME_COLOR},
    "sum": {FeatureType.SUM_RANKS},
    "rank sum": {FeatureType.SUM_RANKS},
    "total": {FeatureType.SUM_RANKS},
    "spread": {FeatureType.RANK_SPREAD},
    "unique": {FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_RANKS, FeatureType.UNIQUE_COLORS},
    "different": {FeatureType.UNIQUE_SUITS, FeatureType.UNIQUE_RANKS, FeatureType.UNIQUE_COLORS},
    "palindrome": {FeatureType.IS_PALINDROME_SUITS, FeatureType.IS_PALINDROME_COLORS},
    "symmetric": {FeatureType.IS_PALINDROME_SUITS, FeatureType.IS_PALINDROME_COLORS},
    "halves": {FeatureType.HALVES_SAME_COLOR},
    "hearts": {FeatureType.SUIT_COUNT, FeatureType.FIRST_CARD, FeatureType.LAST_CARD},
    "diamonds": {FeatureType.SUIT_COUNT, FeatureType.FIRST_CARD, FeatureType.LAST_CARD},
    "clubs": {FeatureType.SUIT_COUNT, FeatureType.FIRST_CARD, FeatureType.LAST_CARD},
    "spades": {FeatureType.SUIT_COUNT, FeatureType.FIRST_CARD, FeatureType.LAST_CARD},
    "red": {FeatureType.COLOR_COUNT, FeatureType.IS_UNIFORM_COLOR},
    "black": {FeatureType.COLOR_COUNT, FeatureType.IS_UNIFORM_COLOR},
    "face card": {FeatureType.MAX_RANK, FeatureType.MIN_RANK},
    "ace": {FeatureType.MAX_RANK, FeatureType.FIRST_CARD, FeatureType.LAST_CARD},
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for the evaluation harness."""
    # Task generation
    seed: int = 42
    n_baseline_samples: int = 10000  # For surprise scoring calibration

    # Train/test split
    dev_fraction: float = 0.2       # Rules for threshold tuning
    test_in_dist_fraction: float = 0.4  # In-distribution test
    test_ood_fraction: float = 0.2  # Out-of-distribution test (Level 5)
    holdout_fraction: float = 0.2   # Reserved for final validation

    # Examples per rule
    n_positive_examples: int = 20
    n_negative_examples: int = 20

    # Evaluation parameters
    top_k_descriptions: int = 5     # Number of descriptions to generate
    n_consistency_trials: int = 5   # Trials for consistency testing

    # Thresholds
    primitive_match_threshold: float = 0.3  # Jaccard threshold for "match"
    description_score_threshold: float = 2.0  # Minimum surprise score


@dataclass
class RuleEvaluation:
    """Evaluation results for a single rule."""
    rule_id: str
    rule_name: str
    level: int
    category: str

    # PRIMARY METRIC: Semantic correctness
    semantic_correctness: float = 0.0   # Does description capture the right semantic feature?

    # Secondary accuracy metrics
    primitive_jaccard: float = 0.0      # Overlap with expected primitives (less important)
    feature_capture_rate: float = 0.0   # Top-k includes expected feature?
    description_score: float = 0.0      # Average surprise score

    # Generated descriptions
    descriptions: List[str] = field(default_factory=list)
    description_primitives: List[List[str]] = field(default_factory=list)
    description_features: List[str] = field(default_factory=list)  # Feature types matched

    # Consistency
    consistency_score: float = 0.0      # Same rule → same descriptions?

    # Metadata
    n_positive_sampled: int = 0
    n_negative_sampled: int = 0
    sampling_success: bool = True


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    config: Dict[str, Any]
    timestamp: str

    # PRIMARY METRIC: Semantic correctness
    overall_semantic_correctness: float = 0.0

    # Summary metrics (secondary)
    overall_accuracy: float = 0.0       # Deprecated: use semantic_correctness
    overall_primitive_jaccard: float = 0.0
    overall_consistency: float = 0.0
    overall_discrimination: float = 0.0

    # By-level metrics
    semantic_correctness_by_level: Dict[int, float] = field(default_factory=dict)
    accuracy_by_level: Dict[int, float] = field(default_factory=dict)
    jaccard_by_level: Dict[int, float] = field(default_factory=dict)

    # By-category metrics
    semantic_correctness_by_category: Dict[str, float] = field(default_factory=dict)
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)

    # Detailed results
    dev_results: List[RuleEvaluation] = field(default_factory=list)
    test_in_dist_results: List[RuleEvaluation] = field(default_factory=list)
    test_ood_results: List[RuleEvaluation] = field(default_factory=list)
    holdout_results: List[RuleEvaluation] = field(default_factory=list)

    # Baseline comparison
    random_baseline_accuracy: float = 0.0
    frequent_baseline_accuracy: float = 0.0


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_primitive_jaccard(
    generated_primitives: List[str],
    expected_primitives: List[str]
) -> float:
    """
    Compute Jaccard similarity between generated and expected primitives.

    Returns value in [0, 1] where 1 = perfect match.
    """
    gen_set = set(p.lower() for p in generated_primitives)
    exp_set = set(p.lower() for p in expected_primitives)

    if not gen_set and not exp_set:
        return 1.0
    if not gen_set or not exp_set:
        return 0.0

    intersection = len(gen_set & exp_set)
    union = len(gen_set | exp_set)

    return intersection / union if union > 0 else 0.0


def compute_feature_capture_rate(
    descriptions: List[Description],
    expected_primitives: List[str],
    top_k: int = 3
) -> float:
    """
    Check if any of the top-k descriptions mention expected primitives.

    Returns 1.0 if at least one expected primitive appears in top-k descriptions,
    0.0 otherwise.
    """
    exp_set = set(p.lower() for p in expected_primitives)

    for desc in descriptions[:top_k]:
        gen_set = set(p.lower() for p in desc.primitives)
        if gen_set & exp_set:
            return 1.0

    return 0.0


def compute_consistency(
    descriptions_trials: List[List[Description]],
    top_k: int = 3
) -> float:
    """
    Compute consistency across multiple trials for the same rule.

    Returns fraction of description pairs that are consistent.
    """
    if len(descriptions_trials) < 2:
        return 1.0

    # Extract top-k description texts from each trial
    trial_texts = []
    for trial in descriptions_trials:
        texts = set(d.text for d in trial[:top_k])
        trial_texts.append(texts)

    # Compute pairwise overlap
    overlaps = []
    for i in range(len(trial_texts)):
        for j in range(i + 1, len(trial_texts)):
            intersection = len(trial_texts[i] & trial_texts[j])
            union = len(trial_texts[i] | trial_texts[j])
            if union > 0:
                overlaps.append(intersection / union)

    return statistics.mean(overlaps) if overlaps else 1.0


def compute_discrimination(
    rule_descriptions: Dict[str, List[str]]
) -> float:
    """
    Compute how well descriptions discriminate between different rules.

    Returns fraction of rule pairs with different descriptions.
    """
    rule_ids = list(rule_descriptions.keys())
    if len(rule_ids) < 2:
        return 1.0

    different_pairs = 0
    total_pairs = 0

    for i in range(len(rule_ids)):
        for j in range(i + 1, len(rule_ids)):
            desc_i = set(rule_descriptions[rule_ids[i]])
            desc_j = set(rule_descriptions[rule_ids[j]])

            # Rules are discriminated if their description sets are different
            if desc_i != desc_j:
                different_pairs += 1
            total_pairs += 1

    return different_pairs / total_pairs if total_pairs > 0 else 1.0


# ============================================================================
# SEMANTIC CORRECTNESS EVALUATION
# ============================================================================

def extract_features_from_description(description: Description) -> Set[FeatureType]:
    """
    Extract feature types from a Description object.

    Uses two methods:
    1. Direct feature extraction from the Description.features list
    2. Text-based keyword matching as fallback

    Returns:
        Set of FeatureType enums that this description relates to
    """
    feature_types: Set[FeatureType] = set()

    # Method 1: Direct feature extraction
    for feature in description.features:
        if hasattr(feature, 'feature_type'):
            feature_types.add(feature.feature_type)

    # Method 2: Text-based keyword matching (fallback/supplement)
    text_lower = description.text.lower()
    for keyword, features in TEXT_TO_FEATURES.items():
        if keyword in text_lower:
            feature_types.update(features)

    return feature_types


def compute_semantic_correctness(
    descriptions: List[Description],
    rule_category: str,
    top_k: int = 3
) -> Tuple[float, List[str]]:
    """
    Compute semantic correctness: does the description capture the right feature?

    The key insight is that we don't require exact primitive name matching.
    Instead, we check if the SEMANTIC CATEGORY of the rule matches the
    FEATURE TYPE extracted by the description generator.

    Args:
        descriptions: List of generated Description objects
        rule_category: The semantic category of the rule (e.g., "uniform", "sorted")
        top_k: Number of top descriptions to consider

    Returns:
        Tuple of (correctness_score, list_of_matched_features)
        - correctness_score: 1.0 if any top-k description captures correct feature, 0.0 otherwise
        - list_of_matched_features: Names of features that matched
    """
    # Get acceptable feature types for this category
    acceptable_features = CATEGORY_TO_FEATURES.get(rule_category, set())

    # If category is unknown, be lenient (return partial credit)
    if not acceptable_features:
        logger.debug(f"Unknown category '{rule_category}', using lenient matching")
        # For unknown categories, accept any non-trivial description
        if descriptions and descriptions[0].score > 1.0:
            return 0.5, ["unknown_category_partial"]
        return 0.0, []

    # Check top-k descriptions for semantic match
    matched_features: List[str] = []

    for desc in descriptions[:top_k]:
        desc_features = extract_features_from_description(desc)

        # Check for intersection with acceptable features
        overlap = desc_features & acceptable_features
        if overlap:
            matched_features.extend([f.value for f in overlap])

    if matched_features:
        return 1.0, matched_features
    return 0.0, []


def compute_semantic_correctness_soft(
    descriptions: List[Description],
    rule_category: str,
    top_k: int = 5
) -> float:
    """
    Soft semantic correctness with partial credit.

    Returns score between 0-1 based on:
    - 1.0: Exact semantic match in top-1
    - 0.8: Exact match in top-2 or top-3
    - 0.5: Exact match in top-4 or top-5
    - 0.3: Related (but not exact) feature in top-3
    - 0.0: No relevant features found
    """
    acceptable_features = CATEGORY_TO_FEATURES.get(rule_category, set())

    if not acceptable_features:
        return 0.5 if descriptions else 0.0

    for rank, desc in enumerate(descriptions[:top_k], 1):
        desc_features = extract_features_from_description(desc)
        overlap = desc_features & acceptable_features

        if overlap:
            if rank == 1:
                return 1.0
            elif rank <= 3:
                return 0.8
            else:
                return 0.5

    # Check for any related features (broader matching)
    all_features_in_top3 = set()
    for desc in descriptions[:3]:
        all_features_in_top3.update(extract_features_from_description(desc))

    if all_features_in_top3:
        # Some features were detected, just not the right ones
        return 0.1

    return 0.0


# ============================================================================
# BASELINE GENERATORS
# ============================================================================

class RandomBaselineGenerator:
    """Baseline that generates random descriptions."""

    def __init__(self, template_pool: List[str]):
        self.templates = template_pool

    def describe_task(
        self,
        positive: List[Hand],
        negative: List[Hand],
        top_k: int = 5
    ) -> List[Description]:
        """Generate random descriptions."""
        descriptions = []
        for _ in range(top_k):
            text = random.choice(self.templates) if self.templates else "random pattern"
            descriptions.append(Description(
                text=text,
                primitives=[],
                score=random.uniform(1.0, 5.0),
                features=[]  # Empty features for baseline
            ))
        return descriptions


class FrequentBaselineGenerator:
    """Baseline that always generates the most common descriptions."""

    def __init__(self, frequent_descriptions: List[str]):
        self.descriptions = frequent_descriptions

    def describe_task(
        self,
        positive: List[Hand],
        negative: List[Hand],
        top_k: int = 5
    ) -> List[Description]:
        """Generate most frequent descriptions."""
        descriptions = []
        for i in range(min(top_k, len(self.descriptions))):
            descriptions.append(Description(
                text=self.descriptions[i],
                primitives=[],
                score=5.0 - i,
                features=[]  # Empty features for baseline
            ))
        return descriptions


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

class DescriptionGeneratorEvaluator:
    """
    Main evaluation harness for the description generator.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        random.seed(config.seed)

        # Initialize components
        self.task_generator = SyntheticTaskGenerator(seed=config.seed)
        self.description_generator = DescriptionGenerator(
            n_baseline_samples=config.n_baseline_samples
        )

        # Storage for evaluation
        self.rules: List[SyntheticRule] = []
        self.dev_rules: List[SyntheticRule] = []
        self.test_in_dist_rules: List[SyntheticRule] = []
        self.test_ood_rules: List[SyntheticRule] = []
        self.holdout_rules: List[SyntheticRule] = []

    def generate_and_split_rules(self) -> Dict[str, int]:
        """
        Generate synthetic rules and split into train/test sets.

        Strategy:
        - Level 5 (compositional) rules go to OOD test set
        - Other levels are split randomly into dev/test_in_dist/holdout
        """
        logger.info("Generating synthetic rules...")
        self.rules = self.task_generator.generate_all()

        # Separate by level for stratified split
        ood_rules = [r for r in self.rules if r.level == 5]
        other_rules = [r for r in self.rules if r.level < 5]

        random.shuffle(ood_rules)
        random.shuffle(other_rules)

        # Split OOD rules
        n_ood = len(ood_rules)
        n_ood_test = int(n_ood * 0.6)  # 60% for testing
        n_ood_holdout = n_ood - n_ood_test  # 40% for holdout

        self.test_ood_rules = ood_rules[:n_ood_test]
        self.holdout_rules = ood_rules[n_ood_test:]

        # Split other rules (levels 1-4)
        n_other = len(other_rules)
        n_dev = int(n_other * self.config.dev_fraction)
        n_test = int(n_other * self.config.test_in_dist_fraction)
        n_holdout_other = n_other - n_dev - n_test

        self.dev_rules = other_rules[:n_dev]
        self.test_in_dist_rules = other_rules[n_dev:n_dev + n_test]
        self.holdout_rules.extend(other_rules[n_dev + n_test:])

        split_info = {
            "total": len(self.rules),
            "dev": len(self.dev_rules),
            "test_in_dist": len(self.test_in_dist_rules),
            "test_ood": len(self.test_ood_rules),
            "holdout": len(self.holdout_rules)
        }

        logger.info(f"Split: {split_info}")
        return split_info

    def evaluate_rule(
        self,
        rule: SyntheticRule,
        n_consistency_trials: int = 1
    ) -> RuleEvaluation:
        """
        Evaluate the description generator on a single rule.
        """
        # Sample examples
        positive = rule.sample_positive(
            self.config.n_positive_examples,
            max_attempts=5000
        )
        negative = rule.sample_negative(
            self.config.n_negative_examples,
            max_attempts=5000
        )

        # Check if sampling succeeded
        sampling_success = len(positive) >= 5 and len(negative) >= 5

        if not sampling_success:
            logger.warning(f"Insufficient samples for {rule.id}: "
                         f"pos={len(positive)}, neg={len(negative)}")
            return RuleEvaluation(
                rule_id=rule.id,
                rule_name=rule.name,
                level=rule.level,
                category=rule.category,
                n_positive_sampled=len(positive),
                n_negative_sampled=len(negative),
                sampling_success=False
            )

        # Generate descriptions
        descriptions = self.description_generator.describe_task(
            positive, negative, top_k=self.config.top_k_descriptions
        )

        # Compute PRIMARY METRIC: Semantic correctness
        semantic_score, matched_features = compute_semantic_correctness(
            descriptions, rule.category, top_k=3
        )

        # Compute secondary metrics
        all_primitives = []
        for desc in descriptions:
            all_primitives.extend(desc.primitives)

        primitive_jaccard = compute_primitive_jaccard(
            all_primitives, rule.expected_primitives
        )

        feature_capture_rate = compute_feature_capture_rate(
            descriptions, rule.expected_primitives, top_k=3
        )

        avg_score = statistics.mean(d.score for d in descriptions) if descriptions else 0.0

        # Consistency testing (if requested)
        consistency_score = 1.0
        if n_consistency_trials > 1:
            trials = []
            for _ in range(n_consistency_trials):
                pos_sample = rule.sample_positive(
                    self.config.n_positive_examples, max_attempts=5000
                )
                neg_sample = rule.sample_negative(
                    self.config.n_negative_examples, max_attempts=5000
                )
                if len(pos_sample) >= 5 and len(neg_sample) >= 5:
                    trial_descs = self.description_generator.describe_task(
                        pos_sample, neg_sample,
                        top_k=self.config.top_k_descriptions
                    )
                    trials.append(trial_descs)

            if len(trials) >= 2:
                consistency_score = compute_consistency(trials)

        return RuleEvaluation(
            rule_id=rule.id,
            rule_name=rule.name,
            level=rule.level,
            category=rule.category,
            semantic_correctness=semantic_score,
            primitive_jaccard=primitive_jaccard,
            feature_capture_rate=feature_capture_rate,
            description_score=avg_score,
            descriptions=[d.text for d in descriptions],
            description_primitives=[d.primitives for d in descriptions],
            description_features=matched_features,
            consistency_score=consistency_score,
            n_positive_sampled=len(positive),
            n_negative_sampled=len(negative),
            sampling_success=True
        )

    def evaluate_rule_set(
        self,
        rules: List[SyntheticRule],
        set_name: str,
        with_consistency: bool = False
    ) -> List[RuleEvaluation]:
        """Evaluate all rules in a set."""
        logger.info(f"Evaluating {set_name} ({len(rules)} rules)...")
        results = []

        for i, rule in enumerate(rules):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(rules)}")

            n_trials = self.config.n_consistency_trials if with_consistency else 1
            result = self.evaluate_rule(rule, n_trials)
            results.append(result)

        return results

    def compute_baselines(
        self,
        test_rules: List[SyntheticRule]
    ) -> Tuple[float, float]:
        """
        Compute baseline accuracies for comparison.

        Returns:
            (random_accuracy, frequent_accuracy)
        """
        logger.info("Computing baselines...")

        # Collect description templates from training
        template_pool = [
            "all cards are the same suit",
            "all cards are the same color",
            "first and last card match",
            "hand is sorted by rank",
            "contains a pair",
            "uniform color pattern",
            "more red than black cards",
            "ranks are in sequence",
        ]

        random_baseline = RandomBaselineGenerator(template_pool)
        frequent_baseline = FrequentBaselineGenerator(template_pool[:5])

        random_hits = 0
        frequent_hits = 0
        n_testable = 0

        for rule in test_rules:
            positive = rule.sample_positive(10, max_attempts=3000)
            negative = rule.sample_negative(10, max_attempts=3000)

            if len(positive) < 5 or len(negative) < 5:
                continue

            n_testable += 1

            # Random baseline
            random_descs = random_baseline.describe_task(positive, negative)
            random_prims = []
            for d in random_descs:
                random_prims.extend(d.primitives)
            if compute_primitive_jaccard(random_prims, rule.expected_primitives) >= 0.1:
                random_hits += 1

            # Frequent baseline (just check if any match)
            frequent_descs = frequent_baseline.describe_task(positive, negative)
            # These have no primitives, so we do text matching
            for desc in frequent_descs:
                if any(p.lower() in desc.text.lower() for p in rule.expected_primitives[:2]):
                    frequent_hits += 1
                    break

        random_acc = random_hits / n_testable if n_testable > 0 else 0.0
        frequent_acc = frequent_hits / n_testable if n_testable > 0 else 0.0

        return random_acc, frequent_acc

    def run_evaluation(
        self,
        include_holdout: bool = False,
        quick_mode: bool = False
    ) -> EvaluationResults:
        """
        Run the full evaluation pipeline.

        Args:
            include_holdout: Whether to evaluate on holdout set (for final validation)
            quick_mode: Use fewer rules for faster testing
        """
        logger.info("=" * 60)
        logger.info("DESCRIPTION GENERATOR EVALUATION")
        logger.info("=" * 60)

        # Generate and split rules
        split_info = self.generate_and_split_rules()

        if quick_mode:
            # Use smaller subsets for quick testing
            self.dev_rules = self.dev_rules[:10]
            self.test_in_dist_rules = self.test_in_dist_rules[:15]
            self.test_ood_rules = self.test_ood_rules[:10]
            self.holdout_rules = self.holdout_rules[:5]
            logger.info("Quick mode: using reduced rule sets")

        # Initialize results
        results = EvaluationResults(
            config=asdict(self.config),
            timestamp=datetime.now().isoformat()
        )

        # Evaluate dev set (with consistency)
        results.dev_results = self.evaluate_rule_set(
            self.dev_rules, "dev", with_consistency=True
        )

        # Evaluate in-distribution test set
        results.test_in_dist_results = self.evaluate_rule_set(
            self.test_in_dist_rules, "test_in_dist", with_consistency=False
        )

        # Evaluate OOD test set
        results.test_ood_results = self.evaluate_rule_set(
            self.test_ood_rules, "test_ood", with_consistency=False
        )

        # Evaluate holdout (if requested)
        if include_holdout:
            results.holdout_results = self.evaluate_rule_set(
                self.holdout_rules, "holdout", with_consistency=False
            )

        # Compute baselines
        all_test = results.test_in_dist_results + results.test_ood_results
        test_rules = self.test_in_dist_rules + self.test_ood_rules
        results.random_baseline_accuracy, results.frequent_baseline_accuracy = \
            self.compute_baselines(test_rules[:20])  # Use subset for speed

        # Aggregate metrics
        self._compute_aggregate_metrics(results)

        logger.info("Evaluation complete!")
        return results

    def _compute_aggregate_metrics(self, results: EvaluationResults):
        """Compute aggregate metrics from individual results."""

        all_results = (
            results.dev_results +
            results.test_in_dist_results +
            results.test_ood_results
        )

        # Filter to successful evaluations
        valid_results = [r for r in all_results if r.sampling_success]

        if not valid_results:
            logger.warning("No valid evaluation results!")
            return

        # PRIMARY METRIC: Semantic correctness
        results.overall_semantic_correctness = statistics.mean(
            r.semantic_correctness for r in valid_results
        )

        # Secondary overall metrics
        results.overall_accuracy = statistics.mean(
            r.feature_capture_rate for r in valid_results
        )
        results.overall_primitive_jaccard = statistics.mean(
            r.primitive_jaccard for r in valid_results
        )
        results.overall_consistency = statistics.mean(
            r.consistency_score for r in valid_results
        )

        # Discrimination (for test sets only)
        rule_descriptions = {
            r.rule_id: r.descriptions
            for r in valid_results if r.descriptions
        }
        results.overall_discrimination = compute_discrimination(rule_descriptions)

        # By-level metrics
        by_level = defaultdict(list)
        for r in valid_results:
            by_level[r.level].append(r)

        for level, level_results in by_level.items():
            results.semantic_correctness_by_level[level] = statistics.mean(
                r.semantic_correctness for r in level_results
            )
            results.accuracy_by_level[level] = statistics.mean(
                r.feature_capture_rate for r in level_results
            )
            results.jaccard_by_level[level] = statistics.mean(
                r.primitive_jaccard for r in level_results
            )

        # By-category metrics
        by_category = defaultdict(list)
        for r in valid_results:
            by_category[r.category].append(r)

        for cat, cat_results in by_category.items():
            results.semantic_correctness_by_category[cat] = statistics.mean(
                r.semantic_correctness for r in cat_results
            )
            results.accuracy_by_category[cat] = statistics.mean(
                r.feature_capture_rate for r in cat_results
            )


# ============================================================================
# REPORTING
# ============================================================================

def print_report(results: EvaluationResults):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("DESCRIPTION GENERATOR EVALUATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {results.timestamp}")
    print()

    print("-" * 40)
    print("PRIMARY METRIC: SEMANTIC CORRECTNESS")
    print("-" * 40)
    print(f"Overall Semantic Correctness: {results.overall_semantic_correctness:.1%}")
    print()
    print("This measures whether the generated description captures the")
    print("correct SEMANTIC FEATURE (not primitive name matching).")
    print()

    print("-" * 40)
    print("SECONDARY METRICS")
    print("-" * 40)
    print(f"Feature Capture Rate (FCR@3): {results.overall_accuracy:.1%}")
    print(f"Primitive Jaccard Similarity: {results.overall_primitive_jaccard:.1%}")
    print(f"Consistency Score:            {results.overall_consistency:.1%}")
    print(f"Discrimination Score:         {results.overall_discrimination:.1%}")
    print()

    print("-" * 40)
    print("BASELINE COMPARISON")
    print("-" * 40)
    print(f"Random Baseline:    {results.random_baseline_accuracy:.1%}")
    print(f"Frequent Baseline:  {results.frequent_baseline_accuracy:.1%}")
    print(f"Our Generator:      {results.overall_semantic_correctness:.1%}")
    improvement = results.overall_semantic_correctness - results.random_baseline_accuracy
    print(f"Improvement over random: +{improvement:.1%}")
    print()

    print("-" * 40)
    print("SEMANTIC CORRECTNESS BY LEVEL")
    print("-" * 40)
    for level in sorted(results.semantic_correctness_by_level.keys()):
        sem = results.semantic_correctness_by_level.get(level, 0)
        fcr = results.accuracy_by_level.get(level, 0)
        print(f"Level {level}: Semantic={sem:.1%}, FCR={fcr:.1%}")
    print()

    print("-" * 40)
    print("SET SUMMARY")
    print("-" * 40)

    def summarize_set(name: str, rule_results: List[RuleEvaluation]):
        if not rule_results:
            print(f"{name}: No results")
            return
        valid = [r for r in rule_results if r.sampling_success]
        sem = statistics.mean(r.semantic_correctness for r in valid) if valid else 0
        fcr = statistics.mean(r.feature_capture_rate for r in valid) if valid else 0
        print(f"{name}:")
        print(f"  Rules: {len(rule_results)} ({len(valid)} valid)")
        print(f"  Semantic Correctness: {sem:.1%}")
        print(f"  FCR@3: {fcr:.1%}")

    summarize_set("Dev Set", results.dev_results)
    summarize_set("Test In-Dist", results.test_in_dist_results)
    summarize_set("Test OOD (Level 5)", results.test_ood_results)
    if results.holdout_results:
        summarize_set("Holdout", results.holdout_results)

    print()
    print("-" * 40)
    print("SAMPLE RESULTS")
    print("-" * 40)

    all_results = (results.dev_results + results.test_in_dist_results +
                  results.test_ood_results)

    # Show some successes and failures based on SEMANTIC CORRECTNESS
    successes = [r for r in all_results if r.semantic_correctness >= 0.5 and r.sampling_success]
    failures = [r for r in all_results if r.semantic_correctness < 0.5 and r.sampling_success]

    print("\nSemantically Correct Descriptions (sample):")
    for r in successes[:3]:
        print(f"  {r.rule_name} [Level {r.level}] - Category: {r.category}")
        print(f"    Matched Features: {r.description_features[:3] if r.description_features else 'N/A'}")
        print(f"    Generated: {r.descriptions[0][:60] if r.descriptions else 'N/A'}...")

    print("\nSemantically Incorrect Descriptions (sample):")
    for r in failures[:3]:
        print(f"  {r.rule_name} [Level {r.level}] - Category: {r.category}")
        print(f"    Expected Features: {CATEGORY_TO_FEATURES.get(r.category, set())}")
        print(f"    Generated: {r.descriptions[0][:60] if r.descriptions else 'N/A'}...")

    print("\n" + "=" * 70)


def save_results(results: EvaluationResults, output_path: str):
    """Save results to JSON file."""
    # Convert to serializable dict
    results_dict = {
        "config": results.config,
        "timestamp": results.timestamp,
        "primary_metric": {
            "semantic_correctness": results.overall_semantic_correctness,
        },
        "overall_metrics": {
            "semantic_correctness": results.overall_semantic_correctness,
            "accuracy": results.overall_accuracy,
            "primitive_jaccard": results.overall_primitive_jaccard,
            "consistency": results.overall_consistency,
            "discrimination": results.overall_discrimination,
        },
        "baselines": {
            "random": results.random_baseline_accuracy,
            "frequent": results.frequent_baseline_accuracy,
        },
        "by_level": {
            "semantic_correctness": results.semantic_correctness_by_level,
            "accuracy": results.accuracy_by_level,
            "jaccard": results.jaccard_by_level,
        },
        "by_category": {
            "semantic_correctness": results.semantic_correctness_by_category,
            "accuracy": results.accuracy_by_category,
        },
        "dev_results": [asdict(r) for r in results.dev_results],
        "test_in_dist_results": [asdict(r) for r in results.test_in_dist_results],
        "test_ood_results": [asdict(r) for r in results.test_ood_results],
        "holdout_results": [asdict(r) for r in results.holdout_results],
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {output_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the description generator on synthetic rules"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode with reduced rule sets"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--include-holdout", action="store_true",
        help="Include holdout set in evaluation (for final validation only)"
    )

    args = parser.parse_args()

    # Configure
    config = EvaluationConfig(seed=args.seed)

    if args.quick:
        config.n_baseline_samples = 2000
        config.n_positive_examples = 10
        config.n_negative_examples = 10
        config.n_consistency_trials = 2

    # Run evaluation
    evaluator = DescriptionGeneratorEvaluator(config)
    results = evaluator.run_evaluation(
        include_holdout=args.include_holdout,
        quick_mode=args.quick
    )

    # Report
    print_report(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "results_description_eval"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"eval_{timestamp}.json"
        save_results(results, str(output_path))


if __name__ == "__main__":
    main()
