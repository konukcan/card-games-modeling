#!/usr/bin/env python3
"""
Description Generator System for Card Game Learning

This module generates human-readable descriptions for hands and tasks that:
1. Map to primitive patterns (guiding synthesis)
2. Highlight surprising/informative features
3. Support both hand-level and task-level description

The system integrates with the existing DreamCoder primitive library.

Architecture:
- FeatureExtractor: Extracts structured features from hands/tasks
- SurpriseScorer: Computes informativeness of features vs baseline
- DescriptionVocabulary: Maps features to natural language phrases
- DescriptionGenerator: End-to-end pipeline producing ranked descriptions
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import math
import random

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES,
    card_color, sample_hand, hand_to_string
)


# ============================================================================
# FEATURE TYPES
# ============================================================================

class FeatureType(Enum):
    """Categories of features we extract."""
    # Hand-level atomic features
    SUIT_COUNT = "suit_count"
    COLOR_COUNT = "color_count"
    RANK_COUNT = "rank_count"
    UNIQUE_SUITS = "unique_suits"
    UNIQUE_RANKS = "unique_ranks"
    UNIQUE_COLORS = "unique_colors"

    # Positional features
    FIRST_CARD = "first_card"
    LAST_CARD = "last_card"
    FIRST_HALF = "first_half"
    SECOND_HALF = "second_half"

    # Pattern features
    HAS_PAIR = "has_pair"
    HAS_TRIPLE = "has_triple"
    IS_SORTED = "is_sorted"
    IS_FLUSH = "is_flush"
    IS_UNIFORM_COLOR = "is_uniform_color"

    # Aggregate features
    SUM_RANKS = "sum_ranks"
    MAX_RANK = "max_rank"
    MIN_RANK = "min_rank"
    RANK_SPREAD = "rank_spread"

    # Relational features
    HALVES_SAME_COLOR = "halves_same_color"
    HALVES_SAME_SUITS = "halves_same_suits"
    ENDS_SAME_SUIT = "ends_same_suit"
    ENDS_SAME_COLOR = "ends_same_color"
    IS_PALINDROME_SUITS = "is_palindrome_suits"
    IS_PALINDROME_COLORS = "is_palindrome_colors"

    # Comparative (task-level)
    DISTINGUISHING = "distinguishing"


@dataclass
class Feature:
    """A single extracted feature with metadata."""
    feature_type: FeatureType
    value: Any
    raw_value: Any = None  # Original value before normalization
    position: Optional[str] = None  # "hand", "first_half", "second_half", "first", "last"
    confidence: float = 1.0

    def __hash__(self):
        return hash((self.feature_type, str(self.value), self.position))

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        return (self.feature_type == other.feature_type and
                self.value == other.value and
                self.position == other.position)


@dataclass
class FeatureSet:
    """Collection of features extracted from a hand or task."""
    features: List[Feature] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, feature: Feature):
        self.features.append(feature)

    def get_by_type(self, feature_type: FeatureType) -> List[Feature]:
        return [f for f in self.features if f.feature_type == feature_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": [(f.feature_type.value, f.value, f.position) for f in self.features],
            "metadata": self.metadata
        }


# ============================================================================
# MODULE 1: FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Extracts structured features from hands and tasks.

    Features are designed to:
    1. Be computable from primitives in lean_primitives.py
    2. Cover the patterns used in the rule catalogue
    3. Support hierarchical analysis (whole hand, halves, positions)
    """

    def __init__(self):
        self._feature_functions: Dict[FeatureType, Callable] = self._build_feature_functions()

    def _build_feature_functions(self) -> Dict[FeatureType, Callable]:
        """Build mapping from feature types to extraction functions."""
        return {
            FeatureType.SUIT_COUNT: self._extract_suit_counts,
            FeatureType.COLOR_COUNT: self._extract_color_counts,
            FeatureType.UNIQUE_SUITS: self._extract_unique_suits,
            FeatureType.UNIQUE_RANKS: self._extract_unique_ranks,
            FeatureType.UNIQUE_COLORS: self._extract_unique_colors,
            FeatureType.HAS_PAIR: self._extract_has_pair,
            FeatureType.HAS_TRIPLE: self._extract_has_triple,
            FeatureType.IS_SORTED: self._extract_is_sorted,
            FeatureType.IS_FLUSH: self._extract_is_flush,
            FeatureType.IS_UNIFORM_COLOR: self._extract_is_uniform_color,
            FeatureType.SUM_RANKS: self._extract_sum_ranks,
            FeatureType.MAX_RANK: self._extract_max_rank,
            FeatureType.MIN_RANK: self._extract_min_rank,
            FeatureType.RANK_SPREAD: self._extract_rank_spread,
            FeatureType.FIRST_CARD: self._extract_first_card,
            FeatureType.LAST_CARD: self._extract_last_card,
            FeatureType.ENDS_SAME_SUIT: self._extract_ends_same_suit,
            FeatureType.ENDS_SAME_COLOR: self._extract_ends_same_color,
            FeatureType.HALVES_SAME_COLOR: self._extract_halves_same_color,
            FeatureType.IS_PALINDROME_SUITS: self._extract_is_palindrome_suits,
            FeatureType.IS_PALINDROME_COLORS: self._extract_is_palindrome_colors,
        }

    def extract_hand_features(self, hand: Hand) -> FeatureSet:
        """
        Extract all features from a single hand.

        Args:
            hand: List of Card objects

        Returns:
            FeatureSet containing all extracted features
        """
        feature_set = FeatureSet()
        feature_set.metadata["hand_size"] = len(hand)

        for feature_type, extract_fn in self._feature_functions.items():
            try:
                features = extract_fn(hand)
                for feature in features:
                    feature_set.add(feature)
            except Exception as e:
                # Log but don't fail - some features may not apply to all hands
                pass

        # Add half-based features
        if len(hand) >= 4:
            self._extract_half_features(hand, feature_set)

        return feature_set

    def extract_task_features(
        self,
        positive_hands: List[Hand],
        negative_hands: List[Hand]
    ) -> FeatureSet:
        """
        Extract features that distinguish positive from negative hands.

        Args:
            positive_hands: Hands that satisfy the rule
            negative_hands: Hands that don't satisfy the rule

        Returns:
            FeatureSet with distinguishing features
        """
        feature_set = FeatureSet()
        feature_set.metadata["n_positive"] = len(positive_hands)
        feature_set.metadata["n_negative"] = len(negative_hands)

        # Extract features from all hands
        pos_features = [self.extract_hand_features(h) for h in positive_hands]
        neg_features = [self.extract_hand_features(h) for h in negative_hands]

        # Find features that distinguish the groups
        distinguishing = self._find_distinguishing_features(pos_features, neg_features)

        for feature, score in distinguishing:
            feature.feature_type = FeatureType.DISTINGUISHING
            feature.confidence = score
            feature_set.add(feature)

        return feature_set

    def extract_comparative_features(
        self,
        hand: Hand,
        other_hands: List[Hand]
    ) -> FeatureSet:
        """
        Extract features that make this hand different from others.

        Useful for describing what's surprising about a specific hand
        relative to a set of comparison hands.
        """
        feature_set = FeatureSet()
        my_features = self.extract_hand_features(hand)
        other_feature_sets = [self.extract_hand_features(h) for h in other_hands]

        # Count feature occurrences in other hands
        other_feature_counts: Counter = Counter()
        for fs in other_feature_sets:
            for f in fs.features:
                other_feature_counts[(f.feature_type, str(f.value))] += 1

        # Find rare features in this hand
        for f in my_features.features:
            key = (f.feature_type, str(f.value))
            count = other_feature_counts.get(key, 0)
            rarity = 1.0 - (count / len(other_hands)) if other_hands else 0.0
            if rarity > 0.5:  # Feature appears in less than half of others
                f.confidence = rarity
                feature_set.add(f)

        return feature_set

    # -------------------------------------------------------------------------
    # Feature extraction implementations
    # -------------------------------------------------------------------------

    def _extract_suit_counts(self, hand: Hand) -> List[Feature]:
        """Count cards of each suit."""
        counts = Counter(c.suit for c in hand)
        features = []
        for suit, count in counts.items():
            features.append(Feature(
                feature_type=FeatureType.SUIT_COUNT,
                value={"suit": suit.value, "count": count},
                raw_value=(suit, count),
                position="hand"
            ))
        return features

    def _extract_color_counts(self, hand: Hand) -> List[Feature]:
        """Count cards of each color."""
        counts = Counter(card_color(c) for c in hand)
        features = []
        for color, count in counts.items():
            features.append(Feature(
                feature_type=FeatureType.COLOR_COUNT,
                value={"color": color.value, "count": count},
                raw_value=(color, count),
                position="hand"
            ))
        return features

    def _extract_unique_suits(self, hand: Hand) -> List[Feature]:
        """Count unique suits."""
        n = len(set(c.suit for c in hand))
        return [Feature(
            feature_type=FeatureType.UNIQUE_SUITS,
            value=n,
            position="hand"
        )]

    def _extract_unique_ranks(self, hand: Hand) -> List[Feature]:
        """Count unique ranks."""
        n = len(set(c.rank for c in hand))
        return [Feature(
            feature_type=FeatureType.UNIQUE_RANKS,
            value=n,
            position="hand"
        )]

    def _extract_unique_colors(self, hand: Hand) -> List[Feature]:
        """Count unique colors."""
        n = len(set(card_color(c) for c in hand))
        return [Feature(
            feature_type=FeatureType.UNIQUE_COLORS,
            value=n,
            position="hand"
        )]

    def _extract_has_pair(self, hand: Hand) -> List[Feature]:
        """Check if hand has a pair of same ranks."""
        rank_counts = Counter(c.rank for c in hand)
        has_pair = any(count >= 2 for count in rank_counts.values())
        return [Feature(
            feature_type=FeatureType.HAS_PAIR,
            value=has_pair,
            position="hand"
        )]

    def _extract_has_triple(self, hand: Hand) -> List[Feature]:
        """Check if hand has three of a kind."""
        rank_counts = Counter(c.rank for c in hand)
        has_triple = any(count >= 3 for count in rank_counts.values())
        return [Feature(
            feature_type=FeatureType.HAS_TRIPLE,
            value=has_triple,
            position="hand"
        )]

    def _extract_is_sorted(self, hand: Hand) -> List[Feature]:
        """Check if ranks are in non-decreasing order."""
        if len(hand) < 2:
            is_sorted = True
        else:
            vals = [RANK_VALUES[c.rank] for c in hand]
            is_sorted = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        return [Feature(
            feature_type=FeatureType.IS_SORTED,
            value=is_sorted,
            position="hand"
        )]

    def _extract_is_flush(self, hand: Hand) -> List[Feature]:
        """Check if all cards are same suit."""
        is_flush = len(set(c.suit for c in hand)) == 1 if hand else False
        return [Feature(
            feature_type=FeatureType.IS_FLUSH,
            value=is_flush,
            position="hand"
        )]

    def _extract_is_uniform_color(self, hand: Hand) -> List[Feature]:
        """Check if all cards are same color."""
        is_uniform = len(set(card_color(c) for c in hand)) == 1 if hand else False
        return [Feature(
            feature_type=FeatureType.IS_UNIFORM_COLOR,
            value=is_uniform,
            position="hand"
        )]

    def _extract_sum_ranks(self, hand: Hand) -> List[Feature]:
        """Sum of all rank values."""
        total = sum(RANK_VALUES[c.rank] for c in hand)
        return [Feature(
            feature_type=FeatureType.SUM_RANKS,
            value=total,
            position="hand"
        )]

    def _extract_max_rank(self, hand: Hand) -> List[Feature]:
        """Maximum rank value."""
        if not hand:
            return []
        max_val = max(RANK_VALUES[c.rank] for c in hand)
        return [Feature(
            feature_type=FeatureType.MAX_RANK,
            value=max_val,
            position="hand"
        )]

    def _extract_min_rank(self, hand: Hand) -> List[Feature]:
        """Minimum rank value."""
        if not hand:
            return []
        min_val = min(RANK_VALUES[c.rank] for c in hand)
        return [Feature(
            feature_type=FeatureType.MIN_RANK,
            value=min_val,
            position="hand"
        )]

    def _extract_rank_spread(self, hand: Hand) -> List[Feature]:
        """Difference between max and min rank."""
        if not hand:
            return []
        vals = [RANK_VALUES[c.rank] for c in hand]
        spread = max(vals) - min(vals)
        return [Feature(
            feature_type=FeatureType.RANK_SPREAD,
            value=spread,
            position="hand"
        )]

    def _extract_first_card(self, hand: Hand) -> List[Feature]:
        """Properties of first card."""
        if not hand:
            return []
        card = hand[0]
        return [Feature(
            feature_type=FeatureType.FIRST_CARD,
            value={"suit": card.suit.value, "rank": card.rank.value,
                   "color": card_color(card).value},
            position="first"
        )]

    def _extract_last_card(self, hand: Hand) -> List[Feature]:
        """Properties of last card."""
        if not hand:
            return []
        card = hand[-1]
        return [Feature(
            feature_type=FeatureType.LAST_CARD,
            value={"suit": card.suit.value, "rank": card.rank.value,
                   "color": card_color(card).value},
            position="last"
        )]

    def _extract_ends_same_suit(self, hand: Hand) -> List[Feature]:
        """Check if first and last cards have same suit."""
        if len(hand) < 2:
            return []
        same = hand[0].suit == hand[-1].suit
        return [Feature(
            feature_type=FeatureType.ENDS_SAME_SUIT,
            value=same,
            position="hand"
        )]

    def _extract_ends_same_color(self, hand: Hand) -> List[Feature]:
        """Check if first and last cards have same color."""
        if len(hand) < 2:
            return []
        same = card_color(hand[0]) == card_color(hand[-1])
        return [Feature(
            feature_type=FeatureType.ENDS_SAME_COLOR,
            value=same,
            position="hand"
        )]

    def _extract_halves_same_color(self, hand: Hand) -> List[Feature]:
        """Check if both halves have same uniform color status."""
        if len(hand) < 4:
            return []
        k = len(hand) // 2
        left, right = hand[:k], hand[k:]
        left_uniform = len(set(card_color(c) for c in left)) == 1
        right_uniform = len(set(card_color(c) for c in right)) == 1
        return [Feature(
            feature_type=FeatureType.HALVES_SAME_COLOR,
            value=left_uniform == right_uniform,
            position="hand"
        )]

    def _extract_is_palindrome_suits(self, hand: Hand) -> List[Feature]:
        """Check if suit sequence is palindromic."""
        suits = [c.suit for c in hand]
        is_pal = suits == suits[::-1]
        return [Feature(
            feature_type=FeatureType.IS_PALINDROME_SUITS,
            value=is_pal,
            position="hand"
        )]

    def _extract_is_palindrome_colors(self, hand: Hand) -> List[Feature]:
        """Check if color sequence is palindromic."""
        colors = [card_color(c) for c in hand]
        is_pal = colors == colors[::-1]
        return [Feature(
            feature_type=FeatureType.IS_PALINDROME_COLORS,
            value=is_pal,
            position="hand"
        )]

    def _extract_half_features(self, hand: Hand, feature_set: FeatureSet):
        """Extract features for each half of the hand."""
        k = len(hand) // 2
        left, right = hand[:k], hand[k:]

        # Left half features
        for half, name in [(left, "first_half"), (right, "second_half")]:
            # Uniform color in half
            colors = set(card_color(c) for c in half)
            feature_set.add(Feature(
                feature_type=FeatureType.IS_UNIFORM_COLOR,
                value=len(colors) == 1,
                position=name
            ))

            # Unique suits in half
            suits = set(c.suit for c in half)
            feature_set.add(Feature(
                feature_type=FeatureType.UNIQUE_SUITS,
                value=len(suits),
                position=name
            ))

    def _find_distinguishing_features(
        self,
        pos_features: List[FeatureSet],
        neg_features: List[FeatureSet]
    ) -> List[Tuple[Feature, float]]:
        """
        Find features that best distinguish positive from negative examples.

        Uses a simple discriminative score based on feature prevalence.
        """
        # Collect all features
        pos_feature_counts: Counter = Counter()
        neg_feature_counts: Counter = Counter()

        for fs in pos_features:
            for f in fs.features:
                key = (f.feature_type, str(f.value), f.position)
                pos_feature_counts[key] += 1

        for fs in neg_features:
            for f in fs.features:
                key = (f.feature_type, str(f.value), f.position)
                neg_feature_counts[key] += 1

        n_pos = len(pos_features) if pos_features else 1
        n_neg = len(neg_features) if neg_features else 1

        # Score each feature by how discriminative it is
        scored_features = []
        all_keys = set(pos_feature_counts.keys()) | set(neg_feature_counts.keys())

        for key in all_keys:
            pos_rate = pos_feature_counts.get(key, 0) / n_pos
            neg_rate = neg_feature_counts.get(key, 0) / n_neg

            # Discriminative score: high if feature is common in pos, rare in neg
            # (or vice versa)
            disc_score = abs(pos_rate - neg_rate)

            if disc_score > 0.3:  # Threshold for significance
                ftype, val, pos = key
                feature = Feature(
                    feature_type=FeatureType(ftype.value) if isinstance(ftype, FeatureType) else ftype,
                    value=val,
                    position=pos
                )
                scored_features.append((feature, disc_score))

        # Sort by discriminative power
        scored_features.sort(key=lambda x: -x[1])
        return scored_features


# ============================================================================
# MODULE 2: SURPRISE SCORING
# ============================================================================

@dataclass
class BaselineDistribution:
    """
    Baseline distribution over random hands.

    Used to compute how surprising/informative a feature is.
    """
    feature_type: FeatureType
    distribution: Dict[Any, float]  # value -> probability
    mean: Optional[float] = None
    std: Optional[float] = None

    def probability(self, value: Any) -> float:
        """Get probability of a specific value."""
        return self.distribution.get(str(value), 0.0)

    def surprise(self, value: Any) -> float:
        """
        Compute surprise (self-information) of a value.

        surprise = -log2(P(value))
        Higher = more surprising/informative
        """
        p = self.probability(value)
        if p <= 0:
            return 10.0  # Max surprise for unseen values
        return -math.log2(p)


class SurpriseScorer:
    """
    Computes how surprising/informative features are relative to baseline.

    The baseline is computed by sampling many random hands and computing
    the empirical distribution of each feature.
    """

    def __init__(self, n_baseline_samples: int = 10000, hand_size: int = 6):
        self.n_samples = n_baseline_samples
        self.hand_size = hand_size
        self.extractor = FeatureExtractor()
        self.baselines: Dict[FeatureType, BaselineDistribution] = {}
        self._build_baselines()

    def _build_baselines(self):
        """Build baseline distributions by sampling random hands."""
        feature_values: Dict[FeatureType, List[Any]] = {}

        for _ in range(self.n_samples):
            hand = sample_hand(self.hand_size)
            features = self.extractor.extract_hand_features(hand)

            for f in features.features:
                if f.position == "hand":  # Only use full-hand features for baseline
                    if f.feature_type not in feature_values:
                        feature_values[f.feature_type] = []
                    feature_values[f.feature_type].append(str(f.value))

        # Build distributions
        for ftype, values in feature_values.items():
            counts = Counter(values)
            total = len(values)
            distribution = {v: c / total for v, c in counts.items()}

            # Compute mean/std for numeric features
            mean, std = None, None
            try:
                numeric_vals = [float(v) if v not in ['True', 'False'] else (1.0 if v == 'True' else 0.0)
                               for v in values]
                mean = sum(numeric_vals) / len(numeric_vals)
                variance = sum((x - mean) ** 2 for x in numeric_vals) / len(numeric_vals)
                std = math.sqrt(variance)
            except (ValueError, TypeError):
                pass

            self.baselines[ftype] = BaselineDistribution(
                feature_type=ftype,
                distribution=distribution,
                mean=mean,
                std=std
            )

    def score_feature(self, feature: Feature) -> float:
        """
        Score how surprising a feature is.

        Returns:
            Surprise score (higher = more informative)
        """
        if feature.feature_type not in self.baselines:
            return 0.0

        baseline = self.baselines[feature.feature_type]
        return baseline.surprise(feature.value)

    def score_feature_set(self, feature_set: FeatureSet) -> List[Tuple[Feature, float]]:
        """
        Score all features in a set.

        Returns:
            List of (feature, surprise_score) tuples, sorted by score descending
        """
        scored = [(f, self.score_feature(f)) for f in feature_set.features]
        scored.sort(key=lambda x: -x[1])
        return scored

    def get_top_surprising(
        self,
        feature_set: FeatureSet,
        top_k: int = 5
    ) -> List[Tuple[Feature, float]]:
        """Get the top-k most surprising features."""
        scored = self.score_feature_set(feature_set)
        return scored[:top_k]


# ============================================================================
# MODULE 3: DESCRIPTION VOCABULARY
# ============================================================================

@dataclass
class DescriptionTemplate:
    """
    A template for generating natural language descriptions.

    Maps feature patterns to human-readable phrases with primitive hints.
    """
    template_id: str
    feature_type: FeatureType
    condition: Callable[[Feature], bool]  # When this template applies
    template: str  # Template string with {placeholders}
    primitives: List[str]  # Related primitives for synthesis guidance
    priority: int = 0  # Higher = prefer this template

    def matches(self, feature: Feature) -> bool:
        """Check if this template matches a feature."""
        if feature.feature_type != self.feature_type:
            return False
        return self.condition(feature)

    def render(self, feature: Feature) -> str:
        """Render the template with feature values."""
        if isinstance(feature.value, dict):
            return self.template.format(**feature.value)
        return self.template.format(value=feature.value)


class DescriptionVocabulary:
    """
    Vocabulary of descriptive phrases that map to primitives.

    The vocabulary is designed to:
    1. Be human-readable and brief
    2. Map directly to primitive operations
    3. Support compositional templates
    """

    def __init__(self):
        self.templates: List[DescriptionTemplate] = self._build_templates()
        self.compositional_templates = self._build_compositional_templates()

    def _build_templates(self) -> List[DescriptionTemplate]:
        """Build the vocabulary of description templates."""
        templates = []

        # ----- Suit count templates -----
        templates.append(DescriptionTemplate(
            template_id="all_same_suit",
            feature_type=FeatureType.IS_FLUSH,
            condition=lambda f: f.value == True,
            template="all cards are the same suit",
            primitives=["all_same_suit"],
            priority=10
        ))

        templates.append(DescriptionTemplate(
            template_id="suit_count_majority",
            feature_type=FeatureType.SUIT_COUNT,
            condition=lambda f: isinstance(f.value, dict) and f.value.get("count", 0) >= 3,
            template="has {count} {suit}",
            primitives=["count_suit", "has_suit"],
            priority=5
        ))

        templates.append(DescriptionTemplate(
            template_id="exactly_one_suit",
            feature_type=FeatureType.SUIT_COUNT,
            condition=lambda f: isinstance(f.value, dict) and f.value.get("count") == 1,
            template="has exactly one {suit}",
            primitives=["count_suit", "eq", "1"],
            priority=4
        ))

        # ----- Color templates -----
        templates.append(DescriptionTemplate(
            template_id="all_same_color",
            feature_type=FeatureType.IS_UNIFORM_COLOR,
            condition=lambda f: f.value == True,
            template="all cards are the same color",
            primitives=["all_same_color"],
            priority=10
        ))

        templates.append(DescriptionTemplate(
            template_id="color_balance",
            feature_type=FeatureType.COLOR_COUNT,
            condition=lambda f: isinstance(f.value, dict),
            template="{count} {color} cards",
            primitives=["count_color"],
            priority=3
        ))

        # ----- Unique counts -----
        templates.append(DescriptionTemplate(
            template_id="n_suits",
            feature_type=FeatureType.UNIQUE_SUITS,
            condition=lambda f: True,
            template="has {value} different suits",
            primitives=["n_unique_suits"],
            priority=6
        ))

        templates.append(DescriptionTemplate(
            template_id="n_ranks",
            feature_type=FeatureType.UNIQUE_RANKS,
            condition=lambda f: True,
            template="has {value} different ranks",
            primitives=["n_unique_ranks"],
            priority=5
        ))

        # ----- Pair/Triple -----
        templates.append(DescriptionTemplate(
            template_id="has_pair",
            feature_type=FeatureType.HAS_PAIR,
            condition=lambda f: f.value == True,
            template="contains a pair (same rank)",
            primitives=["has_pair", "n_unique_ranks", "lt", "length"],
            priority=8
        ))

        templates.append(DescriptionTemplate(
            template_id="no_pair",
            feature_type=FeatureType.HAS_PAIR,
            condition=lambda f: f.value == False,
            template="no repeated ranks",
            primitives=["n_unique_ranks", "eq", "length"],
            priority=7
        ))

        templates.append(DescriptionTemplate(
            template_id="has_triple",
            feature_type=FeatureType.HAS_TRIPLE,
            condition=lambda f: f.value == True,
            template="contains three of a kind",
            primitives=["count", "ge", "3"],
            priority=9
        ))

        # ----- Sorted -----
        templates.append(DescriptionTemplate(
            template_id="is_sorted",
            feature_type=FeatureType.IS_SORTED,
            condition=lambda f: f.value == True,
            template="ranks are in increasing order",
            primitives=["is_sorted", "map", "rank_val", "all", "le"],
            priority=10
        ))

        templates.append(DescriptionTemplate(
            template_id="not_sorted",
            feature_type=FeatureType.IS_SORTED,
            condition=lambda f: f.value == False,
            template="ranks are not sorted",
            primitives=["not", "is_sorted"],
            priority=3
        ))

        # ----- Positional (first/last) -----
        templates.append(DescriptionTemplate(
            template_id="first_card_suit",
            feature_type=FeatureType.FIRST_CARD,
            condition=lambda f: isinstance(f.value, dict) and "suit" in f.value,
            template="starts with a {suit}",
            primitives=["head", "get_suit", "eq"],
            priority=6
        ))

        templates.append(DescriptionTemplate(
            template_id="first_card_color",
            feature_type=FeatureType.FIRST_CARD,
            condition=lambda f: isinstance(f.value, dict) and "color" in f.value,
            template="first card is {color}",
            primitives=["head", "get_color", "eq"],
            priority=5
        ))

        templates.append(DescriptionTemplate(
            template_id="last_card_suit",
            feature_type=FeatureType.LAST_CARD,
            condition=lambda f: isinstance(f.value, dict) and "suit" in f.value,
            template="ends with a {suit}",
            primitives=["last", "get_suit", "eq"],
            priority=6
        ))

        templates.append(DescriptionTemplate(
            template_id="ends_same_suit",
            feature_type=FeatureType.ENDS_SAME_SUIT,
            condition=lambda f: f.value == True,
            template="first and last cards share the same suit",
            primitives=["head", "last", "get_suit", "eq"],
            priority=8
        ))

        templates.append(DescriptionTemplate(
            template_id="ends_same_color",
            feature_type=FeatureType.ENDS_SAME_COLOR,
            condition=lambda f: f.value == True,
            template="first and last cards are the same color",
            primitives=["head", "last", "get_color", "eq"],
            priority=7
        ))

        # ----- Palindrome -----
        templates.append(DescriptionTemplate(
            template_id="palindrome_suits",
            feature_type=FeatureType.IS_PALINDROME_SUITS,
            condition=lambda f: f.value == True,
            template="suit pattern is symmetric (palindrome)",
            primitives=["map", "get_suit", "reverse", "eq", "zip_with"],
            priority=9
        ))

        templates.append(DescriptionTemplate(
            template_id="palindrome_colors",
            feature_type=FeatureType.IS_PALINDROME_COLORS,
            condition=lambda f: f.value == True,
            template="color pattern is symmetric (palindrome)",
            primitives=["map", "get_color", "reverse", "eq", "zip_with"],
            priority=8
        ))

        # ----- Aggregate ranks -----
        templates.append(DescriptionTemplate(
            template_id="sum_ranks_high",
            feature_type=FeatureType.SUM_RANKS,
            condition=lambda f: isinstance(f.value, (int, float)) and f.value > 50,
            template="total rank sum is high ({value})",
            primitives=["sum_ranks", "gt"],
            priority=4
        ))

        templates.append(DescriptionTemplate(
            template_id="sum_ranks_low",
            feature_type=FeatureType.SUM_RANKS,
            condition=lambda f: isinstance(f.value, (int, float)) and f.value < 30,
            template="total rank sum is low ({value})",
            primitives=["sum_ranks", "lt"],
            priority=4
        ))

        templates.append(DescriptionTemplate(
            template_id="max_rank_face",
            feature_type=FeatureType.MAX_RANK,
            condition=lambda f: isinstance(f.value, (int, float)) and f.value >= 11,
            template="highest card is a face card or ace",
            primitives=["max_rank", "ge", "11"],
            priority=5
        ))

        templates.append(DescriptionTemplate(
            template_id="rank_spread_narrow",
            feature_type=FeatureType.RANK_SPREAD,
            condition=lambda f: isinstance(f.value, (int, float)) and f.value <= 3,
            template="ranks are clustered (spread of {value})",
            primitives=["max_rank", "min_rank", "-", "le"],
            priority=6
        ))

        templates.append(DescriptionTemplate(
            template_id="rank_spread_wide",
            feature_type=FeatureType.RANK_SPREAD,
            condition=lambda f: isinstance(f.value, (int, float)) and f.value >= 10,
            template="ranks span widely (spread of {value})",
            primitives=["max_rank", "min_rank", "-", "ge"],
            priority=5
        ))

        # ----- Half-based features -----
        templates.append(DescriptionTemplate(
            template_id="halves_color_match",
            feature_type=FeatureType.HALVES_SAME_COLOR,
            condition=lambda f: f.value == True,
            template="both halves have matching color uniformity",
            primitives=["first_half", "second_half", "all_same_color", "eq"],
            priority=7
        ))

        return templates

    def _build_compositional_templates(self) -> Dict[str, str]:
        """
        Build templates for compositional descriptions.

        These combine multiple features into coherent descriptions.
        """
        return {
            "AND": "{desc1} AND {desc2}",
            "BUT": "{desc1}, BUT {desc2}",
            "WHILE": "{desc1} WHILE {desc2}",
            "LEFT_HALF": "the left half {desc}",
            "RIGHT_HALF": "the right half {desc}",
            "BOTH_HALVES": "both halves {desc}",
            "FIRST_CARD": "the first card {desc}",
            "LAST_CARD": "the last card {desc}",
            "NEGATION": "NOT {desc}",
        }

    def find_matching_templates(self, feature: Feature) -> List[DescriptionTemplate]:
        """Find all templates that match a given feature."""
        matches = [t for t in self.templates if t.matches(feature)]
        matches.sort(key=lambda t: -t.priority)
        return matches

    def describe_feature(self, feature: Feature) -> Optional[str]:
        """Generate a description for a single feature."""
        templates = self.find_matching_templates(feature)
        if not templates:
            return None
        return templates[0].render(feature)

    def get_primitives_for_feature(self, feature: Feature) -> List[str]:
        """Get the primitives associated with a feature's best template."""
        templates = self.find_matching_templates(feature)
        if not templates:
            return []
        return templates[0].primitives

    def compose_descriptions(
        self,
        descriptions: List[str],
        composition_type: str = "AND"
    ) -> str:
        """Compose multiple descriptions into one."""
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]

        template = self.compositional_templates.get(composition_type, "{desc1} AND {desc2}")

        # Build composite description
        result = descriptions[0]
        for desc in descriptions[1:]:
            result = template.format(desc1=result, desc2=desc)

        return result


# ============================================================================
# MODULE 4: DESCRIPTION GENERATION PIPELINE
# ============================================================================

@dataclass
class Description:
    """A generated description with metadata."""
    text: str
    score: float  # Combined surprise + confidence score
    primitives: List[str]  # Primitives that this description relates to
    features: List[Feature]  # Features that contributed

    def __str__(self):
        return self.text


class DescriptionGenerator:
    """
    End-to-end pipeline for generating descriptions.

    Pipeline:
    1. Extract features from hand/task
    2. Score features by surprise/informativeness
    3. Select top features
    4. Map to vocabulary descriptions
    5. Return ranked list of descriptions
    """

    def __init__(
        self,
        n_baseline_samples: int = 5000,
        hand_size: int = 6,
        top_k_features: int = 5
    ):
        self.extractor = FeatureExtractor()
        self.scorer = SurpriseScorer(n_baseline_samples, hand_size)
        self.vocabulary = DescriptionVocabulary()
        self.top_k = top_k_features

    def describe_hand(self, hand: Hand, top_k: Optional[int] = None) -> List[Description]:
        """
        Generate descriptions for a single hand.

        Args:
            hand: List of Card objects
            top_k: Number of descriptions to return (default: self.top_k)

        Returns:
            List of Description objects, sorted by score
        """
        if top_k is None:
            top_k = self.top_k

        # Step 1: Extract features
        features = self.extractor.extract_hand_features(hand)

        # Step 2: Score by surprise
        scored_features = self.scorer.score_feature_set(features)

        # Step 3-4: Map to descriptions
        descriptions = []
        seen_texts = set()

        for feature, surprise_score in scored_features:
            text = self.vocabulary.describe_feature(feature)
            if text and text not in seen_texts:
                seen_texts.add(text)
                primitives = self.vocabulary.get_primitives_for_feature(feature)
                descriptions.append(Description(
                    text=text,
                    score=surprise_score * feature.confidence,
                    primitives=primitives,
                    features=[feature]
                ))

        # Sort by score and return top k
        descriptions.sort(key=lambda d: -d.score)
        return descriptions[:top_k]

    def describe_task(
        self,
        positive_hands: List[Hand],
        negative_hands: List[Hand],
        top_k: Optional[int] = None
    ) -> List[Description]:
        """
        Generate descriptions that distinguish positive from negative hands.

        Args:
            positive_hands: Hands that satisfy the rule
            negative_hands: Hands that don't satisfy the rule
            top_k: Number of descriptions to return

        Returns:
            List of Description objects describing what distinguishes the groups
        """
        if top_k is None:
            top_k = self.top_k

        # Extract distinguishing features
        task_features = self.extractor.extract_task_features(positive_hands, negative_hands)

        # Generate descriptions for distinguishing features
        descriptions = []
        seen_texts = set()

        # Also analyze patterns in positive hands
        pos_patterns = self._find_common_patterns(positive_hands)
        neg_patterns = self._find_common_patterns(negative_hands)

        # Describe what positive hands have that negative don't
        for pattern, prevalence in pos_patterns.items():
            neg_prev = neg_patterns.get(pattern, 0.0)
            if prevalence > 0.7 and neg_prev < 0.3:
                text = f"winning hands often: {pattern}"
                if text not in seen_texts:
                    seen_texts.add(text)
                    descriptions.append(Description(
                        text=text,
                        score=prevalence - neg_prev,
                        primitives=[],
                        features=[]
                    ))

        # Describe what negative hands have that positive don't
        for pattern, prevalence in neg_patterns.items():
            pos_prev = pos_patterns.get(pattern, 0.0)
            if prevalence > 0.7 and pos_prev < 0.3:
                text = f"losing hands often: {pattern}"
                if text not in seen_texts:
                    seen_texts.add(text)
                    descriptions.append(Description(
                        text=text,
                        score=prevalence - pos_prev,
                        primitives=[],
                        features=[]
                    ))

        descriptions.sort(key=lambda d: -d.score)
        return descriptions[:top_k]

    def describe_why_surprising(
        self,
        hand: Hand,
        comparison_hands: List[Hand],
        top_k: Optional[int] = None
    ) -> List[Description]:
        """
        Describe what makes this hand different from a set of comparison hands.

        Args:
            hand: The hand to describe
            comparison_hands: Hands to compare against
            top_k: Number of descriptions to return

        Returns:
            List of descriptions of what's unusual about this hand
        """
        if top_k is None:
            top_k = self.top_k

        # Get comparative features
        comparative = self.extractor.extract_comparative_features(hand, comparison_hands)

        descriptions = []
        seen_texts = set()

        for feature in comparative.features:
            text = self.vocabulary.describe_feature(feature)
            if text and text not in seen_texts:
                prefix = "unusually, " if feature.confidence > 0.8 else ""
                full_text = prefix + text
                seen_texts.add(full_text)
                primitives = self.vocabulary.get_primitives_for_feature(feature)
                descriptions.append(Description(
                    text=full_text,
                    score=feature.confidence,
                    primitives=primitives,
                    features=[feature]
                ))

        descriptions.sort(key=lambda d: -d.score)
        return descriptions[:top_k]

    def _find_common_patterns(self, hands: List[Hand]) -> Dict[str, float]:
        """Find patterns common across a set of hands."""
        if not hands:
            return {}

        pattern_counts: Counter = Counter()

        for hand in hands:
            features = self.extractor.extract_hand_features(hand)
            for feature in features.features:
                text = self.vocabulary.describe_feature(feature)
                if text:
                    pattern_counts[text] += 1

        # Normalize by number of hands
        return {pattern: count / len(hands) for pattern, count in pattern_counts.items()}


# ============================================================================
# MODULE 5: TRAINING PROCEDURE
# ============================================================================

class SyntheticTrainingDataGenerator:
    """
    Generates synthetic training data for training/fine-tuning the description generator.

    Training data consists of:
    - Programs (compositional rules)
    - Hands (positive and negative examples)
    - Labels (what the rule tests)
    - Ground truth descriptions (from rule metadata)
    """

    def __init__(self, hand_size: int = 6, n_examples_per_task: int = 50):
        self.hand_size = hand_size
        self.n_examples = n_examples_per_task
        self.generator = DescriptionGenerator()

    def generate_training_example(
        self,
        rule_predicate: Callable[[Hand], bool],
        rule_description: str,
        rule_primitives: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a single training example for a rule.

        Args:
            rule_predicate: Function that tests the rule (Hand -> bool)
            rule_description: Human-written description of the rule
            rule_primitives: Primitives used in the rule

        Returns:
            Dictionary with training data
        """
        positive_hands = []
        negative_hands = []

        # Sample hands until we have enough of each type
        max_attempts = self.n_examples * 20
        attempts = 0

        while (len(positive_hands) < self.n_examples or
               len(negative_hands) < self.n_examples) and attempts < max_attempts:
            hand = sample_hand(self.hand_size)
            attempts += 1

            if rule_predicate(hand):
                if len(positive_hands) < self.n_examples:
                    positive_hands.append(hand)
            else:
                if len(negative_hands) < self.n_examples:
                    negative_hands.append(hand)

        # Generate descriptions
        generated_descriptions = self.generator.describe_task(
            positive_hands[:10],  # Use subset for efficiency
            negative_hands[:10]
        )

        return {
            "positive_hands": [[str(c) for c in h] for h in positive_hands],
            "negative_hands": [[str(c) for c in h] for h in negative_hands],
            "ground_truth_description": rule_description,
            "ground_truth_primitives": rule_primitives,
            "generated_descriptions": [
                {"text": d.text, "score": d.score, "primitives": d.primitives}
                for d in generated_descriptions
            ],
            "n_positive": len(positive_hands),
            "n_negative": len(negative_hands),
        }

    def generate_training_dataset(
        self,
        rules: List[Tuple[Callable, str, List[str]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate a full training dataset from a list of rules.

        Args:
            rules: List of (predicate, description, primitives) tuples

        Returns:
            List of training examples
        """
        dataset = []
        for predicate, description, primitives in rules:
            example = self.generate_training_example(predicate, description, primitives)
            dataset.append(example)
        return dataset


class DescriptionValidator:
    """
    Validates that generated descriptions are useful for synthesis.

    A description is useful if:
    1. It uniquely identifies the rule among alternatives
    2. It maps to primitives that appear in the rule's program
    3. It is human-readable and concise
    """

    def __init__(self, generator: DescriptionGenerator):
        self.generator = generator

    def validate_description_uniqueness(
        self,
        description: Description,
        target_hands: List[Hand],
        distractor_hands: List[Hand]
    ) -> float:
        """
        Check if the description uniquely identifies target hands.

        Returns:
            Score between 0-1 where 1 means perfect discrimination
        """
        # This would need a semantic matcher to check if hands match description
        # For now, return a placeholder score based on the description's confidence
        return description.score

    def validate_primitive_coverage(
        self,
        description: Description,
        expected_primitives: List[str]
    ) -> float:
        """
        Check how well the description's primitives overlap with expected ones.

        Returns:
            Jaccard similarity between primitive sets
        """
        if not expected_primitives or not description.primitives:
            return 0.0

        desc_prims = set(description.primitives)
        expected_prims = set(expected_primitives)

        intersection = len(desc_prims & expected_prims)
        union = len(desc_prims | expected_prims)

        return intersection / union if union > 0 else 0.0

    def validate_conciseness(self, description: Description, max_words: int = 15) -> bool:
        """Check if description is concise enough."""
        word_count = len(description.text.split())
        return word_count <= max_words


# ============================================================================
# DEMO / TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DESCRIPTION GENERATOR SYSTEM DEMO")
    print("=" * 70)

    # Initialize the generator
    print("\nInitializing description generator (building baselines)...")
    generator = DescriptionGenerator(n_baseline_samples=2000)

    # Test 1: Describe a single hand
    print("\n" + "-" * 50)
    print("TEST 1: Describe a single hand")
    print("-" * 50)

    test_hand = sample_hand(6)
    print(f"Hand: {hand_to_string(test_hand)}")

    descriptions = generator.describe_hand(test_hand, top_k=5)
    print("\nGenerated descriptions:")
    for i, desc in enumerate(descriptions, 1):
        print(f"  {i}. {desc.text} (score: {desc.score:.2f})")
        print(f"     Primitives: {', '.join(desc.primitives)}")

    # Test 2: Create a flush hand and describe it
    print("\n" + "-" * 50)
    print("TEST 2: Describe a flush (all same suit)")
    print("-" * 50)

    flush_hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.HEARTS, Rank.NINE),
    ]
    print(f"Hand: {hand_to_string(flush_hand)}")

    descriptions = generator.describe_hand(flush_hand, top_k=5)
    print("\nGenerated descriptions:")
    for i, desc in enumerate(descriptions, 1):
        print(f"  {i}. {desc.text} (score: {desc.score:.2f})")

    # Test 3: Describe a task (what distinguishes positive from negative)
    print("\n" + "-" * 50)
    print("TEST 3: Describe a task (uniform color rule)")
    print("-" * 50)

    # Rule: all cards same color
    def uniform_color(hand):
        colors = set(card_color(c) for c in hand)
        return len(colors) == 1

    # Generate positive and negative examples
    positive = []
    negative = []
    for _ in range(100):
        h = sample_hand(6)
        if uniform_color(h):
            positive.append(h)
        else:
            negative.append(h)

    print(f"Generated {len(positive)} positive and {len(negative)} negative hands")

    task_descriptions = generator.describe_task(positive[:20], negative[:20], top_k=5)
    print("\nTask descriptions (what distinguishes winning from losing):")
    for i, desc in enumerate(task_descriptions, 1):
        print(f"  {i}. {desc.text} (score: {desc.score:.2f})")

    # Test 4: Describe why a hand is surprising
    print("\n" + "-" * 50)
    print("TEST 4: Describe what's surprising about a hand")
    print("-" * 50)

    # Create a sorted hand (rare)
    sorted_hand = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.HEARTS, Rank.FOUR),
        Card(Suit.DIAMONDS, Rank.SIX),
        Card(Suit.SPADES, Rank.EIGHT),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.CLUBS, Rank.QUEEN),
    ]
    comparison_hands = [sample_hand(6) for _ in range(50)]

    print(f"Target hand: {hand_to_string(sorted_hand)}")

    surprising = generator.describe_why_surprising(sorted_hand, comparison_hands, top_k=5)
    print("\nWhat's unusual about this hand:")
    for i, desc in enumerate(surprising, 1):
        print(f"  {i}. {desc.text} (rarity: {desc.score:.2f})")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
