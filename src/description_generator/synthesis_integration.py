#!/usr/bin/env python3
"""
Synthesis Integration Module

Connects the description generator with DreamCoder program synthesis.
Descriptions are used to:
1. Bias the recognition network toward relevant primitives
2. Provide semantic guidance during enumeration
3. Generate self-explanation prompts for human learners

This module bridges natural language descriptions and program synthesis.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, sample_hand, hand_to_string
from description_generator.description_generator import (
    DescriptionGenerator,
    Description,
    Feature,
    FeatureType
)


# ============================================================================
# PRIMITIVE-DESCRIPTION MAPPING
# ============================================================================

# Comprehensive mapping from primitives to semantic categories
PRIMITIVE_CATEGORIES = {
    # Suit-related primitives
    "suit_access": ["get_suit", "CLUBS", "DIAMONDS", "HEARTS", "SPADES"],
    "suit_counting": ["count_suit", "n_unique_suits", "has_suit", "all_same_suit"],

    # Color-related primitives
    "color_access": ["get_color", "RED", "BLACK"],
    "color_counting": ["count_color", "n_unique_colors", "has_color", "all_same_color"],

    # Rank-related primitives
    "rank_access": ["get_rank", "rank_val"],
    "rank_aggregate": ["sum_ranks", "max_rank", "min_rank"],
    "rank_counting": ["n_unique_ranks"],

    # Position-related primitives
    "position_access": ["head", "last", "at", "first_half", "second_half"],
    "position_structure": ["take", "drop", "reverse", "length", "half_len"],

    # Comparison primitives
    "comparison": ["eq", "lt", "le", "gt", "ge"],
    "boolean": ["and", "or", "not", "if"],

    # Higher-order primitives
    "iteration": ["map", "filter", "all", "any", "unique"],
    "pairing": ["zip_with", "adjacent_pairs"],

    # Arithmetic
    "arithmetic": ["+", "-", "mod"],

    # Constants
    "numeric": ["0", "1", "2", "3", "4", "5", "true", "false"],
}

# Reverse mapping: primitive -> categories
PRIMITIVE_TO_CATEGORIES: Dict[str, List[str]] = {}
for category, primitives in PRIMITIVE_CATEGORIES.items():
    for prim in primitives:
        if prim not in PRIMITIVE_TO_CATEGORIES:
            PRIMITIVE_TO_CATEGORIES[prim] = []
        PRIMITIVE_TO_CATEGORIES[prim].append(category)


# ============================================================================
# DESCRIPTION-TO-PRIMITIVE SCORING
# ============================================================================

@dataclass
class PrimitiveBias:
    """
    Bias scores for primitives based on descriptions.

    Used to adjust recognition network predictions or enumeration ordering.
    """
    primitive: str
    score: float  # Higher = more relevant
    source_descriptions: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"PrimitiveBias({self.primitive}, score={self.score:.2f})"


class DescriptionToPrimitiveBiaser:
    """
    Converts descriptions into bias scores for primitives.

    This enables descriptions to guide program synthesis by adjusting
    the probability of using certain primitives.
    """

    def __init__(self):
        # Build the full list of primitives from categories
        self.all_primitives: Set[str] = set()
        for prims in PRIMITIVE_CATEGORIES.values():
            self.all_primitives.update(prims)

        # Description keyword -> primitive category mapping
        self.keyword_to_category = self._build_keyword_mapping()

    def _build_keyword_mapping(self) -> Dict[str, List[str]]:
        """Map natural language keywords to primitive categories."""
        return {
            # Suit keywords
            "suit": ["suit_access", "suit_counting"],
            "suits": ["suit_access", "suit_counting"],
            "clubs": ["suit_access"],
            "diamonds": ["suit_access"],
            "hearts": ["suit_access"],
            "spades": ["suit_access"],
            "flush": ["suit_counting"],

            # Color keywords
            "color": ["color_access", "color_counting"],
            "colors": ["color_access", "color_counting"],
            "red": ["color_access"],
            "black": ["color_access"],
            "uniform": ["color_counting", "suit_counting"],

            # Rank keywords
            "rank": ["rank_access", "rank_aggregate"],
            "ranks": ["rank_access", "rank_aggregate"],
            "sum": ["rank_aggregate", "arithmetic"],
            "total": ["rank_aggregate", "arithmetic"],
            "highest": ["rank_aggregate", "comparison"],
            "lowest": ["rank_aggregate", "comparison"],
            "face": ["rank_access", "comparison"],
            "spread": ["rank_aggregate", "arithmetic"],

            # Position keywords
            "first": ["position_access"],
            "last": ["position_access"],
            "starts": ["position_access"],
            "ends": ["position_access"],
            "half": ["position_access", "position_structure"],
            "halves": ["position_access", "position_structure"],
            "left": ["position_access", "position_structure"],
            "right": ["position_access", "position_structure"],

            # Pattern keywords
            "sorted": ["comparison", "iteration"],
            "increasing": ["comparison", "iteration"],
            "decreasing": ["comparison", "iteration"],
            "palindrome": ["position_structure", "comparison", "pairing"],
            "symmetric": ["position_structure", "comparison"],
            "pattern": ["iteration", "pairing"],

            # Counting keywords
            "pair": ["rank_counting", "comparison"],
            "triple": ["rank_counting", "comparison"],
            "different": ["rank_counting", "suit_counting", "color_counting"],
            "unique": ["iteration"],
            "count": ["rank_counting", "suit_counting", "color_counting"],
            "exactly": ["comparison", "numeric"],

            # Boolean keywords
            "same": ["comparison"],
            "share": ["comparison"],
            "match": ["comparison"],
            "both": ["boolean", "position_access"],
            "either": ["boolean"],
            "not": ["boolean"],
            "and": ["boolean"],
            "or": ["boolean"],

            # Comparison keywords
            "more": ["comparison"],
            "less": ["comparison"],
            "greater": ["comparison"],
            "fewer": ["comparison"],
            "at least": ["comparison"],
            "at most": ["comparison"],
        }

    def compute_biases(
        self,
        descriptions: List[Description],
        base_score: float = 1.0
    ) -> List[PrimitiveBias]:
        """
        Compute primitive biases from a list of descriptions.

        Args:
            descriptions: List of Description objects
            base_score: Baseline score for all primitives

        Returns:
            List of PrimitiveBias objects sorted by score
        """
        # Initialize scores for all primitives
        scores: Dict[str, float] = {p: base_score for p in self.all_primitives}
        sources: Dict[str, List[str]] = {p: [] for p in self.all_primitives}

        for desc in descriptions:
            # Method 1: Use primitives directly from description
            for prim in desc.primitives:
                if prim in scores:
                    scores[prim] += desc.score
                    sources[prim].append(desc.text)

            # Method 2: Extract keywords and map to categories
            text_lower = desc.text.lower()
            for keyword, categories in self.keyword_to_category.items():
                if keyword in text_lower:
                    for category in categories:
                        if category in PRIMITIVE_CATEGORIES:
                            for prim in PRIMITIVE_CATEGORIES[category]:
                                scores[prim] += desc.score * 0.5  # Partial weight
                                if desc.text not in sources[prim]:
                                    sources[prim].append(desc.text)

        # Create bias objects
        biases = []
        for prim, score in scores.items():
            categories = PRIMITIVE_TO_CATEGORIES.get(prim, [])
            biases.append(PrimitiveBias(
                primitive=prim,
                score=score,
                source_descriptions=sources[prim],
                categories=categories
            ))

        # Sort by score descending
        biases.sort(key=lambda b: -b.score)
        return biases

    def get_top_primitives(
        self,
        descriptions: List[Description],
        top_k: int = 10
    ) -> List[str]:
        """Get the top-k most relevant primitives for descriptions."""
        biases = self.compute_biases(descriptions)
        return [b.primitive for b in biases[:top_k]]

    def compute_bias_vector(
        self,
        descriptions: List[Description],
        primitive_order: List[str]
    ) -> np.ndarray:
        """
        Compute a bias vector aligned with a specific primitive ordering.

        This can be used to adjust recognition network logits.

        Args:
            descriptions: List of descriptions
            primitive_order: List of primitives in the order expected

        Returns:
            numpy array of bias scores
        """
        biases = self.compute_biases(descriptions)
        bias_dict = {b.primitive: b.score for b in biases}

        vector = np.zeros(len(primitive_order))
        for i, prim in enumerate(primitive_order):
            vector[i] = bias_dict.get(prim, 0.0)

        # Normalize to reasonable range for logit adjustment
        if vector.max() > 0:
            vector = vector / vector.max() * 2.0  # Scale to [0, 2]

        return vector


# ============================================================================
# SYNTHESIS GUIDANCE
# ============================================================================

@dataclass
class SynthesisHint:
    """A hint for the synthesis process based on description analysis."""
    hint_type: str  # "prefer_primitive", "avoid_primitive", "structure_hint"
    content: str  # The hint content
    confidence: float  # How confident we are in this hint
    primitives: List[str]  # Related primitives


class SynthesisGuidance:
    """
    Provides guidance to the synthesis process based on descriptions.

    This module analyzes descriptions and generates hints that can:
    1. Prioritize certain primitives in enumeration
    2. Suggest program structure (e.g., "likely needs halves comparison")
    3. Prune unlikely search directions
    """

    def __init__(self):
        self.generator = DescriptionGenerator(n_baseline_samples=2000)
        self.biaser = DescriptionToPrimitiveBiaser()

    def analyze_task(
        self,
        positive_hands: List[Hand],
        negative_hands: List[Hand]
    ) -> Tuple[List[Description], List[SynthesisHint]]:
        """
        Analyze a task and generate synthesis hints.

        Args:
            positive_hands: Hands that satisfy the rule
            negative_hands: Hands that don't satisfy the rule

        Returns:
            Tuple of (descriptions, hints)
        """
        # Generate descriptions
        descriptions = self.generator.describe_task(
            positive_hands, negative_hands, top_k=10
        )

        # Generate hints from descriptions
        hints = self._generate_hints(descriptions, positive_hands, negative_hands)

        return descriptions, hints

    def get_primitive_priorities(
        self,
        positive_hands: List[Hand],
        negative_hands: List[Hand]
    ) -> Dict[str, float]:
        """
        Get priority scores for each primitive based on task analysis.

        Higher score = more likely to be useful for this task.
        """
        descriptions = self.generator.describe_task(
            positive_hands, negative_hands, top_k=10
        )
        biases = self.biaser.compute_biases(descriptions)
        return {b.primitive: b.score for b in biases}

    def _generate_hints(
        self,
        descriptions: List[Description],
        positive_hands: List[Hand],
        negative_hands: List[Hand]
    ) -> List[SynthesisHint]:
        """Generate synthesis hints from descriptions."""
        hints = []

        # Analyze description patterns
        has_position = any("first" in d.text.lower() or "last" in d.text.lower()
                          or "half" in d.text.lower() for d in descriptions)
        has_uniform = any("same" in d.text.lower() or "uniform" in d.text.lower()
                         for d in descriptions)
        has_comparison = any("more" in d.text.lower() or "less" in d.text.lower()
                            or "greater" in d.text.lower() for d in descriptions)
        has_palindrome = any("palindrome" in d.text.lower() or "symmetric" in d.text.lower()
                            for d in descriptions)

        if has_position:
            hints.append(SynthesisHint(
                hint_type="structure_hint",
                content="Rule likely involves positional access (first/last/halves)",
                confidence=0.8,
                primitives=["head", "last", "first_half", "second_half", "at"]
            ))

        if has_uniform:
            hints.append(SynthesisHint(
                hint_type="prefer_primitive",
                content="Rule likely checks for uniformity",
                confidence=0.7,
                primitives=["all_same_suit", "all_same_color", "n_unique_suits", "eq"]
            ))

        if has_comparison:
            hints.append(SynthesisHint(
                hint_type="prefer_primitive",
                content="Rule likely involves numeric comparison",
                confidence=0.6,
                primitives=["gt", "lt", "ge", "le", "eq"]
            ))

        if has_palindrome:
            hints.append(SynthesisHint(
                hint_type="structure_hint",
                content="Rule likely checks for palindromic structure",
                confidence=0.8,
                primitives=["reverse", "zip_with", "eq", "map"]
            ))

        # Add primitive hints from descriptions
        for desc in descriptions[:3]:  # Top 3 descriptions
            if desc.primitives:
                hints.append(SynthesisHint(
                    hint_type="prefer_primitive",
                    content=f"Based on: '{desc.text}'",
                    confidence=min(desc.score / 5.0, 1.0),
                    primitives=desc.primitives
                ))

        return hints


# ============================================================================
# SELF-EXPLANATION PROMPTS
# ============================================================================

class SelfExplanationPromptGenerator:
    """
    Generates self-explanation prompts for human learners.

    These prompts encourage learners to think about what makes
    winning hands different from losing hands, supporting learning.
    """

    def __init__(self):
        self.generator = DescriptionGenerator(n_baseline_samples=2000)

    def generate_prompts(
        self,
        positive_hands: List[Hand],
        negative_hands: List[Hand],
        n_prompts: int = 3
    ) -> List[str]:
        """
        Generate self-explanation prompts for a learning task.

        Args:
            positive_hands: Winning hands
            negative_hands: Losing hands
            n_prompts: Number of prompts to generate

        Returns:
            List of prompt strings
        """
        prompts = []

        # Analyze the task
        descriptions = self.generator.describe_task(
            positive_hands[:10], negative_hands[:10], top_k=5
        )

        # Generate prompts based on descriptions
        prompt_templates = [
            "Look at the winning hands. What do they have in common?",
            "Compare a winning hand to a losing hand. What's different?",
            "Can you describe a pattern that makes a hand win?",
            "What would you change in a losing hand to make it win?",
            "If you saw a new hand, how would you predict if it wins?",
        ]

        # Add description-specific prompts
        for desc in descriptions[:2]:
            if "winning" in desc.text or "losing" in desc.text:
                prompts.append(
                    f"The analysis suggests: {desc.text}. "
                    "Can you verify this with the examples?"
                )

        # Add general prompts
        prompts.extend(prompt_templates[:n_prompts - len(prompts)])

        return prompts[:n_prompts]

    def generate_feedback_prompt(
        self,
        hand: Hand,
        is_correct: bool,
        correct_label: bool
    ) -> str:
        """
        Generate a feedback prompt after a learner's prediction.

        Args:
            hand: The hand that was classified
            is_correct: Whether the learner's prediction was correct
            correct_label: The actual label (True = winning)

        Returns:
            Feedback prompt string
        """
        hand_str = hand_to_string(hand)
        descriptions = self.generator.describe_hand(hand, top_k=3)

        if is_correct:
            feedback = f"Correct! This hand {hand_str} is a "
            feedback += "winning hand. " if correct_label else "losing hand. "
            if descriptions:
                feedback += f"Notice that it {descriptions[0].text}."
        else:
            feedback = f"Not quite. This hand {hand_str} is actually a "
            feedback += "winning hand. " if correct_label else "losing hand. "
            if descriptions:
                feedback += f"Consider that it {descriptions[0].text}. "
                feedback += "How might this relate to the rule?"

        return feedback


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    from rules.cards import card_color, Suit

    print("=" * 70)
    print("SYNTHESIS INTEGRATION DEMO")
    print("=" * 70)

    # Create a test rule: uniform color
    def uniform_color(hand):
        colors = set(card_color(c) for c in hand)
        return len(colors) == 1

    # Generate examples
    positive = []
    negative = []
    for _ in range(100):
        h = sample_hand(6)
        if uniform_color(h):
            positive.append(h)
        else:
            negative.append(h)

    print(f"\nGenerated {len(positive)} positive, {len(negative)} negative examples")

    # Test 1: Description to primitive biasing
    print("\n" + "-" * 50)
    print("TEST 1: Primitive Biasing from Descriptions")
    print("-" * 50)

    generator = DescriptionGenerator(n_baseline_samples=2000)
    descriptions = generator.describe_task(positive[:10], negative[:10], top_k=5)

    print("Descriptions:")
    for d in descriptions:
        print(f"  - {d.text} (score: {d.score:.2f})")

    biaser = DescriptionToPrimitiveBiaser()
    biases = biaser.compute_biases(descriptions)

    print("\nTop 10 primitives by bias score:")
    for b in biases[:10]:
        print(f"  {b.primitive}: {b.score:.2f}")
        if b.source_descriptions:
            print(f"    Sources: {b.source_descriptions[:2]}")

    # Test 2: Synthesis hints
    print("\n" + "-" * 50)
    print("TEST 2: Synthesis Hints")
    print("-" * 50)

    guidance = SynthesisGuidance()
    _, hints = guidance.analyze_task(positive[:10], negative[:10])

    print("Generated hints:")
    for hint in hints:
        print(f"  [{hint.hint_type}] {hint.content}")
        print(f"    Primitives: {hint.primitives}")
        print(f"    Confidence: {hint.confidence:.2f}")

    # Test 3: Self-explanation prompts
    print("\n" + "-" * 50)
    print("TEST 3: Self-Explanation Prompts")
    print("-" * 50)

    prompt_gen = SelfExplanationPromptGenerator()
    prompts = prompt_gen.generate_prompts(positive[:10], negative[:10], n_prompts=3)

    print("Generated learning prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    # Test 4: Feedback generation
    print("\n" + "-" * 50)
    print("TEST 4: Feedback Prompts")
    print("-" * 50)

    test_hand = positive[0] if positive else sample_hand(6)
    feedback = prompt_gen.generate_feedback_prompt(test_hand, is_correct=False, correct_label=True)
    print(f"Incorrect prediction feedback:\n  {feedback}")

    feedback = prompt_gen.generate_feedback_prompt(test_hand, is_correct=True, correct_label=True)
    print(f"\nCorrect prediction feedback:\n  {feedback}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
