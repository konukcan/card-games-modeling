"""
Description Generator Module for Card Game Learning

This module provides tools for generating human-readable descriptions
of hands and tasks that map to primitive patterns in the DSL.

Main components:
- FeatureExtractor: Extracts structured features from hands
- SurpriseScorer: Computes informativeness of features
- DescriptionVocabulary: Maps features to natural language
- DescriptionGenerator: End-to-end pipeline
- SynthesisGuidance: Connects descriptions to program synthesis
- SelfExplanationPromptGenerator: Generates learning prompts

Example usage:
    from description_generator import DescriptionGenerator, SynthesisGuidance

    generator = DescriptionGenerator()

    # Describe a single hand
    descriptions = generator.describe_hand(hand)

    # Describe what distinguishes winning from losing hands
    task_descs = generator.describe_task(positive_hands, negative_hands)

    # Get synthesis guidance
    guidance = SynthesisGuidance()
    descriptions, hints = guidance.analyze_task(positive_hands, negative_hands)
"""

from .description_generator import (
    # Feature types and structures
    FeatureType,
    Feature,
    FeatureSet,

    # Core modules
    FeatureExtractor,
    SurpriseScorer,
    BaselineDistribution,
    DescriptionVocabulary,
    DescriptionTemplate,

    # Main generator
    DescriptionGenerator,
    Description,

    # Training utilities
    SyntheticTrainingDataGenerator,
    DescriptionValidator,
)

from .synthesis_integration import (
    # Primitive biasing
    PrimitiveBias,
    DescriptionToPrimitiveBiaser,
    PRIMITIVE_CATEGORIES,
    PRIMITIVE_TO_CATEGORIES,

    # Synthesis guidance
    SynthesisHint,
    SynthesisGuidance,

    # Self-explanation
    SelfExplanationPromptGenerator,
)

__all__ = [
    # Core description generation
    "FeatureType",
    "Feature",
    "FeatureSet",
    "FeatureExtractor",
    "SurpriseScorer",
    "BaselineDistribution",
    "DescriptionVocabulary",
    "DescriptionTemplate",
    "DescriptionGenerator",
    "Description",
    "SyntheticTrainingDataGenerator",
    "DescriptionValidator",

    # Synthesis integration
    "PrimitiveBias",
    "DescriptionToPrimitiveBiaser",
    "PRIMITIVE_CATEGORIES",
    "PRIMITIVE_TO_CATEGORIES",
    "SynthesisHint",
    "SynthesisGuidance",
    "SelfExplanationPromptGenerator",
]
