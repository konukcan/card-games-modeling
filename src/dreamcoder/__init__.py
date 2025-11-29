"""
DreamCoder Integration Module

This module implements the core DreamCoder components:
- Recognition Network: Neural guidance for program search
- Enumeration: Best-first search over program space
- Compression: Library learning from solved tasks
- Wake-Sleep: Iterative improvement loop

Based on Ellis et al. (2023) "DreamCoder: Growing Generalizable, Interpretable Knowledge
with Wake-Sleep Bayesian Program Learning"
"""

from .recognition import (
    RecognitionNetwork,
    train_recognition_network,
    predict_primitives,
    save_model,
    load_model,
    PRIMITIVE_LIST,
    PRIMITIVE_TO_IDX,
    NUM_PRIMITIVES,
)

__all__ = [
    'RecognitionNetwork',
    'train_recognition_network',
    'predict_primitives',
    'save_model',
    'load_model',
    'PRIMITIVE_LIST',
    'PRIMITIVE_TO_IDX',
    'NUM_PRIMITIVES',
]
