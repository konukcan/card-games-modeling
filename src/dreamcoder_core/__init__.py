"""
DreamCoder Core Implementation for Card Game Rules

This package implements a cognitively-realistic version of DreamCoder
for learning card game rules from examples.

The goal is NOT maximum accuracy, but rather:
1. Cognitively realistic learning difficulty
2. Measuring transfer effects between rules
3. Understanding which rules are easier/harder to learn
4. Tracking library growth over time

Based on Ellis et al. (2021, 2023) "DreamCoder: Growing generalizable,
interpretable knowledge with wake-sleep Bayesian program learning"
"""

# Import as modules are created
from .type_system import *
from .program import *
from .grammar import *
from .enumeration import *
from .compression import *
from .wake_sleep import *
