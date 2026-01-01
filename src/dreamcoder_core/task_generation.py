#!/usr/bin/env python3
"""
Unified Task Generation for DreamCoder Card Games
==================================================

This module provides a single, authoritative implementation for generating
training and holdout tasks from rules. It replaces multiple inconsistent
implementations scattered across the codebase.

KEY FEATURES:
1. Guaranteed balanced examples (equal positives and negatives)
2. Near-miss negative generation (flip one card from positive to create negative)
3. Separation of positive pools (training vs seed vs holdout) to prevent data leakage
4. Explicit failure if balance cannot be achieved
5. Pre-generation and caching support for reproducibility

ARCHITECTURE:
    Positive Examples are partitioned into THREE disjoint pools:

    ┌─────────────────────────────────────────────────────────────────┐
    │  TRAINING       │  SEED (hidden)   │  HOLDOUT                   │
    │  POSITIVES      │  for near-miss   │  POSITIVES                 │
    │  (model sees)   │  generation      │  (verification)            │
    │                 │  (model NEVER    │                            │
    │                 │   sees these)    │                            │
    └─────────────────┴─────────────────┴─────────────────────────────┘

    Near-miss negatives are generated from SEED positives, NOT training positives.
    This prevents the model from learning the near-miss generation pattern.

Author: Can Konuk
Date: January 2025
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

# Local imports
from dreamcoder_core.task import Task
from dreamcoder_core.type_system import arrow, HAND, BOOL

# Conditional import for cards module (may be imported from different contexts)
try:
    from rules.cards import Hand, Card, Suit, Rank, sample_hand
except ImportError:
    # Try parent path import
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rules.cards import Hand, Card, Suit, Rank, sample_hand


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TaskGenerationConfig:
    """
    Configuration for task generation.

    The total number of examples will be:
    - Training: n_training_positives + n_training_negatives
    - Holdout: n_holdout_positives + n_holdout_negatives

    The n_seed_positives are used ONLY for generating near-miss negatives
    and are NEVER shown to the model as training examples.
    """
    # Example counts
    n_training_positives: int = 20      # Positives the model sees
    n_seed_positives: int = 20          # Hidden seeds for near-miss generation
    n_training_negatives: int = 20      # Negatives the model sees (from near-miss)
    n_holdout_positives: int = 10       # For verification
    n_holdout_negatives: int = 10       # Random negatives for holdout

    # Sampling parameters
    hand_size: int = 6
    max_sampling_attempts: int = 200_000  # Max attempts to find examples
    max_near_miss_attempts_per_seed: int = 200  # Per seed positive

    # Balance requirements (STRICT)
    min_positive_ratio: float = 0.8     # Fail if < 80% of target positives found
    require_exact_balance: bool = True  # Enforce exact counts

    # Near-miss configuration
    use_near_miss_negatives: bool = True  # If False, use random negatives
    near_miss_positions_to_try: int = 4   # How many card positions to try flipping

    # Fallback options
    allow_random_negative_fallback: bool = True  # If near-miss fails, use random

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskGenerationConfig':
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def default_training(cls) -> 'TaskGenerationConfig':
        """Default config for training tasks."""
        return cls(
            n_training_positives=20,
            n_seed_positives=20,
            n_training_negatives=20,
            n_holdout_positives=20,
            n_holdout_negatives=20,
        )

    @classmethod
    def default_evaluation(cls) -> 'TaskGenerationConfig':
        """Config for evaluation with more holdout."""
        return cls(
            n_training_positives=40,
            n_seed_positives=40,
            n_training_negatives=40,
            n_holdout_positives=50,
            n_holdout_negatives=50,
        )

    @classmethod
    def minimal_test(cls) -> 'TaskGenerationConfig':
        """Minimal config for quick tests."""
        return cls(
            n_training_positives=5,
            n_seed_positives=5,
            n_training_negatives=5,
            n_holdout_positives=5,
            n_holdout_negatives=5,
            max_sampling_attempts=10000,
        )


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class TaskGenerationResult:
    """Result of task generation with detailed statistics."""
    task: Optional[Task]
    stats: Dict[str, Any]
    success: bool
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (excluding Task object)."""
        return {
            'success': self.success,
            'failure_reason': self.failure_reason,
            'stats': self.stats,
            'task_name': self.task.name if self.task else None,
            'n_training': len(self.task.examples) if self.task else 0,
            'n_holdout': len(self.task.holdout) if self.task else 0,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sample_single_card() -> Card:
    """Sample a single random card."""
    return Card(random.choice(list(Suit)), random.choice(list(Rank)))


def hand_to_key(hand: Hand) -> tuple:
    """Create a hashable key for a hand to detect duplicates."""
    return tuple((c.suit.name, c.rank.name) for c in hand)


def hand_to_serializable(hand: Hand) -> List[Dict[str, str]]:
    """Convert hand to JSON-serializable format."""
    return [{'suit': c.suit.name, 'rank': c.rank.name} for c in hand]


def hand_from_serializable(data: List[Dict[str, str]]) -> Hand:
    """Reconstruct hand from JSON format."""
    return [Card(Suit[d['suit']], Rank[d['rank']]) for d in data]


def create_near_miss_negative(
    positive_hand: Hand,
    rule_fn: Callable[[Hand], bool],
    max_attempts: int = 200,
    positions_to_try: int = 4
) -> Optional[Hand]:
    """
    Create a near-miss negative by flipping exactly one card.

    The resulting hand must:
    1. Differ from the positive by exactly ONE card
    2. Evaluate to False under the rule

    Args:
        positive_hand: A hand that evaluates to True
        rule_fn: The rule evaluation function
        max_attempts: Max attempts per position
        positions_to_try: How many positions to try

    Returns:
        A hand differing by one card that evaluates to False, or None if impossible
    """
    hand_size = len(positive_hand)
    positions = list(range(hand_size))
    random.shuffle(positions)

    attempts_per_position = max(1, max_attempts // min(positions_to_try, hand_size))

    for pos in positions[:positions_to_try]:
        for _ in range(attempts_per_position):
            new_card = sample_single_card()

            # Skip if same card (no change)
            if new_card == positive_hand[pos]:
                continue

            # Create modified hand
            new_hand = list(positive_hand)
            new_hand[pos] = new_card

            try:
                result = rule_fn(new_hand)
                if result == False:
                    return new_hand
            except Exception:
                # Rule evaluation failed - skip this modification
                continue

    return None


# ============================================================================
# MAIN TASK GENERATION FUNCTION
# ============================================================================

def create_unified_task(
    rule,  # Rule object with .eval() and .id
    config: TaskGenerationConfig,
    seed: Optional[int] = None
) -> TaskGenerationResult:
    """
    Create a task with guaranteed balanced examples and near-miss negatives.

    GUARANTEES:
    1. Equal positives and negatives in training set
    2. Near-miss negatives generated from SEPARATE seed pool
    3. Holdout examples are disjoint from training
    4. Fails explicitly if balance cannot be achieved

    Args:
        rule: Rule object with .eval(hand) -> bool and .id attribute
        config: TaskGenerationConfig with all parameters
        seed: Random seed for reproducibility

    Returns:
        TaskGenerationResult with task and statistics
    """
    if seed is not None:
        random.seed(seed)

    stats = {
        'rule_id': getattr(rule, 'id', str(rule)),
        'attempts': 0,
        'positives_found': 0,
        'negatives_found_random': 0,
        'near_miss_successes': 0,
        'near_miss_failures': 0,
        'config': config.to_dict(),
    }

    # Total positives needed: training + seeds + holdout
    total_positives_needed = (
        config.n_training_positives +
        config.n_seed_positives +
        config.n_holdout_positives
    )

    # Total random negatives needed for holdout
    total_random_negatives_needed = config.n_holdout_negatives

    # Also collect some extra random negatives as fallback
    extra_negatives_buffer = config.n_training_negatives if config.allow_random_negative_fallback else 0

    # ================================================================
    # PHASE 1: Collect ALL positives and random negatives by sampling
    # ================================================================
    all_positives: List[Hand] = []
    all_random_negatives: List[Hand] = []
    seen_hands: Set[tuple] = set()

    for attempt in range(config.max_sampling_attempts):
        stats['attempts'] += 1

        # Check if we have enough
        if (len(all_positives) >= total_positives_needed and
            len(all_random_negatives) >= total_random_negatives_needed + extra_negatives_buffer):
            break

        hand = sample_hand(config.hand_size)
        key = hand_to_key(hand)

        if key in seen_hands:
            continue
        seen_hands.add(key)

        try:
            result = rule.eval(hand)

            if result == True and len(all_positives) < total_positives_needed:
                all_positives.append(hand)
                stats['positives_found'] += 1
            elif result == False:
                if len(all_random_negatives) < total_random_negatives_needed + extra_negatives_buffer:
                    all_random_negatives.append(hand)
                    stats['negatives_found_random'] += 1

        except Exception as e:
            logger.debug(f"Rule evaluation failed for {getattr(rule, 'id', rule)}: {e}")
            continue

    # ================================================================
    # PHASE 2: Check if we have enough positives
    # ================================================================
    if len(all_positives) < total_positives_needed:
        shortfall = total_positives_needed - len(all_positives)
        min_required = int(total_positives_needed * config.min_positive_ratio)

        if len(all_positives) < min_required:
            failure_msg = (
                f"Could not find enough positive examples for {getattr(rule, 'id', rule)}. "
                f"Found {len(all_positives)}, needed {total_positives_needed} "
                f"(minimum: {min_required}, shortfall: {shortfall}). "
                f"This rule may be too rare for random sampling."
            )
            logger.warning(failure_msg)

            return TaskGenerationResult(
                task=None,
                stats=stats,
                success=False,
                failure_reason=failure_msg
            )
        else:
            # Adjust counts proportionally
            ratio = len(all_positives) / total_positives_needed
            config = TaskGenerationConfig(
                n_training_positives=int(config.n_training_positives * ratio),
                n_seed_positives=int(config.n_seed_positives * ratio),
                n_training_negatives=int(config.n_training_negatives * ratio),
                n_holdout_positives=int(config.n_holdout_positives * ratio),
                n_holdout_negatives=config.n_holdout_negatives,
                hand_size=config.hand_size,
                use_near_miss_negatives=config.use_near_miss_negatives,
                near_miss_positions_to_try=config.near_miss_positions_to_try,
                max_near_miss_attempts_per_seed=config.max_near_miss_attempts_per_seed,
                allow_random_negative_fallback=config.allow_random_negative_fallback,
            )
            logger.info(f"Adjusted counts for {getattr(rule, 'id', rule)} due to limited positives")

    # ================================================================
    # PHASE 3: Partition positives into three disjoint pools
    # ================================================================
    random.shuffle(all_positives)

    # Pool 1: Training positives (model sees these)
    training_positives = all_positives[:config.n_training_positives]

    # Pool 2: Seed positives (NEVER seen, only used for near-miss generation)
    seed_start = config.n_training_positives
    seed_end = seed_start + config.n_seed_positives
    seed_positives = all_positives[seed_start:seed_end]

    # Pool 3: Holdout positives (for verification)
    holdout_start = seed_end
    holdout_positives = all_positives[holdout_start:holdout_start + config.n_holdout_positives]

    # ================================================================
    # PHASE 4: Generate near-miss negatives from seed pool
    # ================================================================
    training_negatives: List[Hand] = []
    near_miss_tracking: List[Tuple[int, int]] = []  # (seed_idx, which card flipped)

    if config.use_near_miss_negatives:
        for seed_idx, seed_hand in enumerate(seed_positives):
            if len(training_negatives) >= config.n_training_negatives:
                break

            near_miss = create_near_miss_negative(
                positive_hand=seed_hand,
                rule_fn=rule.eval,
                max_attempts=config.max_near_miss_attempts_per_seed,
                positions_to_try=config.near_miss_positions_to_try
            )

            if near_miss is not None:
                training_negatives.append(near_miss)
                stats['near_miss_successes'] += 1
            else:
                stats['near_miss_failures'] += 1

    # ================================================================
    # PHASE 5: Fill remaining negatives with random if near-miss insufficient
    # ================================================================
    negatives_shortfall = config.n_training_negatives - len(training_negatives)

    if negatives_shortfall > 0:
        if config.allow_random_negative_fallback:
            logger.info(
                f"Near-miss generated {len(training_negatives)}/{config.n_training_negatives} "
                f"for {getattr(rule, 'id', rule)}, filling {negatives_shortfall} with random"
            )

            # Use pre-collected random negatives
            random_fill = all_random_negatives[:negatives_shortfall]
            training_negatives.extend(random_fill)
            all_random_negatives = all_random_negatives[negatives_shortfall:]
        else:
            # Try more sampling
            extra_attempts = 0
            while len(training_negatives) < config.n_training_negatives and extra_attempts < 50000:
                extra_attempts += 1
                stats['attempts'] += 1

                hand = sample_hand(config.hand_size)
                key = hand_to_key(hand)

                if key in seen_hands:
                    continue
                seen_hands.add(key)

                try:
                    if rule.eval(hand) == False:
                        training_negatives.append(hand)
                        stats['negatives_found_random'] += 1
                except Exception:
                    continue

    # ================================================================
    # PHASE 6: Final balance check
    # ================================================================
    if config.require_exact_balance:
        # Check training balance
        if len(training_positives) != len(training_negatives):
            # Truncate to smaller count
            min_count = min(len(training_positives), len(training_negatives))
            training_positives = training_positives[:min_count]
            training_negatives = training_negatives[:min_count]

            if min_count == 0:
                failure_msg = (
                    f"Could not generate any balanced examples for {getattr(rule, 'id', rule)}. "
                    f"Positives: {stats['positives_found']}, "
                    f"Near-miss successes: {stats['near_miss_successes']}, "
                    f"Random negatives: {stats['negatives_found_random']}"
                )
                logger.warning(failure_msg)
                return TaskGenerationResult(
                    task=None,
                    stats=stats,
                    success=False,
                    failure_reason=failure_msg
                )

    # ================================================================
    # PHASE 7: Assemble training and holdout examples
    # ================================================================
    training_examples = (
        [(h, True) for h in training_positives] +
        [(h, False) for h in training_negatives]
    )
    random.shuffle(training_examples)

    # Use remaining random negatives for holdout
    holdout_negatives = all_random_negatives[:config.n_holdout_negatives]

    holdout_examples = (
        [(h, True) for h in holdout_positives] +
        [(h, False) for h in holdout_negatives]
    )
    random.shuffle(holdout_examples)

    # ================================================================
    # PHASE 8: Create Task object
    # ================================================================
    task = Task(
        name=getattr(rule, 'id', str(rule)),
        request_type=arrow(HAND, BOOL),
        examples=training_examples,
        holdout=holdout_examples,
        family=getattr(rule, 'family', ''),
        difficulty_level=getattr(rule, 'level', 0),
        near_miss_pairs=[],  # Could track if needed
        rule_fn=rule.eval
    )

    # Update stats with final counts
    stats['training_positives'] = len(training_positives)
    stats['training_negatives'] = len(training_negatives)
    stats['holdout_positives'] = len(holdout_positives)
    stats['holdout_negatives'] = len(holdout_negatives)
    stats['training_total'] = len(training_examples)
    stats['holdout_total'] = len(holdout_examples)
    stats['near_miss_ratio'] = stats['near_miss_successes'] / max(1, len(seed_positives))
    stats['balance_achieved'] = len(training_positives) == len(training_negatives)

    logger.info(
        f"Created task {getattr(rule, 'id', rule)}: "
        f"{len(training_examples)} training ({stats['training_positives']}+/{stats['training_negatives']}-), "
        f"{len(holdout_examples)} holdout, "
        f"near-miss ratio: {stats['near_miss_ratio']:.1%}"
    )

    return TaskGenerationResult(
        task=task,
        stats=stats,
        success=True
    )


# ============================================================================
# BATCH TASK GENERATION
# ============================================================================

def create_tasks_from_rules(
    rules: List,
    config: Optional[TaskGenerationConfig] = None,
    seed: int = 42,
    skip_failures: bool = True,
    verbose: bool = True
) -> Tuple[List[Task], List[Tuple[str, str]]]:
    """
    Create Task objects from a list of rules.

    Args:
        rules: List of Rule objects with .eval() and .id
        config: TaskGenerationConfig (uses defaults if None)
        seed: Base random seed
        skip_failures: If True, skip rules that fail; if False, raise exception
        verbose: Print progress

    Returns:
        (tasks, failures) - List of successful tasks and list of (rule_id, reason) tuples
    """
    if config is None:
        config = TaskGenerationConfig.default_training()

    tasks = []
    failures = []

    for i, rule in enumerate(rules):
        # Use different seed per rule for reproducibility
        rule_seed = seed + i * 1000

        if verbose:
            print(f"  Generating task {i+1}/{len(rules)}: {getattr(rule, 'id', rule)}...", end=" ")

        result = create_unified_task(rule, config, seed=rule_seed)

        if result.success:
            tasks.append(result.task)
            if verbose:
                print(f"OK ({result.stats['training_total']} train, {result.stats['holdout_total']} holdout)")
        else:
            failures.append((getattr(rule, 'id', str(rule)), result.failure_reason))
            if verbose:
                print(f"FAILED: {result.failure_reason[:60]}...")
            if not skip_failures:
                raise ValueError(f"Task generation failed for {getattr(rule, 'id', rule)}: {result.failure_reason}")

    return tasks, failures


# ============================================================================
# SERIALIZATION FOR PRE-RECORDED DATASETS
# ============================================================================

@dataclass
class PrerecordedDataset:
    """A pre-recorded dataset of tasks for reproducible experiments."""
    tasks: List[Dict]  # Serialized tasks
    config: Dict       # Generation config
    metadata: Dict     # Generation metadata (timestamp, versions, etc.)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        path = Path(path)
        data = {
            'tasks': self.tasks,
            'config': self.config,
            'metadata': self.metadata
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'PrerecordedDataset':
        """Load from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            tasks=data['tasks'],
            config=data['config'],
            metadata=data['metadata']
        )


def serialize_task(task: Task) -> Dict:
    """Convert a Task to JSON-serializable format."""
    return {
        'name': task.name,
        'family': task.family,
        'difficulty_level': task.difficulty_level,
        'training_examples': [
            {'hand': hand_to_serializable(hand), 'label': label}
            for hand, label in task.examples
        ],
        'holdout_examples': [
            {'hand': hand_to_serializable(hand), 'label': label}
            for hand, label in task.holdout
        ],
        'n_training': len(task.examples),
        'n_holdout': len(task.holdout),
        'training_balance': {
            'positives': sum(1 for _, l in task.examples if l),
            'negatives': sum(1 for _, l in task.examples if not l)
        },
        'holdout_balance': {
            'positives': sum(1 for _, l in task.holdout if l),
            'negatives': sum(1 for _, l in task.holdout if not l)
        }
    }


def deserialize_task(data: Dict) -> Task:
    """Reconstruct a Task from JSON format."""
    training_examples = [
        (hand_from_serializable(ex['hand']), ex['label'])
        for ex in data['training_examples']
    ]
    holdout_examples = [
        (hand_from_serializable(ex['hand']), ex['label'])
        for ex in data['holdout_examples']
    ]

    return Task(
        name=data['name'],
        request_type=arrow(HAND, BOOL),
        examples=training_examples,
        holdout=holdout_examples,
        family=data.get('family', ''),
        difficulty_level=data.get('difficulty_level', 0)
    )


def generate_and_save_dataset(
    rules: List,
    output_path: Union[str, Path],
    config: Optional[TaskGenerationConfig] = None,
    seed: int = 42,
    rule_source: str = "unknown"
) -> Tuple[List[Task], PrerecordedDataset]:
    """
    Generate tasks from rules and save to a JSON file.

    Args:
        rules: List of Rule objects
        output_path: Path to save the JSON file
        config: TaskGenerationConfig
        seed: Random seed
        rule_source: Description of where rules came from (for metadata)

    Returns:
        (tasks, dataset) - The generated tasks and the dataset object
    """
    if config is None:
        config = TaskGenerationConfig.default_training()

    print(f"Generating tasks for {len(rules)} rules...")
    tasks, failures = create_tasks_from_rules(rules, config, seed, skip_failures=True, verbose=True)

    # Serialize tasks
    serialized_tasks = [serialize_task(task) for task in tasks]

    # Create metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'seed': seed,
        'rule_source': rule_source,
        'n_rules_attempted': len(rules),
        'n_tasks_generated': len(tasks),
        'n_failures': len(failures),
        'failures': [{'rule_id': rid, 'reason': reason} for rid, reason in failures]
    }

    # Create dataset
    dataset = PrerecordedDataset(
        tasks=serialized_tasks,
        config=config.to_dict(),
        metadata=metadata
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(output_path)

    print(f"\nSaved dataset to: {output_path}")
    print(f"  Tasks: {len(tasks)}/{len(rules)}")
    print(f"  Failures: {len(failures)}")

    return tasks, dataset


def load_prerecorded_tasks(path: Union[str, Path]) -> List[Task]:
    """
    Load pre-recorded tasks from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        List of Task objects
    """
    dataset = PrerecordedDataset.from_json(path)
    return [deserialize_task(data) for data in dataset.tasks]


# ============================================================================
# DREAMING SUPPORT
# ============================================================================

def create_dream_task(
    program_fn: Callable[[Hand], bool],
    dream_id: str,
    config: Optional[TaskGenerationConfig] = None,
    seed: Optional[int] = None
) -> TaskGenerationResult:
    """
    Create a dream task from a program function (for dreaming phase).

    This is used during the sleep phase to generate training data for
    the recognition model from sampled programs.

    Unlike pre-recorded tasks, dream tasks are generated on-the-fly
    with fresh random data each time.

    Args:
        program_fn: A function (Hand -> bool) from evaluating a sampled program
        dream_id: Identifier for this dream
        config: Generation config (uses lighter defaults for dreams)
        seed: Optional random seed

    Returns:
        TaskGenerationResult with the dream task
    """
    if config is None:
        # Lighter config for dreams (no holdout needed)
        config = TaskGenerationConfig(
            n_training_positives=10,
            n_seed_positives=10,
            n_training_negatives=10,
            n_holdout_positives=0,  # No holdout for dreams
            n_holdout_negatives=0,
            max_sampling_attempts=50000,
            use_near_miss_negatives=True,
        )

    # Create a mock rule object
    class MockRule:
        def __init__(self, rule_id, eval_fn):
            self.id = rule_id
            self.family = "dream"
            self.level = 0
            self._eval_fn = eval_fn

        def eval(self, hand):
            return self._eval_fn(hand)

    mock_rule = MockRule(dream_id, program_fn)

    return create_unified_task(mock_rule, config, seed)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TASK GENERATION MODULE TEST")
    print("=" * 70)

    # Test with a simple rule
    class SimpleRule:
        def __init__(self, rule_id):
            self.id = rule_id
            self.family = "test"
            self.level = 0

        def eval(self, hand):
            # Simple rule: first card is red
            from rules.cards import Color
            return hand[0].suit in [Suit.HEARTS, Suit.DIAMONDS]

    rule = SimpleRule("test_first_red")
    config = TaskGenerationConfig.minimal_test()

    print("\n1. Testing single task generation...")
    result = create_unified_task(rule, config, seed=42)

    if result.success:
        print(f"   SUCCESS: {result.task}")
        print(f"   Training: {result.stats['training_positives']}+ / {result.stats['training_negatives']}-")
        print(f"   Holdout: {result.stats['holdout_positives']}+ / {result.stats['holdout_negatives']}-")
        print(f"   Near-miss ratio: {result.stats['near_miss_ratio']:.1%}")
    else:
        print(f"   FAILED: {result.failure_reason}")

    print("\n2. Testing serialization...")
    if result.success:
        serialized = serialize_task(result.task)
        deserialized = deserialize_task(serialized)
        print(f"   Original: {result.task}")
        print(f"   Deserialized: {deserialized}")
        assert len(result.task.examples) == len(deserialized.examples)
        print("   Serialization OK!")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
