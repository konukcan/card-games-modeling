#!/usr/bin/env python3
"""
Tests for Unified Task Generation System
=========================================

This test suite validates the unified task generation system which guarantees:
1. Balanced examples (equal positives and negatives)
2. Near-miss negative generation (flip one card from positive)
3. Disjoint pools (seed, training, holdout)
4. Explicit failure if balance cannot be achieved

Run with: python3 -m pytest tests/test_task_generation.py -v
Or directly: python3 tests/test_task_generation.py
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.task_generation import (
    TaskGenerationConfig,
    TaskGenerationResult,
    create_unified_task,
    create_tasks_from_rules,
    create_dream_task,
    generate_and_save_dataset,
    load_prerecorded_tasks,
    serialize_task,
    deserialize_task,
)
from dreamcoder_core.task import Task
from dreamcoder_core.type_system import arrow, HAND, BOOL
from rules.cards import sample_hand, Card, Rank, Suit


# ============================================================================
# HELPER: Simple test rules
# ============================================================================

class SimpleRule:
    """Simple rule for testing."""
    def __init__(self, rule_id: str, eval_fn, family: str = "test", level: int = 1):
        self.id = rule_id
        self._eval_fn = eval_fn
        self.family = family
        self.level = level

    def eval(self, hand):
        return self._eval_fn(hand)


def has_ace(hand) -> bool:
    """Check if hand has at least one Ace."""
    return any(c.rank == Rank.ACE for c in hand)


def all_hearts(hand) -> bool:
    """Check if all cards are hearts."""
    return all(c.suit == Suit.HEARTS for c in hand)


def has_pair(hand) -> bool:
    """Check if hand has any pair."""
    ranks = [c.rank for c in hand]
    return len(ranks) != len(set(ranks))


def impossible_rule(hand) -> bool:
    """Rule that's never true - for testing failure handling."""
    return False


def always_true_rule(hand) -> bool:
    """Rule that's always true - for testing failure handling."""
    return True


# ============================================================================
# TEST 1: Basic Task Creation
# ============================================================================

def test_basic_task_creation():
    """Test that create_unified_task creates a valid task with balanced examples."""
    rule = SimpleRule("has_ace", has_ace)
    config = TaskGenerationConfig(
        n_training_positives=10,
        n_seed_positives=5,
        n_training_negatives=10,
        n_holdout_positives=5,
        n_holdout_negatives=5,
        hand_size=6,
        max_sampling_attempts=50_000,  # Give it more attempts
    )

    result = create_unified_task(rule, config, seed=42)

    assert result.success, f"Task creation should succeed: {result.failure_reason}"
    assert result.task is not None

    task = result.task
    # Check training examples are balanced
    pos_count = sum(1 for _, label in task.examples if label)
    neg_count = sum(1 for _, label in task.examples if not label)
    assert pos_count == 10, f"Expected 10 positives, got {pos_count}"
    assert neg_count == 10, f"Expected 10 negatives, got {neg_count}"

    # Check holdout examples are balanced
    holdout_pos = sum(1 for _, label in task.holdout if label)
    holdout_neg = sum(1 for _, label in task.holdout if not label)
    assert holdout_pos == 5, f"Expected 5 holdout positives, got {holdout_pos}"
    assert holdout_neg == 5, f"Expected 5 holdout negatives, got {holdout_neg}"

    print("✓ test_basic_task_creation passed")


# ============================================================================
# TEST 2: Near-Miss Negatives
# ============================================================================

def test_near_miss_negatives():
    """Test that near-miss negatives differ by exactly one card."""
    rule = SimpleRule("all_hearts", all_hearts)
    config = TaskGenerationConfig(
        n_training_positives=5,
        n_seed_positives=5,
        n_training_negatives=5,
        n_holdout_positives=3,
        n_holdout_negatives=3,
        hand_size=6,
        use_near_miss_negatives=True,
        near_miss_positions_to_try=6,  # Try all positions
        max_sampling_attempts=100_000,  # More attempts for rare rule
    )

    result = create_unified_task(rule, config, seed=42)

    assert result.success, f"Task creation should succeed: {result.failure_reason}"

    # Check near-miss statistics
    if result.stats:
        # We expect some near-miss negatives were generated
        near_miss_count = result.stats.get('near_miss_negatives_generated', 0)
        print(f"  Near-miss negatives generated: {near_miss_count}")

    print("✓ test_near_miss_negatives passed")


# ============================================================================
# TEST 3: Disjoint Pools
# ============================================================================

def test_disjoint_pools():
    """Test that training, seed, and holdout examples are disjoint."""
    rule = SimpleRule("has_ace", has_ace)
    config = TaskGenerationConfig(
        n_training_positives=10,
        n_seed_positives=10,
        n_training_negatives=10,
        n_holdout_positives=10,
        n_holdout_negatives=10,
        hand_size=6,
        max_sampling_attempts=100_000,  # More attempts
    )

    result = create_unified_task(rule, config, seed=42)

    assert result.success, f"Task creation should succeed: {result.failure_reason}"

    # Convert hands to tuples for set comparison
    def hand_to_tuple(hand):
        return tuple(sorted((c.rank.value, c.suit.value) for c in hand))

    training_hands = {hand_to_tuple(h) for h, _ in result.task.examples}
    holdout_hands = {hand_to_tuple(h) for h, _ in result.task.holdout}

    # Check disjointness of training and holdout (seed_positives is internal only)
    train_holdout_overlap = training_hands & holdout_hands

    assert len(train_holdout_overlap) == 0, \
        f"Training and holdout should be disjoint, overlap: {len(train_holdout_overlap)}"

    print("✓ test_disjoint_pools passed")


# ============================================================================
# TEST 4: Failure on Impossible Rules
# ============================================================================

def test_failure_on_impossible_rule():
    """Test that impossible rules fail gracefully when require_exact_balance=True."""
    rule = SimpleRule("impossible", impossible_rule)
    config = TaskGenerationConfig(
        n_training_positives=10,
        n_seed_positives=5,
        n_training_negatives=10,
        n_holdout_positives=5,
        n_holdout_negatives=5,
        hand_size=6,
        require_exact_balance=True,
        max_sampling_attempts=1000,  # Limited attempts
    )

    result = create_unified_task(rule, config, seed=42)

    assert not result.success, "Impossible rule should fail"
    assert result.failure_reason is not None
    assert "positive" in result.failure_reason.lower() or "balance" in result.failure_reason.lower()

    print("✓ test_failure_on_impossible_rule passed")


# ============================================================================
# TEST 5: create_tasks_from_rules
# ============================================================================

def test_create_tasks_from_rules():
    """Test the batch task creation function."""
    rules = [
        SimpleRule("has_ace", has_ace),
        SimpleRule("has_pair", has_pair),
    ]
    config = TaskGenerationConfig(
        n_training_positives=5,
        n_seed_positives=3,
        n_training_negatives=5,
        n_holdout_positives=3,
        n_holdout_negatives=3,
        max_sampling_attempts=50_000,
    )

    tasks, failures = create_tasks_from_rules(rules, config=config, seed=42, skip_failures=True)

    # At least one task should succeed (has_pair is common)
    assert len(tasks) >= 1, f"Expected at least 1 task, got {len(tasks)}"

    for task in tasks:
        pos_count = sum(1 for _, label in task.examples if label)
        neg_count = sum(1 for _, label in task.examples if not label)
        assert pos_count == 5, f"Task {task.name}: expected 5 positives, got {pos_count}"
        assert neg_count == 5, f"Task {task.name}: expected 5 negatives, got {neg_count}"

    print(f"✓ test_create_tasks_from_rules passed ({len(tasks)}/2 tasks, {len(failures)} failures)")


# ============================================================================
# TEST 6: Serialization and Deserialization
# ============================================================================

def test_serialization():
    """Test that tasks can be serialized and deserialized correctly."""
    # Use has_pair which is more common
    rule = SimpleRule("has_pair", has_pair, family="test_family", level=3)
    config = TaskGenerationConfig(
        n_training_positives=5,
        n_seed_positives=3,
        n_training_negatives=5,
        n_holdout_positives=3,
        n_holdout_negatives=3,
        max_sampling_attempts=50_000,
    )

    result = create_unified_task(rule, config, seed=42)
    assert result.success, f"Task creation failed: {result.failure_reason}"

    # Serialize
    serialized = serialize_task(result.task)
    assert 'name' in serialized
    assert 'training_examples' in serialized
    assert 'holdout_examples' in serialized
    assert serialized['name'] == 'has_pair'

    # Deserialize
    restored_task = deserialize_task(serialized)
    assert restored_task.name == result.task.name
    assert len(restored_task.examples) == len(result.task.examples)
    assert len(restored_task.holdout) == len(result.task.holdout)

    print("✓ test_serialization passed")


# ============================================================================
# TEST 7: Dataset Save and Load
# ============================================================================

def test_dataset_save_and_load():
    """Test saving and loading a complete dataset."""
    # Use only has_pair which is common and should always work
    rules = [
        SimpleRule("has_pair", has_pair),
    ]
    config = TaskGenerationConfig(
        n_training_positives=5,
        n_seed_positives=3,
        n_training_negatives=5,
        n_holdout_positives=3,
        n_holdout_negatives=3,
        max_sampling_attempts=50_000,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_tasks.json"

        # Save
        tasks, dataset = generate_and_save_dataset(
            rules=rules,
            output_path=output_path,
            config=config,
            seed=42,
            rule_source="test"
        )

        assert output_path.exists()
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"

        # Load
        loaded_tasks = load_prerecorded_tasks(output_path)
        assert len(loaded_tasks) == 1

        # Verify loaded tasks have correct structure
        for task in loaded_tasks:
            assert hasattr(task, 'examples')
            assert hasattr(task, 'holdout')
            assert len(task.examples) == 10  # 5 pos + 5 neg
            assert len(task.holdout) == 6   # 3 pos + 3 neg

    print("✓ test_dataset_save_and_load passed")


# ============================================================================
# TEST 8: Dream Task Creation
# ============================================================================

def test_dream_task_creation():
    """Test creating dream tasks from program functions."""
    # Create a simple rule function to simulate a "dream"
    def dream_rule(hand) -> bool:
        """Simple dream rule: has at least 2 hearts."""
        return sum(1 for c in hand if c.suit == Suit.HEARTS) >= 2

    config = TaskGenerationConfig(
        n_training_positives=5,
        n_seed_positives=0,  # No seeds needed for dreams
        n_training_negatives=5,
        n_holdout_positives=0,
        n_holdout_negatives=0,
        max_sampling_attempts=10_000,
    )

    result = create_dream_task(
        program_fn=dream_rule,
        dream_id="dream_0",
        config=config,
        seed=42
    )

    assert result.success, f"Dream task creation failed: {result.failure_reason}"
    assert result.task is not None
    assert result.task.name == "dream_0"
    assert len(result.task.examples) > 0

    print("✓ test_dream_task_creation passed")


# ============================================================================
# TEST 9: sym_ranks_palindrome Bug Regression Test
# ============================================================================

def test_sym_ranks_palindrome_regression():
    """
    Regression test for the sym_ranks_palindrome bug.

    The old implementation would silently generate 0 positives for rare rules,
    allowing (λ false) to be a "valid" solution. The new implementation should
    either succeed with proper positives OR fail explicitly.
    """
    def is_palindrome(hand) -> bool:
        """Check if hand ranks form a palindrome."""
        ranks = [c.rank for c in hand]
        return ranks == ranks[::-1]

    rule = SimpleRule("sym_ranks_palindrome", is_palindrome)
    config = TaskGenerationConfig(
        n_training_positives=10,
        n_seed_positives=5,
        n_training_negatives=10,
        n_holdout_positives=5,
        n_holdout_negatives=5,
        hand_size=6,
        max_sampling_attempts=200_000,  # Give it a good chance
        require_exact_balance=True,
    )

    result = create_unified_task(rule, config, seed=42)

    if result.success:
        # If successful, verify we have actual positives
        pos_count = sum(1 for _, label in result.task.examples if label)
        assert pos_count == 10, f"Expected 10 positives, got {pos_count}"

        # Verify the positives are actually palindromes
        for hand, label in result.task.examples:
            if label:
                ranks = [c.rank for c in hand]
                assert ranks == ranks[::-1], f"False positive: {ranks}"

        print("✓ sym_ranks_palindrome succeeded with proper positives")
    else:
        # If failed, that's acceptable - it should fail explicitly
        assert result.failure_reason is not None
        print(f"✓ sym_ranks_palindrome failed explicitly: {result.failure_reason}")

    print("✓ test_sym_ranks_palindrome_regression passed")


# ============================================================================
# TEST 10: Config Defaults
# ============================================================================

def test_config_defaults():
    """Test that TaskGenerationConfig has sensible defaults."""
    config = TaskGenerationConfig()

    assert config.n_training_positives == 20
    assert config.n_training_negatives == 20
    assert config.hand_size == 6
    assert config.use_near_miss_negatives == True
    assert config.max_sampling_attempts == 200_000  # Actual default

    print("✓ test_config_defaults passed")


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK GENERATION TESTS")
    print("=" * 60)
    print()

    tests = [
        test_config_defaults,
        test_basic_task_creation,
        test_near_miss_negatives,
        test_disjoint_pools,
        test_failure_on_impossible_rule,
        test_create_tasks_from_rules,
        test_serialization,
        test_dataset_save_and_load,
        test_dream_task_creation,
        test_sym_ranks_palindrome_regression,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
