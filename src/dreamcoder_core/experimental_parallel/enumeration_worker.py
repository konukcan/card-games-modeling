#!/usr/bin/env python3
"""PyPy Worker for Enumeration - DO NOT EDIT DIRECTLY"""
import sys
import pickle
import json
import time
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.primitives import build_lean_grammar
from rules.cards import Card, Suit, Rank


def deserialize_hand(hand_data):
    """Convert JSON hand data back to Card objects."""
    cards = []
    for card_dict in hand_data:
        suit = Suit[card_dict['suit']]
        rank = Rank[card_dict['rank']]
        cards.append(Card(suit, rank))
    return tuple(cards)


def evaluate_program(program, hand):
    """Evaluate a program on a hand.

    Returns None if evaluation fails due to expected runtime errors.
    Unexpected errors are logged to stderr for debugging.
    """
    try:
        fn = program.evaluate([])
        return fn(hand)
    except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError):
        # Expected errors from malformed or incompatible programs
        return None
    except RecursionError:
        # Infinite recursion in program - treat as failure
        return None
    except Exception as e:
        # Unexpected error - log for debugging but don't crash worker
        print(f"UNEXPECTED ERROR in evaluate_program: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def enumerate_task(task_data, grammar_productions, max_depth, max_programs, timeout):
    """
    Enumerate programs for a single task.

    Uses early pruning: stop evaluating examples as soon as one fails.
    """
    task_name = task_data['name']
    raw_examples = task_data['examples']

    # Deserialize examples (hand_data, expected_output)
    examples = []
    for hand_data, expected in raw_examples:
        hand = deserialize_hand(hand_data)
        examples.append((hand, expected))

    # Build grammar
    grammar = build_lean_grammar()

    # Apply any updated production weights
    if grammar_productions:
        # Could restore grammar weights here
        pass

    request_type = arrow(HAND, BOOL)

    results = []
    programs_tried = 0
    start_time = time.time()

    for program, log_prob in enumerate_simple(grammar, request_type, max_depth=max_depth):
        programs_tried += 1

        if programs_tried > max_programs:
            break
        if time.time() - start_time > timeout:
            break

        # Evaluate with EARLY PRUNING
        try:
            all_correct = True
            correct = 0

            for inp, expected in examples:
                result = evaluate_program(program, inp)
                if result == expected:
                    correct += 1
                else:
                    all_correct = False
                    break  # EARLY EXIT - key optimization!

            if all_correct:
                # Full match
                results.append({
                    'program': str(program),
                    'log_probability': log_prob,
                    'programs_enumerated': programs_tried,
                    'time_found': time.time() - start_time
                })

                # Stop after finding 5 solutions
                if len(results) >= 5:
                    break

        except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError) as e:
            # Expected evaluation errors - continue to next program
            continue
        except Exception as e:
            # Unexpected error - log but continue enumeration
            print(f"Unexpected error evaluating program: {type(e).__name__}: {e}", file=sys.stderr)
            continue

    return {
        'task_name': task_name,
        'solved': len(results) > 0,
        'n_solutions': len(results),
        'programs_searched': programs_tried,
        'time': time.time() - start_time,
        'solutions': results
    }


def main():
    """Main worker entry point."""
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())

    task_data = input_data['task']
    grammar_productions = input_data.get('grammar_productions')
    max_depth = input_data['max_depth']
    max_programs = input_data['max_programs']
    timeout = input_data['timeout']

    result = enumerate_task(task_data, grammar_productions, max_depth, max_programs, timeout)

    # Write result to stdout
    print(json.dumps(result))


if __name__ == '__main__':
    main()
