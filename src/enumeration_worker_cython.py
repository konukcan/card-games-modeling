#!/usr/bin/env python3
"""PyPy Worker for Enumeration"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.lean_primitives import build_lean_grammar
from rules.cards import Card, Suit, Rank


def deserialize_hand(hand_data):
    cards = []
    for card_dict in hand_data:
        suit = Suit[card_dict['suit']]
        rank = Rank[card_dict['rank']]
        cards.append(Card(suit, rank))
    return tuple(cards)


def evaluate_program(program, hand):
    try:
        fn = program.evaluate([])
        return fn(hand)
    except:
        return None


def enumerate_task(task_data, max_depth, max_programs, timeout):
    task_name = task_data['name']
    raw_examples = task_data['examples']

    examples = []
    for hand_data, expected in raw_examples:
        hand = deserialize_hand(hand_data)
        examples.append((hand, expected))

    grammar = build_lean_grammar()
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

        try:
            all_correct = True
            for inp, expected in examples:
                result = evaluate_program(program, inp)
                if result != expected:
                    all_correct = False
                    break

            if all_correct:
                results.append({
                    'program': str(program),
                    'log_probability': log_prob,
                    'programs_enumerated': programs_tried,
                    'time_found': time.time() - start_time
                })
                if len(results) >= 5:
                    break

        except:
            pass

    return {
        'task_name': task_name,
        'solved': len(results) > 0,
        'n_solutions': len(results),
        'programs_searched': programs_tried,
        'time': time.time() - start_time,
        'solutions': results
    }


def main():
    input_data = json.loads(sys.stdin.read())
    result = enumerate_task(
        input_data['task'],
        input_data['max_depth'],
        input_data['max_programs'],
        input_data['timeout']
    )
    print(json.dumps(result))


if __name__ == '__main__':
    main()
