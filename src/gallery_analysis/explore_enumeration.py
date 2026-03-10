"""
Exploration script: Run the enumerator at various depths and report statistics.

This helps us understand:
1. How many programs are produced at each depth?
2. How many pass exemplar filtering (for a sample rule)?
3. What do the programs look like?
4. How long does enumeration take?
5. What's the base rate distribution of enumerated programs?

Usage:
    cd src
    python -m gallery_analysis.explore_enumeration
"""
import sys
import time
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank, H, D, S, C
from gallery_analysis.enumerator import enumerate_hypotheses
from gallery_analysis.exemplars import load_exemplars, generate_probe_set


def evaluate_on_hands(pred_fn, hands):
    """Count hits/misses on a list of hands."""
    hits = 0
    misses = 0
    errors = 0
    for hand in hands:
        try:
            if pred_fn(hand):
                hits += 1
            else:
                misses += 1
        except Exception:
            errors += 1
    return hits, misses, errors


def estimate_base_rate(pred_fn, n_samples=1000, seed=42):
    """Quick base rate estimate using random hands."""
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    hits = 0
    errors = 0
    for _ in range(n_samples):
        hand = rng.sample(deck, 6)
        try:
            if pred_fn(hand):
                hits += 1
        except Exception:
            errors += 1
    return hits / n_samples, errors


def main():
    # Load exemplars for a few sample rules to test filtering
    print("Loading exemplars...")
    exemplars = load_exemplars()

    sample_rules = ["all_red", "all_same_suit", "strict_increasing"]
    sample_hands = {}
    for rule_id in sample_rules:
        if rule_id in exemplars:
            sample_hands[rule_id] = exemplars[rule_id]["hands_primary"]
            print(f"  {rule_id}: {len(sample_hands[rule_id])} exemplar hands")

    # Test enumeration at increasing depths
    configs = [
        {"max_depth": 3, "max_cost": 15.0, "timeout": 30},
        {"max_depth": 4, "max_cost": 20.0, "timeout": 60},
        {"max_depth": 5, "max_cost": 25.0, "timeout": 120},
        {"max_depth": 6, "max_cost": 30.0, "timeout": 300},
    ]

    for config in configs:
        depth = config["max_depth"]
        print(f"\n{'='*70}")
        print(f"DEPTH {depth} (max_cost={config['max_cost']}, timeout={config['timeout']}s)")
        print(f"{'='*70}")

        t0 = time.time()
        programs = enumerate_hypotheses(
            max_depth=depth,
            max_programs=100_000,  # high cap to see natural limits
            max_cost=config["max_cost"],
            timeout=config["timeout"],
        )
        elapsed = time.time() - t0

        print(f"\n  Total programs enumerated: {len(programs)}")
        print(f"  Time: {elapsed:.1f}s")
        if programs:
            costs = [-lp for _, _, lp in programs]
            print(f"  Cost range: {min(costs):.1f} to {max(costs):.1f}")

        if not programs:
            print("  No programs produced, skipping analysis.")
            continue

        # Show first 10 programs
        print(f"\n  First 10 programs:")
        for prog_str, pred_fn, log_prob in programs[:10]:
            br, errs = estimate_base_rate(pred_fn, n_samples=500)
            print(f"    cost={-log_prob:.1f}  base_rate={br:.3f}  {prog_str}")

        # Show last 10 programs (highest cost)
        if len(programs) > 20:
            print(f"\n  Last 10 programs (highest cost):")
            for prog_str, pred_fn, log_prob in programs[-10:]:
                br, errs = estimate_base_rate(pred_fn, n_samples=500)
                print(f"    cost={-log_prob:.1f}  base_rate={br:.3f}  {prog_str}")

        # Base rate distribution
        print(f"\n  Base rate distribution (sampled on 500 random hands):")
        base_rates = []
        always_true = 0
        always_false = 0
        error_programs = 0
        for prog_str, pred_fn, log_prob in programs:
            br, errs = estimate_base_rate(pred_fn, n_samples=500)
            if errs > 250:  # more than half are errors
                error_programs += 1
                continue
            base_rates.append(br)
            if br >= 0.998:
                always_true += 1
            elif br <= 0.002:
                always_false += 1

        print(f"    Always true (>99.8%):  {always_true} ({100*always_true/max(len(programs),1):.1f}%)")
        print(f"    Always false (<0.2%):   {always_false} ({100*always_false/max(len(programs),1):.1f}%)")
        print(f"    Error-heavy (>50% err): {error_programs} ({100*error_programs/max(len(programs),1):.1f}%)")
        print(f"    Non-trivial:            {len(base_rates) - always_true - always_false}")

        # Bucket the base rates
        if base_rates:
            buckets = Counter()
            for br in base_rates:
                if br <= 0.002:
                    buckets["0-0.2%"] += 1
                elif br <= 0.01:
                    buckets["0.2-1%"] += 1
                elif br <= 0.05:
                    buckets["1-5%"] += 1
                elif br <= 0.20:
                    buckets["5-20%"] += 1
                elif br <= 0.50:
                    buckets["20-50%"] += 1
                elif br <= 0.80:
                    buckets["50-80%"] += 1
                elif br <= 0.95:
                    buckets["80-95%"] += 1
                elif br <= 0.998:
                    buckets["95-99.8%"] += 1
                else:
                    buckets["99.8-100%"] += 1

            print(f"\n    Base rate buckets:")
            for bucket in ["0-0.2%", "0.2-1%", "1-5%", "5-20%", "20-50%",
                           "50-80%", "80-95%", "95-99.8%", "99.8-100%"]:
                count = buckets.get(bucket, 0)
                bar = "#" * min(count, 50)
                print(f"      {bucket:>10}: {count:>5}  {bar}")

        # Exemplar filtering: how many programs pass for each sample rule?
        print(f"\n  Exemplar filtering (how many programs cover all 6 exemplars?):")
        for rule_id, hands in sample_hands.items():
            n_all_hit = 0
            n_5of6 = 0
            n_4of6 = 0
            for prog_str, pred_fn, log_prob in programs:
                hits, misses, errors = evaluate_on_hands(pred_fn, hands)
                if hits == 6:
                    n_all_hit += 1
                elif hits >= 5:
                    n_5of6 += 1
                elif hits >= 4:
                    n_4of6 += 1

            print(f"    {rule_id}:")
            print(f"      6/6 hits: {n_all_hit}")
            print(f"      5/6 hits: {n_5of6}")
            print(f"      4/6 hits: {n_4of6}")
            total_viable = n_all_hit + n_5of6
            print(f"      Viable (>=5/6): {total_viable} ({100*total_viable/max(len(programs),1):.1f}%)")

    print(f"\n{'='*70}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
