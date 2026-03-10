"""
Exploration script: Are always-true/always-false programs genuinely trivial,
or are they meaningful hypotheses with extreme base rates?

Strategy:
  - Take all programs that appear always-false on 500 random hands
  - Test them against exemplar hands from ALL 60 rules
  - If a program is still false on every single exemplar from every rule,
    it's genuinely trivial (semantically equivalent to `false`)
  - If it's true on SOME exemplars, it's a meaningful restrictive hypothesis
    that just happens to rarely fire on random hands

Same logic for always-true programs, but testing against "negative" hands
(random hands that we know violate most rules).

Also: timing test for fingerprinting with different probe set sizes.

Usage:
    cd src
    python -m gallery_analysis.explore_trivial_vs_rare
"""
import sys
import time
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank, H, D, S, C
from gallery_analysis.enumerator import enumerate_hypotheses
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.gallery_rules import GALLERY_RULES


def estimate_base_rate(pred_fn, n_samples=500, seed=42):
    """Quick base rate estimate."""
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    hits = 0
    for _ in range(n_samples):
        hand = rng.sample(deck, 6)
        try:
            if pred_fn(hand):
                hits += 1
        except Exception:
            pass
    return hits / n_samples


def main():
    print("Loading exemplars...")
    exemplars = load_exemplars()

    # Collect ALL exemplar hands from all 60 rules (360 hands total)
    all_exemplar_hands = []
    exemplar_rule_ids = []
    for rule_id, data in exemplars.items():
        for hand in data["hands_primary"]:
            all_exemplar_hands.append(hand)
            exemplar_rule_ids.append(rule_id)
    print(f"  {len(all_exemplar_hands)} total exemplar hands from {len(exemplars)} rules")

    # Enumerate at depth 5 (hits the 100K cap quickly)
    print("\nEnumerating programs at depth 5...")
    t0 = time.time()
    programs = enumerate_hypotheses(
        max_depth=5, max_programs=100_000, max_cost=25.0, timeout=120
    )
    print(f"  {len(programs)} programs in {time.time()-t0:.1f}s")

    # Classify into always-false, always-true, non-trivial
    print("\nClassifying by base rate (500 random hands)...")
    always_false = []  # base_rate <= 0.002
    always_true = []   # base_rate >= 0.998
    non_trivial = []
    for prog_str, pred_fn, log_prob in programs:
        br = estimate_base_rate(pred_fn, n_samples=500)
        if br <= 0.002:
            always_false.append((prog_str, pred_fn, log_prob, br))
        elif br >= 0.998:
            always_true.append((prog_str, pred_fn, log_prob, br))
        else:
            non_trivial.append((prog_str, pred_fn, log_prob, br))

    print(f"  Always-false: {len(always_false)}")
    print(f"  Always-true:  {len(always_true)}")
    print(f"  Non-trivial:  {len(non_trivial)}")

    # =========================================================================
    # PART 1: Analyze always-false programs against all 360 exemplar hands
    # =========================================================================
    print(f"\n{'='*70}")
    print("PART 1: Are always-false programs genuinely trivial?")
    print(f"{'='*70}")
    print("Testing each always-false program against all 360 exemplar hands...")

    genuinely_trivial_false = []  # true on 0 exemplar hands
    meaningful_rare = []          # true on at least 1 exemplar hand

    for prog_str, pred_fn, log_prob, br in always_false:
        hits_by_rule = defaultdict(int)
        total_hits = 0
        for hand, rule_id in zip(all_exemplar_hands, exemplar_rule_ids):
            try:
                if pred_fn(hand):
                    hits_by_rule[rule_id] += 1
                    total_hits += 1
            except Exception:
                pass

        if total_hits == 0:
            genuinely_trivial_false.append((prog_str, pred_fn, log_prob))
        else:
            meaningful_rare.append((prog_str, pred_fn, log_prob, total_hits, dict(hits_by_rule)))

    print(f"\n  Genuinely trivial (false on ALL 360 exemplars): {len(genuinely_trivial_false)}")
    print(f"  Meaningful but rare (true on some exemplars):   {len(meaningful_rare)}")
    pct = 100 * len(genuinely_trivial_false) / max(len(always_false), 1)
    print(f"  → {pct:.1f}% of always-false programs are genuinely trivial")

    # Show some examples of genuinely trivial false programs
    print(f"\n  Examples of genuinely trivial false programs (first 15):")
    for prog_str, _, log_prob in genuinely_trivial_false[:15]:
        print(f"    cost={-log_prob:.1f}  {prog_str}")

    # Show examples of meaningful rare programs
    print(f"\n  Examples of meaningful rare programs (first 15):")
    meaningful_rare.sort(key=lambda x: -x[3])  # sort by total hits
    for prog_str, _, log_prob, total_hits, hits_by_rule in meaningful_rare[:15]:
        top_rules = sorted(hits_by_rule.items(), key=lambda x: -x[1])[:3]
        rules_str = ", ".join(f"{r}:{n}" for r, n in top_rules)
        print(f"    cost={-log_prob:.1f}  hits={total_hits:>3}  rules=[{rules_str}]  {prog_str}")

    # Distribution of hit counts among meaningful rare programs
    if meaningful_rare:
        hit_counts = [x[3] for x in meaningful_rare]
        print(f"\n  Hit count distribution among meaningful rare programs:")
        buckets = Counter()
        for h in hit_counts:
            if h == 1:
                buckets["1"] += 1
            elif h <= 5:
                buckets["2-5"] += 1
            elif h <= 20:
                buckets["6-20"] += 1
            elif h <= 60:
                buckets["21-60"] += 1
            elif h <= 180:
                buckets["61-180"] += 1
            else:
                buckets["181-360"] += 1
        for bucket in ["1", "2-5", "6-20", "21-60", "61-180", "181-360"]:
            count = buckets.get(bucket, 0)
            bar = "#" * min(count, 50)
            print(f"    {bucket:>7} hits: {count:>5}  {bar}")

    # =========================================================================
    # PART 2: Analyze always-true programs
    # =========================================================================
    print(f"\n{'='*70}")
    print("PART 2: Are always-true programs genuinely trivial?")
    print(f"{'='*70}")

    # Generate "negative" test hands — hands specifically designed to violate rules
    # We'll use random hands (most random hands violate most rules) plus some edge cases
    rng = random.Random(99)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    negative_hands = [rng.sample(deck, 6) for _ in range(200)]

    # Also add some specific edge cases
    negative_hands.extend([
        [H("2"), S("3"), D("4"), C("5"), H("6"), S("7")],   # mixed suits, consecutive
        [S("A"), S("A"), S("A"), S("A"), S("A"), S("A")],    # impossible (same card), but tests
        [H("2"), H("3"), H("4"), H("5"), H("6"), H("7")],   # all hearts, consecutive
        [S("2"), H("2"), D("2"), C("2"), S("3"), H("3")],    # four of a kind + pair
    ])

    print(f"  Testing {len(always_true)} always-true programs against {len(negative_hands)} test hands...")

    genuinely_trivial_true = []   # true on ALL negative hands
    meaningful_permissive = []    # false on at least 1 negative hand

    for prog_str, pred_fn, log_prob, br in always_true:
        n_false = 0
        for hand in negative_hands:
            try:
                if not pred_fn(hand):
                    n_false += 1
            except Exception:
                n_false += 1  # errors count as "not always true"
        if n_false == 0:
            genuinely_trivial_true.append((prog_str, pred_fn, log_prob))
        else:
            meaningful_permissive.append((prog_str, pred_fn, log_prob, n_false))

    print(f"\n  Genuinely trivial (true on ALL {len(negative_hands)} test hands): {len(genuinely_trivial_true)}")
    print(f"  Meaningful but permissive (false on some):   {len(meaningful_permissive)}")
    pct = 100 * len(genuinely_trivial_true) / max(len(always_true), 1)
    print(f"  → {pct:.1f}% of always-true programs are genuinely trivial")

    print(f"\n  Examples of genuinely trivial true programs (first 15):")
    for prog_str, _, log_prob in genuinely_trivial_true[:15]:
        print(f"    cost={-log_prob:.1f}  {prog_str}")

    print(f"\n  Examples of meaningful permissive programs (first 15):")
    meaningful_permissive.sort(key=lambda x: -x[3])
    for prog_str, _, log_prob, n_false in meaningful_permissive[:15]:
        print(f"    cost={-log_prob:.1f}  false_on={n_false:>3}/{len(negative_hands)}  {prog_str}")

    # =========================================================================
    # PART 3: Timing test for fingerprinting at different probe set sizes
    # =========================================================================
    print(f"\n{'='*70}")
    print("PART 3: Fingerprinting timing at different probe set sizes")
    print(f"{'='*70}")

    # Use a subset of programs for timing (evaluating 100K programs × 100K probes
    # would be very slow, so we'll extrapolate from smaller tests)
    n_test_programs = 1000
    test_programs = programs[:n_test_programs]

    for n_probes in [200, 1_000, 10_000, 50_000, 100_000]:
        probes = generate_probe_set(n_probes=n_probes, seed=42)

        t0 = time.time()
        fingerprints = set()
        for prog_str, pred_fn, log_prob in test_programs:
            bits = []
            for hand in probes:
                try:
                    result = pred_fn(hand)
                    bits.append("1" if result else "0")
                except Exception:
                    bits.append("E")
            fp = "".join(bits)
            fingerprints.add(fp)
        elapsed = time.time() - t0

        # Extrapolate to 100K programs
        extrapolated = elapsed * (100_000 / n_test_programs)

        print(f"\n  {n_probes:>7} probes × {n_test_programs} programs:")
        print(f"    Time: {elapsed:.2f}s (extrapolated to 100K programs: {extrapolated:.0f}s = {extrapolated/60:.1f}min)")
        print(f"    Unique fingerprints: {len(fingerprints)} / {n_test_programs}")
        print(f"    Dedup ratio: {100*(1 - len(fingerprints)/n_test_programs):.1f}%")

    print(f"\n{'='*70}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
