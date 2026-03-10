"""
Exploration script: Enumeration efficiency tour and upstream trivial filtering.

Questions answered:
  1. How fast can we enumerate at depth 5-8? How many programs, what cost range?
  2. How many survive the trivial filter at each depth?
  3. What syntactic patterns characterize trivial programs?
     (double negation, and/or with constants, if-true, etc.)
  4. Could we block those patterns upstream in the enumerator?

Usage:
    cd src
    python -m gallery_analysis.explore_efficiency
"""
import sys
import time
import re
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank
from gallery_analysis.enumerator import enumerate_hypotheses
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.hypothesis_table import is_trivial, filter_trivial, compute_fingerprint


# =========================================================================
# PART 1: Enumeration speed and scale at depths 5-8
# =========================================================================

def profile_enumeration():
    """Profile enumeration at increasing depths."""
    print("=" * 70)
    print("PART 1: Enumeration speed and scale")
    print("=" * 70)

    # Load exemplars for the trivial filter
    print("\nLoading exemplars for trivial filter...")
    exemplars = load_exemplars()
    all_exemplar_hands = []
    for rule_id, data in exemplars.items():
        all_exemplar_hands.extend(data["hands_primary"])
    print(f"  {len(all_exemplar_hands)} exemplar hands from {len(exemplars)} rules")

    configs = [
        {"depth": 5, "max_cost": 25.0, "max_programs": 500_000, "timeout": 120},
        {"depth": 6, "max_cost": 30.0, "max_programs": 500_000, "timeout": 300},
        {"depth": 7, "max_cost": 35.0, "max_programs": 500_000, "timeout": 600},
        {"depth": 8, "max_cost": 40.0, "max_programs": 500_000, "timeout": 900},
    ]

    all_results = {}

    for cfg in configs:
        depth = cfg["depth"]
        print(f"\n{'─' * 70}")
        print(f"DEPTH {depth} (max_cost={cfg['max_cost']}, max_programs={cfg['max_programs']}, timeout={cfg['timeout']}s)")
        print(f"{'─' * 70}")

        # Enumerate
        t0 = time.time()
        programs = enumerate_hypotheses(
            max_depth=depth,
            max_programs=cfg["max_programs"],
            max_cost=cfg["max_cost"],
            timeout=cfg["timeout"],
        )
        enum_time = time.time() - t0

        if not programs:
            print("  No programs produced.")
            continue

        costs = [-lp for _, _, lp in programs]
        print(f"\n  Enumerated: {len(programs):,} programs in {enum_time:.1f}s")
        print(f"  Cost range: {min(costs):.1f} to {max(costs):.1f}")
        print(f"  Rate: {len(programs)/enum_time:,.0f} programs/sec")

        # Apply trivial filter
        t1 = time.time()
        survivors, stats = filter_trivial(programs, all_exemplar_hands)
        filter_time = time.time() - t1

        pct_trivial = 100 * (1 - stats["survivors"] / max(stats["total"], 1))
        print(f"\n  Trivial filter ({filter_time:.1f}s):")
        print(f"    Total:          {stats['total']:>8,}")
        print(f"    Trivial true:   {stats['trivial_true']:>8,}  ({100*stats['trivial_true']/max(stats['total'],1):.1f}%)")
        print(f"    Trivial false:  {stats['trivial_false']:>8,}  ({100*stats['trivial_false']/max(stats['total'],1):.1f}%)")
        print(f"    Survivors:      {stats['survivors']:>8,}  ({100*stats['survivors']/max(stats['total'],1):.1f}%)")

        # Cost distribution of survivors
        if survivors:
            surv_costs = [-lp for _, _, lp in survivors]
            print(f"\n  Survivor cost range: {min(surv_costs):.1f} to {max(surv_costs):.1f}")

        # Fingerprint the survivors to see equivalence class count
        if survivors and len(survivors) <= 50_000:
            t2 = time.time()
            probes = generate_probe_set(n_probes=500, seed=42)
            fingerprints = set()
            for _, pred_fn, _ in survivors:
                fp = compute_fingerprint(pred_fn, probes)
                fingerprints.add(fp)
            fp_time = time.time() - t2
            print(f"\n  Fingerprinting survivors ({fp_time:.1f}s, 500 probes):")
            print(f"    Unique fingerprints: {len(fingerprints):,} / {len(survivors):,}")
            print(f"    Dedup ratio: {100*(1 - len(fingerprints)/len(survivors)):.1f}%")

        all_results[depth] = {
            "programs": programs,
            "survivors": survivors,
            "stats": stats,
            "enum_time": enum_time,
            "filter_time": filter_time,
        }

    return all_results


# =========================================================================
# PART 2: Syntactic patterns of trivial programs
# =========================================================================

# Pattern matchers for program strings
TRIVIAL_PATTERNS = {
    "double_negation": r"\(not \(not ",
    "and_false": r"\(and .* false\)|\(and false ",
    "or_true": r"\(or .* true\)|\(or true ",
    "and_true_x": r"\(and true ",     # and true x ≡ x (identity, not always trivial itself)
    "or_false_x": r"\(or false ",     # or false x ≡ x (identity)
    "if_true": r"\(if true ",
    "if_false": r"\(if false ",
    "eq_self_const": r"\(eq (\w+) \1\)",  # eq X X (always true)
    "lt_zero": r"\(lt 0 0\)",        # 0 < 0 is always false
    "gt_self": r"\(gt (\d+) \1\)",   # n > n is always false
    "not_true": r"\(not true\)",     # always false
    "not_false": r"\(not false\)",   # always true
    "literal_true": r"^true$",       # just the constant true
    "literal_false": r"^false$",     # just the constant false
    "eq_same_const": r"\(eq (CLUBS|DIAMONDS|HEARTS|SPADES|RED|BLACK|[0-5]) \1\)",
}


def analyze_trivial_patterns(programs, all_exemplar_hands):
    """Analyze syntactic patterns in trivial programs."""
    print("\n" + "=" * 70)
    print("PART 2: Syntactic patterns of trivial programs")
    print("=" * 70)

    # Separate trivial from non-trivial
    trivial_progs = []
    non_trivial_progs = []

    for prog_str, pred_fn, log_prior in programs:
        if is_trivial(pred_fn, all_exemplar_hands):
            # Classify as always-true or always-false
            try:
                val = pred_fn(all_exemplar_hands[0])
            except Exception:
                val = False
            trivial_progs.append((prog_str, val, log_prior))
        else:
            non_trivial_progs.append((prog_str, pred_fn, log_prior))

    print(f"\n  Total programs: {len(programs):,}")
    print(f"  Trivial: {len(trivial_progs):,}")
    print(f"  Non-trivial: {len(non_trivial_progs):,}")

    # Count pattern occurrences in trivial programs
    print(f"\n  Syntactic pattern analysis (in trivial programs):")
    pattern_counts = Counter()
    pattern_examples = defaultdict(list)
    unmatched = []

    for prog_str, val, log_prior in trivial_progs:
        matched_any = False
        for pattern_name, regex in TRIVIAL_PATTERNS.items():
            if re.search(regex, prog_str):
                pattern_counts[pattern_name] += 1
                if len(pattern_examples[pattern_name]) < 3:
                    pattern_examples[pattern_name].append(prog_str)
                matched_any = True
        if not matched_any:
            unmatched.append((prog_str, val, log_prior))

    # Display results
    print(f"\n  {'Pattern':<25} {'Count':>8} {'%':>8}  Examples")
    print(f"  {'─'*25} {'─'*8} {'─'*8}  {'─'*40}")
    for pattern_name, count in pattern_counts.most_common():
        pct = 100 * count / max(len(trivial_progs), 1)
        examples = pattern_examples[pattern_name]
        ex_str = examples[0][:50] if examples else ""
        print(f"  {pattern_name:<25} {count:>8,} {pct:>7.1f}%  {ex_str}")

    total_matched = sum(pattern_counts.values())
    # Note: programs can match multiple patterns, so total_matched may exceed trivial count
    print(f"\n  Programs matching at least one pattern: check below")
    print(f"  Programs matching NO pattern: {len(unmatched):,}")

    # Show unmatched trivial programs (most interesting - what patterns are we missing?)
    if unmatched:
        print(f"\n  Unmatched trivial programs (first 30):")
        # Group by always-true vs always-false
        unmatched_true = [(p, v, lp) for p, v, lp in unmatched if v]
        unmatched_false = [(p, v, lp) for p, v, lp in unmatched if not v]
        print(f"    Always-true:  {len(unmatched_true):,}")
        print(f"    Always-false: {len(unmatched_false):,}")

        print(f"\n    Always-true examples (first 15):")
        for prog_str, _, lp in sorted(unmatched_true, key=lambda x: x[2])[:15]:
            print(f"      cost={-lp:.1f}  {prog_str[:80]}")

        print(f"\n    Always-false examples (first 15):")
        for prog_str, _, lp in sorted(unmatched_false, key=lambda x: x[2])[:15]:
            print(f"      cost={-lp:.1f}  {prog_str[:80]}")

    return trivial_progs, non_trivial_progs, unmatched


# =========================================================================
# PART 3: Deeper pattern analysis — what makes programs trivial?
# =========================================================================

def deeper_pattern_analysis(trivial_progs):
    """Analyze the structure of trivial programs more carefully."""
    print("\n" + "=" * 70)
    print("PART 3: Structural analysis of trivial programs")
    print("=" * 70)

    # Categorize by top-level structure
    top_level = Counter()
    for prog_str, val, lp in trivial_progs:
        # Extract outermost function application
        # Programs look like: (fn arg1 arg2) or just a constant
        stripped = prog_str.strip()
        if stripped.startswith("("):
            # Find the function name
            inner = stripped[1:]
            fn_name = inner.split()[0] if inner.split() else "?"
            top_level[fn_name] += 1
        else:
            top_level[stripped] += 1

    print(f"\n  Top-level function distribution (trivial programs):")
    for fn, count in top_level.most_common(20):
        pct = 100 * count / len(trivial_progs)
        print(f"    {fn:<20} {count:>8,}  ({pct:.1f}%)")

    # Analyze by "contains boolean constant" vs "doesn't"
    has_bool_const = 0
    has_not = 0
    has_and_or = 0
    for prog_str, val, lp in trivial_progs:
        if " true" in prog_str or " false" in prog_str or prog_str in ("true", "false"):
            has_bool_const += 1
        if "(not " in prog_str:
            has_not += 1
        if "(and " in prog_str or "(or " in prog_str:
            has_and_or += 1

    print(f"\n  Structural features:")
    print(f"    Contains boolean constant (true/false): {has_bool_const:,} ({100*has_bool_const/len(trivial_progs):.1f}%)")
    print(f"    Contains 'not':                         {has_not:,} ({100*has_not/len(trivial_progs):.1f}%)")
    print(f"    Contains 'and'/'or':                    {has_and_or:,} ({100*has_and_or/len(trivial_progs):.1f}%)")

    # Length distribution
    lengths = [len(prog_str) for prog_str, _, _ in trivial_progs]
    print(f"\n  Program string length distribution:")
    buckets = Counter()
    for l in lengths:
        if l <= 10:
            buckets["1-10"] += 1
        elif l <= 30:
            buckets["11-30"] += 1
        elif l <= 60:
            buckets["31-60"] += 1
        elif l <= 100:
            buckets["61-100"] += 1
        else:
            buckets["100+"] += 1
    for bucket in ["1-10", "11-30", "31-60", "61-100", "100+"]:
        count = buckets.get(bucket, 0)
        bar = "#" * min(count // 100, 50)
        print(f"    {bucket:>7}: {count:>8,}  {bar}")


# =========================================================================
# PART 4: Potential upstream filters
# =========================================================================

def summarize_upstream_opportunities(trivial_progs, unmatched):
    """Summarize what could be blocked upstream."""
    print("\n" + "=" * 70)
    print("PART 4: Upstream filtering opportunities")
    print("=" * 70)

    print("""
  CATEGORY A: Constant propagation (can detect syntactically)
  ──────────────────────────────────────────────────────────
  These produce a constant boolean regardless of input:
  - (and X false) → false          Block: and/or with boolean constant child
  - (or X true) → true
  - (not true) → false             Block: not applied to boolean constant
  - (not false) → true
  - (if true X Y) → X              Block: if with boolean constant condition
  - (if false X Y) → Y
  - (eq X X) → true                Block: eq with identical subtrees
  - (not (not X)) → X              Block: double negation (identity)

  CATEGORY B: Semantic triviality (harder to detect syntactically)
  ──────────────────────────────────────────────────────────────
  These are semantically trivial but don't have obvious syntactic markers:
  - (gt (length hand) 0) → always true (hand always has cards)
  - (has_suit hand HEARTS) ∨ (has_suit hand DIAMONDS) ∨ ... → always true
  - (le (n_unique_suits hand) 4) → always true
  - (ge (min_rank hand) 1) → always true (rank values ≥ 2)

  CATEGORY C: Redundant identity wrappers
  ───────────────────────────────────────
  - (and true X) ≡ X              Not trivial per se, but wasteful
  - (or false X) ≡ X
  - (if true X _) ≡ X
  These don't make the program trivial, but add depth without semantics.

  RECOMMENDATION:
  ──────────────
  Category A can be blocked in the enumerator with simple syntactic checks.
  Category B requires the exemplar-based trivial filter (what we have).
  Category C could be blocked for efficiency but doesn't affect correctness.
    """)

    # Quantify: how many trivial programs fall into Category A vs B?
    cat_a_count = 0
    for prog_str, val, lp in trivial_progs:
        # Check Category A patterns
        if any([
            re.search(r"\(and .* false\)|\(and false ", prog_str),
            re.search(r"\(or .* true\)|\(or true ", prog_str),
            re.search(r"\(not true\)|\(not false\)", prog_str),
            re.search(r"\(if true |\(if false ", prog_str),
            re.search(r"\(not \(not ", prog_str),
            prog_str in ("true", "false"),
        ]):
            cat_a_count += 1

    cat_b_count = len(trivial_progs) - cat_a_count
    print(f"  Quantification:")
    print(f"    Category A (syntactically detectable):  {cat_a_count:>8,} ({100*cat_a_count/len(trivial_progs):.1f}%)")
    print(f"    Category B (needs exemplar filter):     {cat_b_count:>8,} ({100*cat_b_count/len(trivial_progs):.1f}%)")
    print(f"    Total trivial:                          {len(trivial_progs):>8,}")

    print(f"\n  If we blocked Category A upstream, we'd enumerate ~{cat_a_count:,} fewer programs")
    print(f"  But the exemplar filter would still be needed for the remaining {cat_b_count:,}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("ENUMERATION EFFICIENCY TOUR")
    print("=" * 70)

    # Part 1: Profile enumeration at different depths
    all_results = profile_enumeration()

    # Use depth-5 results for pattern analysis (fast, representative)
    if 5 in all_results:
        programs = all_results[5]["programs"]

        # Load exemplars
        exemplars = load_exemplars()
        all_exemplar_hands = []
        for rule_id, data in exemplars.items():
            all_exemplar_hands.extend(data["hands_primary"])

        # Part 2: Syntactic patterns
        trivial_progs, non_trivial_progs, unmatched = analyze_trivial_patterns(
            programs, all_exemplar_hands
        )

        # Part 3: Deeper structural analysis
        deeper_pattern_analysis(trivial_progs)

        # Part 4: Upstream opportunities
        summarize_upstream_opportunities(trivial_progs, unmatched)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n  {'Depth':<8} {'Enumerated':>12} {'Time':>8} {'Survivors':>12} {'%Surv':>8} {'Filter':>8}")
    print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*8}")
    for depth in sorted(all_results.keys()):
        r = all_results[depth]
        n = len(r["programs"])
        s = r["stats"]["survivors"]
        pct = 100 * s / max(n, 1)
        print(f"  {depth:<8} {n:>12,} {r['enum_time']:>7.1f}s {s:>12,} {pct:>7.1f}% {r['filter_time']:>7.1f}s")

    print(f"\n{'='*70}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
