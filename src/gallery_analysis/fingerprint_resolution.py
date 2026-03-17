"""
Fingerprint resolution diagnostic: measure how probe count affects equivalence class resolution.

Currently 500 random probes are used for fingerprinting, but this is insufficient —
16 true rules collapse into a single equivalence class because they all accept 0/500
probes. This script measures how many probes are needed to fully resolve equivalence
classes, both with random-only probes and with targeted (exemplar + random) probes.

Usage:
    cd src
    python gallery_analysis/fingerprint_resolution.py \
        --output gallery_analysis/results/fingerprint_resolution.json \
        --verbose 1
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand
from gallery_analysis.enumerator import enumerate_hypotheses_with_stats
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.hypothesis_table import filter_trivial, compute_fingerprint
from gallery_analysis.injection import load_and_validate_injections


# Default injection file path (relative to this module)
_INJECTION_PATH = Path(__file__).parent / "data" / "injected_hypotheses.json"

# Probe counts to test (ascending so we can slice from the same large set)
DEFAULT_PROBE_COUNTS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def _fingerprint_all(
    programs: List[Tuple[str, Callable, float]],
    probes: List[Hand],
    verbose: int = 0,
    label: str = "",
) -> Dict[str, List[int]]:
    """
    Fingerprint all programs on the given probes.

    Returns a dict mapping fingerprint hex hash to list of program indices.
    Prints progress every 10000 programs if verbose >= 1.
    """
    fp_to_indices: Dict[str, List[int]] = {}
    n = len(programs)

    for i, (prog_str, pred_fn, log_prior) in enumerate(programs):
        fp = compute_fingerprint(pred_fn, probes)
        fp_to_indices.setdefault(fp, []).append(i)

        if verbose >= 1 and (i + 1) % 10000 == 0:
            print(
                f"  [{label}] Fingerprinted {i + 1:,}/{n:,} programs "
                f"({len(fp_to_indices):,} classes so far)",
                flush=True,
            )

    return fp_to_indices


def _count_true_resolved(
    true_rule_predicates: List[Tuple[str, Callable]],
    probes: List[Hand],
) -> Tuple[int, int]:
    """
    Count how many true-rule predicates have unique fingerprints.

    Returns:
        (n_resolved, n_total) where n_resolved = number of true rules
        whose fingerprint is not shared with any other true rule.
    """
    fps: Dict[str, List[str]] = {}
    for rule_id, pred_fn in true_rule_predicates:
        fp = compute_fingerprint(pred_fn, probes)
        fps.setdefault(fp, []).append(rule_id)

    # A true rule is "resolved" if its fingerprint is unique among true rules
    n_resolved = sum(1 for group in fps.values() if len(group) == 1)
    return n_resolved, len(true_rule_predicates)


def run_resolution_analysis(
    probe_counts: List[int],
    max_probes: int = 20000,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Run the full fingerprint resolution analysis.

    Steps:
        1. Enumerate programs and apply trivial filter (once)
        2. Load true-rule predicates from injected hypotheses
        3. Load exemplar hands for targeted probes
        4. Generate large random probe set
        5. For each probe count, fingerprint with random-only and targeted probes
        6. Measure number of equivalence classes and true-rule resolution

    Args:
        probe_counts: List of probe counts to test (e.g., [100, 500, 1000, ...]).
        max_probes: Maximum number of random probes to generate.
        verbose: 0=silent, 1=progress, 2=detailed.

    Returns:
        Results dict with probe_counts, random_only, targeted, n_survivors, n_true_rules.
    """
    results: Dict[str, Any] = {}
    t_total_start = time.time()

    # ---- Step 1: Enumerate + trivial filter (once) ----
    if verbose >= 1:
        print("=" * 70)
        print("Step 1: Enumerating programs...")
        print("=" * 70)

    t0 = time.time()
    # Use the same parameters as the main analysis pipeline (analyze.py defaults)
    programs, enum_stats = enumerate_hypotheses_with_stats(
        max_depth=7,
        max_programs=300_000,
        max_cost=35.0,
        timeout=600.0,
        max_list_chain=2,
    )
    t_enum = time.time() - t0

    if verbose >= 1:
        print(
            f"  Enumerated: {enum_stats['total_yielded']:,} yielded, "
            f"{enum_stats['syntactic_rejected']:,} syntactic rejected, "
            f"{enum_stats['accepted']:,} accepted ({t_enum:.1f}s)",
            flush=True,
        )

    if verbose >= 1:
        print("\nStep 2: Trivial filter on 360 curated exemplar hands...")

    t0 = time.time()
    exemplars = load_exemplars()
    all_exemplar_hands: List[Hand] = []
    for rule_id, data in exemplars.items():
        all_exemplar_hands.extend(data["hands_primary"])

    survivors, trivial_stats = filter_trivial(programs, all_exemplar_hands)
    t_trivial = time.time() - t0

    if verbose >= 1:
        print(
            f"  Input: {trivial_stats['total']:,}, "
            f"trivial_true: {trivial_stats['trivial_true']:,}, "
            f"trivial_false: {trivial_stats['trivial_false']:,}, "
            f"survivors: {trivial_stats['survivors']:,} ({t_trivial:.1f}s)",
            flush=True,
        )

    n_survivors = len(survivors)
    results["n_survivors"] = n_survivors

    # Free the full programs list (we only need survivors from here)
    del programs

    # ---- Step 2: Load true-rule predicates ----
    if verbose >= 1:
        print("\nStep 3: Loading true-rule predicates from injections...")

    injections = load_and_validate_injections(str(_INJECTION_PATH))
    true_rules = [
        (entry["id"], entry["predicate"])
        for entry in injections
        if entry["id"].startswith("true__")
    ]
    n_true = len(true_rules)
    results["n_true_rules"] = n_true

    if verbose >= 1:
        print(f"  Loaded {n_true} true-rule predicates")

    # ---- Step 3: Load exemplar hands for targeted probes ----
    # The 360 exemplar hands are already loaded above (all_exemplar_hands)
    n_exemplar = len(all_exemplar_hands)
    if verbose >= 1:
        print(f"\nStep 4: Using {n_exemplar} exemplar hands for targeted probes")

    # ---- Step 4: Generate large random probe set ----
    if verbose >= 1:
        print(f"\nStep 5: Generating {max_probes:,} random probes (seed=42)...")

    t0 = time.time()
    all_random_probes = generate_probe_set(max_probes, seed=42)
    t_probes = time.time() - t0

    if verbose >= 1:
        print(f"  Generated {len(all_random_probes):,} probes ({t_probes:.1f}s)")

    # ---- Step 5: Sweep probe counts ----
    # Filter probe_counts to those <= max_probes
    probe_counts = [n for n in probe_counts if n <= max_probes]
    results["probe_counts"] = probe_counts

    random_results = {"n_equiv_classes": [], "n_true_resolved": []}
    targeted_results = {"n_equiv_classes": [], "n_true_resolved": []}

    if verbose >= 1:
        print("\n" + "=" * 70)
        print("Step 6: Fingerprint resolution sweep")
        print("=" * 70)
        # Print header
        print(
            f"\n{'N_probes':>10} | {'Random: Classes':>16} | "
            f"{'Random: True':>13} | {'Targeted: Classes':>18} | "
            f"{'Targeted: True':>15} | {'Time':>8}"
        )
        print("-" * 95)

    for n_probes in probe_counts:
        t_step = time.time()

        # --- Random-only probes ---
        random_probes = all_random_probes[:n_probes]

        # Fingerprint all survivors
        fp_random = _fingerprint_all(
            survivors, random_probes, verbose=max(0, verbose - 1),
            label=f"random-{n_probes}",
        )
        n_classes_random = len(fp_random)

        # Count true-rule resolution
        n_resolved_random, _ = _count_true_resolved(true_rules, random_probes)

        random_results["n_equiv_classes"].append(n_classes_random)
        random_results["n_true_resolved"].append(n_resolved_random)

        # --- Targeted probes (exemplar + random) ---
        targeted_probes = all_exemplar_hands + random_probes

        fp_targeted = _fingerprint_all(
            survivors, targeted_probes, verbose=max(0, verbose - 1),
            label=f"targeted-{n_probes}",
        )
        n_classes_targeted = len(fp_targeted)

        n_resolved_targeted, _ = _count_true_resolved(true_rules, targeted_probes)

        targeted_results["n_equiv_classes"].append(n_classes_targeted)
        targeted_results["n_true_resolved"].append(n_resolved_targeted)

        dt = time.time() - t_step

        if verbose >= 1:
            print(
                f"{n_probes:>10,} | {n_classes_random:>16,} | "
                f"{n_resolved_random:>6}/{n_true:>2}     | "
                f"{n_classes_targeted:>18,} | "
                f"{n_resolved_targeted:>8}/{n_true:>2}    | "
                f"{dt:>7.1f}s",
                flush=True,
            )

    results["random_only"] = random_results
    results["targeted"] = targeted_results

    t_total = time.time() - t_total_start
    results["total_time_seconds"] = round(t_total, 1)

    if verbose >= 1:
        print(f"\nTotal time: {t_total:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure fingerprint resolution as a function of probe count."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gallery_analysis/results/fingerprint_resolution.json",
        help="Path to save results JSON (default: gallery_analysis/results/fingerprint_resolution.json)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=20000,
        help="Maximum number of random probes to generate (default: 20000)",
    )
    args = parser.parse_args()

    # Run the analysis
    results = run_resolution_analysis(
        probe_counts=DEFAULT_PROBE_COUNTS,
        max_probes=args.max_probes,
        verbose=args.verbose,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if args.verbose >= 1:
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
