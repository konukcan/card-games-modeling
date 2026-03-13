#!/usr/bin/env python3
"""Extract and deduplicate Gemini Flash hypotheses for injection into the Bayesian rule induction engine.

Loads translation results from the LLM phase-0 pipeline, filters to perfect-match
hypotheses (6/6 on gallery exemplar hands), deduplicates by hypothesis text within
each rule_id, and writes a consolidated JSON file.

Usage:
    python3 gallery_analysis/translate_llm_hypotheses.py

    # Or with a custom results directory:
    python3 gallery_analysis/translate_llm_hypotheses.py \
        --results-dir /path/to/phase0_translations
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def find_results_dir() -> Path:
    """Locate the phase0_translations directory.

    The LLM results live in the main repo at llm/results/phase0_translations/.
    Since this script runs inside a git worktree, we walk up from the script's
    location to find the card-games-modelling root (which contains llm/).

    The worktree layout is:
        card-games-modelling/
        ├── llm/results/phase0_translations/   <-- target
        └── .worktrees/bayesian-rule-induction/
            └── src/gallery_analysis/translate_llm_hypotheses.py  <-- this file

    So from __file__ we go up 4 levels to reach card-games-modelling/.
    """
    script_dir = Path(__file__).resolve().parent
    # script_dir = .../card-games-modelling/.worktrees/bayesian-rule-induction/src/gallery_analysis
    # Go up: gallery_analysis -> src -> bayesian-rule-induction -> .worktrees -> card-games-modelling
    repo_root = script_dir.parent.parent.parent.parent
    results_dir = repo_root / "llm" / "results" / "phase0_translations"
    return results_dir


def load_and_filter(results_dir: Path) -> list[dict]:
    """Load all gemini-flash JSON files and keep only perfect matches (6/6).

    Args:
        results_dir: Path to the phase0_translations directory.

    Returns:
        List of raw result dicts with match_rate == 1.0.
    """
    files = sorted(results_dir.glob("gemini-flash_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No gemini-flash_*.json files found in {results_dir}"
        )

    perfect = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        # Keep only hypotheses that matched all 6 gallery exemplar hands
        match_rate = data.get("semantics", {}).get("match_rate", 0.0)
        if match_rate == 1.0:
            perfect.append(data)

    return perfect


def deduplicate(entries: list[dict]) -> list[dict]:
    """Deduplicate by hypothesis text within each rule_id.

    Different generations and conditions may produce identical hypothesis text.
    We keep the first occurrence (arbitrary but deterministic since files are
    sorted alphabetically).

    Args:
        entries: List of perfect-match result dicts.

    Returns:
        Deduplicated list in the output schema.
    """
    # Track seen (rule_id, hypothesis_text) pairs
    seen: set[tuple[str, str]] = set()
    unique = []

    for entry in entries:
        rule_id = entry["rule_id"]
        hypothesis_text = entry["hypothesis"]
        key = (rule_id, hypothesis_text)

        if key in seen:
            continue
        seen.add(key)

        unique.append({
            "rule_id": rule_id,
            "rule_group": entry["rule_group"],
            "hypothesis_text": hypothesis_text,
            "python_lambda": entry["extracted_code"],
            "condition": entry["condition"],
            "source_model": entry["source_model"],
            "dsl_program": None,  # Placeholder — filled in Task 3 (DSL translation)
        })

    return unique


def print_summary(
    n_files: int,
    n_perfect: int,
    hypotheses: list[dict],
) -> None:
    """Print summary statistics to stdout."""
    # Count rules covered
    rules_covered = len({h["rule_id"] for h in hypotheses})

    # Per-rule distribution
    per_rule: dict[str, int] = defaultdict(int)
    for h in hypotheses:
        per_rule[h["rule_id"]] += 1

    avg_per_rule = len(hypotheses) / rules_covered if rules_covered else 0

    print("=" * 60)
    print("Gemini Flash Hypothesis Extraction Summary")
    print("=" * 60)
    print(f"  Files loaded:             {n_files}")
    print(f"  Perfect matches (6/6):    {n_perfect}")
    print(f"  Unique hypotheses:        {len(hypotheses)}")
    print(f"  Rules covered:            {rules_covered}")
    print(f"  Avg hypotheses per rule:  {avg_per_rule:.1f}")
    print()

    # Show rules with most/fewest hypotheses
    sorted_rules = sorted(per_rule.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5 rules (most hypotheses):")
    for rule_id, count in sorted_rules[:5]:
        print(f"    {rule_id}: {count}")
    print()
    print("  Bottom 5 rules (fewest hypotheses):")
    for rule_id, count in sorted_rules[-5:]:
        print(f"    {rule_id}: {count}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and deduplicate Gemini Flash hypotheses."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to phase0_translations directory. Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # Resolve results directory
    results_dir = args.results_dir or find_results_dir()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    print(f"Loading from: {results_dir}")

    # Load and filter
    files = sorted(results_dir.glob("gemini-flash_*.json"))
    n_files = len(files)
    perfect = load_and_filter(results_dir)

    # Deduplicate
    hypotheses = deduplicate(perfect)

    # Save output
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_hypotheses_raw.json"

    with open(output_path, "w") as f:
        json.dump(hypotheses, f, indent=2)

    print(f"Saved to: {output_path}")
    print()
    print_summary(n_files, len(perfect), hypotheses)


if __name__ == "__main__":
    main()
