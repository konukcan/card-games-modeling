"""
Load and validate injected hypotheses from JSON.

This module bridges external hypothesis sources (e.g., LLM-generated rules
translated into DSL) with the Bayesian scoring pipeline. It:

1. Loads hypotheses from a JSON file
2. Validates required fields
3. Parses each DSL program string into a Program AST
4. Creates callable predicates (hand -> bool) from the ASTs
5. Computes grammar-based log-priors for each hypothesis
6. Warns if any prior falls far outside the enumerated range

The main entry point is load_and_validate_injections().
"""

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.program import parse_program, Primitive, Program
from gallery_analysis.dsl_prior import compute_log_prior
from gallery_analysis.enumerator import build_gallery_grammar, _make_evaluator
from gallery_analysis.hypothesis_table import compute_fingerprint


# Fields that every injection entry must have
_REQUIRED_FIELDS = {"id", "source", "true_for_rule", "dsl_program"}


def load_and_validate_injections(
    filepath: str,
    grammar=None,
    warn_prior_threshold: float = 2.0,
    enumerated_prior_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Load injected hypotheses from JSON and validate them.

    For each entry in the JSON array:
    1. Verify it has 'id', 'source', 'true_for_rule', and 'dsl_program' fields
    2. Parse the DSL program string into a Program AST
    3. Create an executable predicate function (hand -> bool)
    4. Compute the grammar-based log-prior via compute_log_prior()
    5. Warn if the prior is far outside the enumerated range

    Args:
        filepath:               Path to the JSON file containing injected
                                hypotheses. Each entry is a dict with at least
                                the fields in _REQUIRED_FIELDS.
        grammar:                A Grammar object. If None, builds the default
                                gallery grammar (no true/false constants).
        warn_prior_threshold:   How far outside the enumerated range (in
                                log-prior units) before printing a warning.
                                Default 2.0 means warn if the prior is more
                                than 2 units below the most expensive
                                enumerated program or above the cheapest.
        enumerated_prior_range: Optional (min_lp, max_lp) tuple giving the
                                range of log-priors from enumeration. Used
                                to detect outlier priors. If None, no range
                                check is performed.

    Returns:
        A list of validated dicts. Each dict is the original JSON entry
        augmented with:
        - 'predicate': callable(Hand) -> bool
        - 'log_prior': float (always <= 0)
        - 'program':   parsed Program AST object

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError:        If any entry is missing the 'dsl_program' field,
                           or if a DSL program cannot be parsed.
    """
    # Build grammar if not provided
    if grammar is None:
        grammar = build_gallery_grammar()

    # Build primitive lookup dict for the parser.
    # parse_program expects a dict mapping name -> Primitive object.
    prim_dict = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            prim_dict[prod.program.name] = prod.program

    # Load JSON
    filepath = Path(filepath)
    with open(filepath) as f:
        raw_entries = json.load(f)

    validated = []

    for i, entry in enumerate(raw_entries):
        entry_id = entry.get("id", f"entry_{i}")

        # --- Check required fields ---
        missing = _REQUIRED_FIELDS - set(entry.keys())
        if "dsl_program" in missing:
            raise ValueError(
                f"Entry '{entry_id}' is missing required field 'dsl_program'"
            )
        if missing:
            # Other missing fields are warned but not fatal
            print(
                f"WARNING: Entry '{entry_id}' missing fields: {missing}",
                file=sys.stderr,
            )

        dsl_str = entry["dsl_program"]

        # --- Parse DSL into Program AST ---
        try:
            program = parse_program(dsl_str, prim_dict)
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Entry '{entry_id}': cannot parse DSL program "
                f"'{dsl_str}': {e}"
            ) from e

        # --- Create executable predicate ---
        predicate = _make_evaluator(program)

        # --- Compute log-prior ---
        try:
            log_prior = compute_log_prior(dsl_str, grammar)
        except ValueError as e:
            raise ValueError(
                f"Entry '{entry_id}': cannot compute log-prior for "
                f"'{dsl_str}': {e}"
            ) from e

        if not math.isfinite(log_prior):
            raise ValueError(
                f"Entry '{entry_id}': log-prior is {log_prior} for "
                f"'{dsl_str}' (expected finite negative value)"
            )

        # --- Warn if prior is outside enumerated range ---
        if enumerated_prior_range is not None:
            min_lp, max_lp = enumerated_prior_range
            if log_prior < min_lp - warn_prior_threshold:
                print(
                    f"WARNING: Entry '{entry_id}' has log_prior={log_prior:.2f}, "
                    f"which is {min_lp - log_prior:.1f} below the enumerated "
                    f"minimum ({min_lp:.2f}). This hypothesis is very expensive "
                    f"under the grammar.",
                    file=sys.stderr,
                )
            elif log_prior > max_lp + warn_prior_threshold:
                print(
                    f"WARNING: Entry '{entry_id}' has log_prior={log_prior:.2f}, "
                    f"which is {log_prior - max_lp:.1f} above the enumerated "
                    f"maximum ({max_lp:.2f}). This is suspiciously cheap.",
                    file=sys.stderr,
                )

        # --- Build validated entry ---
        validated_entry = dict(entry)  # shallow copy of original
        validated_entry["program"] = program
        validated_entry["predicate"] = predicate
        validated_entry["log_prior"] = log_prior

        validated.append(validated_entry)

    return validated


def _log_sum_exp(a: float, b: float) -> float:
    """
    Numerically stable log(exp(a) + exp(b)).

    Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    to avoid overflow/underflow when a or b are very negative.
    """
    mx = max(a, b)
    return mx + math.log(1 + math.exp(-abs(a - b)))


def merge_injected(
    equivalence_classes: List[Dict],
    injected: List[Dict],
    probes: List,
) -> List[Dict]:
    """
    Merge injected hypotheses into enumerated equivalence classes.

    For each injected hypothesis:
    1. Compute its fingerprint on the same probe hands used for enumeration
    2. If fingerprint matches an existing class: merge metadata + update prior
    3. If novel: create a new equivalence class

    This function does NOT modify the input lists. It returns a new list
    containing deep copies of existing classes (with merged metadata where
    applicable) plus new classes for novel injected hypotheses.

    Args:
        equivalence_classes: List of equivalence class dicts from enumeration.
            Each has keys: canonical_program, canonical_prior, summed_prior,
            n_expressions, all_programs, fingerprint, predicate.
        injected: List of validated injection dicts from
            load_and_validate_injections(). Each has keys: id, source,
            true_for_rule, dsl_program, predicate, log_prior, program.
        probes: The same list of probe hands used during enumeration
            fingerprinting. Required so injected hypotheses get comparable
            fingerprints.

    Returns:
        A new list of equivalence class dicts. Classes that absorbed an
        injected hypothesis gain 'source: "merged"', 'injection_ids', and
        optionally 'true_for_rule'. Novel injected hypotheses appear as
        new classes with 'source: "injected"'.
    """
    # Deep-copy existing classes so we don't mutate the caller's data.
    merged = copy.deepcopy(equivalence_classes)

    # Build lookup: fingerprint -> index in merged list.
    fp_to_idx: Dict[str, int] = {}
    for i, ec in enumerate(merged):
        fp_to_idx[ec["fingerprint"]] = i

    n_merged = 0   # injected hypotheses that matched an existing class
    n_novel = 0    # injected hypotheses that created a new class

    for inj in injected:
        # Compute fingerprint for the injected predicate on the same probes.
        fp = compute_fingerprint(inj["predicate"], probes)

        if fp in fp_to_idx:
            # ---- Merge into existing class ----
            idx = fp_to_idx[fp]
            ec = merged[idx]

            # Add program string to the class
            ec["all_programs"].append(inj["dsl_program"])
            ec["n_expressions"] += 1

            # Update summed prior: log(exp(old) + exp(new))
            ec["summed_prior"] = _log_sum_exp(
                ec["summed_prior"], inj["log_prior"]
            )

            # Track injection IDs
            ec.setdefault("injection_ids", []).append(inj.get("id", "unknown"))

            # Track true_for_rule if the injection represents a true rule
            if inj.get("true_for_rule"):
                ec["true_for_rule"] = inj["true_for_rule"]

            # Mark source as merged (was enumerated, now has injection too)
            ec["source"] = "merged"

            n_merged += 1

        else:
            # ---- Create new equivalence class ----
            new_class = {
                "canonical_program": inj["dsl_program"],
                "canonical_prior": inj["log_prior"],
                "summed_prior": inj["log_prior"],
                "n_expressions": 1,
                "all_programs": [inj["dsl_program"]],
                "fingerprint": fp,
                "predicate": inj["predicate"],
                "source": "injected",
                "injection_ids": [inj.get("id", "unknown")],
            }
            if inj.get("true_for_rule"):
                new_class["true_for_rule"] = inj["true_for_rule"]

            merged.append(new_class)
            fp_to_idx[fp] = len(merged) - 1

            n_novel += 1

    # Print merge statistics
    print(
        f"Injection merge: {len(injected)} hypotheses → "
        f"{n_merged} merged into existing classes, "
        f"{n_novel} novel classes created. "
        f"Total classes: {len(merged)} "
        f"(was {len(equivalence_classes)})"
    )

    return merged
