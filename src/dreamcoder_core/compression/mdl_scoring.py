"""
MDL (Minimum Description Length) scoring for compression.

The Minimum Description Length principle says the best model (grammar) is one
that minimizes:

    MDL = DL(grammar) + Σ DL(program_i | grammar)

Where:
    - DL(grammar) = cost of encoding the grammar itself
    - DL(program_i | grammar) = cost of encoding program i using the grammar

This formalizes Occam's Razor: simpler explanations are better.

COMPARISON TO HEURISTIC SCORING:
    Heuristic: savings = (size - 1) × (count - 1)
    MDL:       improvement = old_MDL - new_MDL

    MDL is more principled because it accounts for:
    1. Grammar expansion cost (new production)
    2. Type complexity of the abstraction
    3. Actual encoding cost change, not just AST size

COMPARISON TO ORIGINAL DREAMCODER:
    Original uses sophisticated MDL computation with:
    - Rational number encoding for types
    - Log-probability encoding for program bodies
    - Version space compression

    Our implementation uses simplified proxies but captures the key insight:
    abstractions must earn their grammar cost through program savings.

EXTRACTED FROM: compression.py lines 1621-1890
"""

import math
from typing import Any, Dict, List, Tuple

from ..program import Program, Invented
from ..grammar import Grammar, Production
from ..type_system import Type, TypeContext

# Import from sibling modules
from .data_structures import SubtreeOccurrence
from .quality_filters import passes_abstraction_quality_checks
from .subtree_finding import abstract_subtree
from .rewriting import rewrite_with_invention_detailed


def compute_mdl(
    grammar: Grammar,
    programs: List[Program],
    request_type: Type,
    grammar_weight: float = 1.0
) -> float:
    """
    Compute the full MDL objective.

    MDL = λ × DL(grammar) + Σ DL(program_i | grammar)

    Args:
        grammar: The grammar to evaluate
        programs: Programs to include in the corpus DL
        request_type: Type of the programs (e.g., HAND → BOOL)
        grammar_weight: λ parameter controlling grammar complexity penalty.
                       Higher values = prefer simpler grammars.
                       Default: 1.0 (equal weight)

    Returns:
        Total MDL score (lower is better)

    PARAMETER TUNING:
        grammar_weight < 1: Favor adding abstractions (more aggressive compression)
        grammar_weight = 1: Balanced (default)
        grammar_weight > 1: Favor simpler grammar (more conservative)

    EXAMPLE:
        # Current grammar with 5 primitives, no inventions
        # Programs: [(+ 1 1), (+ 1 2), (+ 1 3), (+ 1 (+ 1 1))]

        old_mdl = compute_mdl(grammar, programs, INT)

        # Add invention: #add1 = (λ (+ 1 $0))
        # Rewrite programs to use it

        new_mdl = compute_mdl(new_grammar, rewritten_programs, INT)

        # If new_mdl < old_mdl, the invention was worthwhile
    """
    grammar_dl = grammar.grammar_description_length()
    programs_dl = sum(grammar.description_length(p, request_type) for p in programs)

    return grammar_weight * grammar_dl + programs_dl


def compute_mdl_detailed(
    grammar: Grammar,
    programs: List[Program],
    request_type: Type,
    grammar_weight: float = 1.0
) -> Dict[str, float]:
    """
    Compute MDL with detailed breakdown.

    Returns dictionary with:
        - grammar_dl: Description length of grammar
        - programs_dl: Sum of program description lengths
        - total_mdl: Weighted sum
        - n_programs: Number of programs
        - n_productions: Number of grammar productions
        - n_invented: Number of invented abstractions
    """
    grammar_dl = grammar.grammar_description_length()
    program_dls = [grammar.description_length(p, request_type) for p in programs]
    programs_dl = sum(program_dls)

    return {
        'grammar_dl': grammar_dl,
        'programs_dl': programs_dl,
        'total_mdl': grammar_weight * grammar_dl + programs_dl,
        'n_programs': len(programs),
        'avg_program_dl': programs_dl / len(programs) if programs else 0,
        'n_productions': len(grammar.productions),
        'n_invented': grammar.invented_count(),
        'grammar_weight': grammar_weight
    }


def evaluate_invention_mdl(
    grammar: Grammar,
    programs: List[Program],
    invention: Invented,
    target: Program,
    n_args: int,
    request_type: Type,
    grammar_weight: float = 1.0
) -> Tuple[float, float, List[Program], Dict[str, Any]]:
    """
    Evaluate the MDL change from adding an invention.

    This is the core decision function: should we add this abstraction?
    If MDL improves (decreases), the invention is worthwhile.

    Args:
        grammar: Current grammar (without the invention)
        programs: Current programs (not yet rewritten)
        invention: The proposed invented abstraction
        target: The subtree pattern the invention replaces
        n_args: Number of arguments the invention takes
        request_type: Type of the programs
        grammar_weight: Weight for grammar complexity penalty

    Returns:
        (old_mdl, new_mdl, rewritten_programs, stats) where:
        - old_mdl: MDL before adding invention
        - new_mdl: MDL after adding invention and rewriting
        - rewritten_programs: Programs rewritten to use invention
        - stats: Detailed statistics about the change

    DECISION RULE:
        if new_mdl < old_mdl:
            Accept invention (it improves compression)
        else:
            Reject invention (grammar cost exceeds program savings)

    WHAT'S COMPUTED:
        1. Current MDL = DL(grammar) + Σ DL(programs)
        2. Create new grammar with invention
        3. Rewrite all programs to use invention
        4. New MDL = DL(new_grammar) + Σ DL(rewritten_programs)
        5. Compare: is new_mdl < old_mdl?
    """
    # Step 1: Compute current MDL
    old_mdl = compute_mdl(grammar, programs, request_type, grammar_weight)
    old_grammar_dl = grammar.grammar_description_length()

    # Step 2: Create new grammar with invention
    ctx = TypeContext()
    tp = invention.infer_type(ctx, [])
    # Use a reasonable initial log probability
    log_prob = math.log(0.1)  # 10% prior probability
    new_grammar = grammar.with_production(Production(invention, tp, log_prob))
    new_grammar = new_grammar.normalize_probabilities()

    # Step 3: Rewrite all programs to use the invention
    rewritten_programs = []
    total_replacements = 0

    for prog in programs:
        result = rewrite_with_invention_detailed(prog, target, invention, n_args)
        rewritten_programs.append(result.program)
        total_replacements += result.n_replacements

    # Step 4: Compute new MDL
    new_mdl = compute_mdl(new_grammar, rewritten_programs, request_type, grammar_weight)
    new_grammar_dl = new_grammar.grammar_description_length()

    # Step 5: Compile statistics
    stats = {
        'old_mdl': old_mdl,
        'new_mdl': new_mdl,
        'mdl_improvement': old_mdl - new_mdl,
        'old_grammar_dl': old_grammar_dl,
        'new_grammar_dl': new_grammar_dl,
        'grammar_dl_increase': new_grammar_dl - old_grammar_dl,
        'old_programs_dl': old_mdl - grammar_weight * old_grammar_dl,
        'new_programs_dl': new_mdl - grammar_weight * new_grammar_dl,
        'programs_dl_decrease': (old_mdl - grammar_weight * old_grammar_dl) -
                                (new_mdl - grammar_weight * new_grammar_dl),
        'total_replacements': total_replacements,
        'programs_affected': sum(1 for i, prog in enumerate(programs)
                                 if prog != rewritten_programs[i]),
        'invention_body_size': invention.body.size(),
        'target_size': target.size()
    }

    return old_mdl, new_mdl, rewritten_programs, stats


def rank_inventions_by_mdl(
    grammar: Grammar,
    programs: List[Program],
    candidates: List[SubtreeOccurrence],
    request_type: Type,
    grammar_weight: float = 1.0,
    top_k: int = 10
) -> List[Tuple[Invented, int, Program, float, Dict[str, Any]]]:
    """
    Rank candidate inventions by MDL improvement.

    Args:
        grammar: Current grammar
        programs: Current programs
        candidates: Candidate subtrees from find_common_subtrees()
        request_type: Type of programs
        grammar_weight: MDL grammar weight
        top_k: Maximum number of candidates to return

    Returns:
        List of (invention, n_args, target, mdl_improvement, stats)
        sorted by mdl_improvement (highest first)
    """
    ranked = []

    for occ in candidates:
        # Create invention from subtree
        invention, n_args = abstract_subtree(occ.subtree)

        # Skip if already in grammar
        if grammar.get_production(invention) is not None:
            continue

        # Quality check: reject trivial or eta-reducible abstractions
        if not passes_abstraction_quality_checks(invention):
            continue

        # Check type inference works
        try:
            ctx = TypeContext()
            invention.infer_type(ctx, [])
        except Exception:
            continue

        # Evaluate MDL change
        old_mdl, new_mdl, _, stats = evaluate_invention_mdl(
            grammar, programs, invention, occ.subtree, n_args,
            request_type, grammar_weight
        )

        mdl_improvement = old_mdl - new_mdl

        # Only keep if there's improvement
        if mdl_improvement > 0:
            ranked.append((invention, n_args, occ.subtree, mdl_improvement, stats))

    # Sort by MDL improvement (highest first)
    ranked.sort(key=lambda x: -x[3])

    return ranked[:top_k]
