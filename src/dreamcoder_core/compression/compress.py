"""
Main compression functions.

This module provides the main entry points for compression (library learning):
- compress_frontiers() - heuristic-based compression (fast)
- compress_frontiers_mdl() - MDL-based compression (principled)
- beam_search_compression() - explores multiple compression paths
- iterative_compression() - multiple rounds for hierarchical abstractions

EXTRACTED FROM: compression.py lines 1890-2935
"""

import math
from typing import Any, Dict, List, Tuple

from ..program import Program, Abstraction, Invented
from ..grammar import Grammar, Production
from ..type_system import Type, TypeContext

# Import from sibling modules
from .data_structures import SubtreeOccurrence, CompressionResult, CompressionState
from .quality_filters import passes_abstraction_quality_checks
from .anti_unification import find_anti_unified_patterns
from .subtree_finding import find_common_subtrees, abstract_subtree
from .arity_search import best_factorization
from .rewriting import rewrite_all_frontiers, rewrite_with_invention_detailed
from .mdl_scoring import compute_mdl, evaluate_invention_mdl


def compress_frontiers(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_inventions: int = 5,
    min_savings: float = 2.0,
    use_anti_unification: bool = True,
    refactor_programs: bool = True
) -> CompressionResult:
    """
    Compress a set of frontiers by extracting common abstractions.

    This is the main compression function. It uses two complementary approaches:
    1. Exact subtree matching - finds identical subtrees across programs
    2. Anti-unification - finds structurally similar patterns that differ only
       in specific positions (e.g., same structure but different property accessor)

    Args:
        grammar: Current grammar
        frontiers: List of frontiers, each is [(program, log_likelihood), ...]
        max_inventions: Maximum number of new abstractions to add
        min_savings: Minimum description length savings to bother
        use_anti_unification: Whether to also find patterns via anti-unification
        refactor_programs: If True, rewrite all programs to use each new invention
                          before looking for the next one. This enables finding
                          HIERARCHICAL abstractions (patterns that use patterns).
                          Default: True (matching original DreamCoder behavior).

    Returns:
        CompressionResult with new grammar, inventions, and optionally rewritten
        frontiers (if refactor_programs=True).

    ALGORITHM:
        1. Collect all programs from frontiers
        2. Find common subtrees (exact matches)
        3. Find anti-unified patterns (structural similarity)
        4. Greedily select inventions by savings:
           a. Create invention from best candidate
           b. Add to grammar
           c. If refactor_programs: rewrite ALL programs to use invention
           d. Re-analyze rewritten programs for next candidate
        5. Return final grammar and rewritten frontiers

    PROGRAM REFACTORING (refactor_programs=True):
        After selecting each invention, we rewrite ALL programs across ALL
        frontiers to use the new abstraction. This is crucial because:

        1. HIERARCHICAL ABSTRACTIONS: Patterns that use other patterns.
           Round 1 finds: #inc = λx.(+ x 1)
           After rewriting: programs contain (#inc x) instead of (+ x 1)
           Round 2 can find: #add2 = λx.(#inc (#inc x))

        2. ACCURATE SAVINGS: The savings calculation assumes programs will
           be rewritten. If we don't rewrite, we overestimate future savings.

        3. GRAMMAR UPDATES: The inside-outside algorithm should see the
           shorter, rewritten programs.
    """
    # Collect all programs from frontiers
    all_programs = []
    for frontier in frontiers:
        for prog, _ in frontier:
            all_programs.append(prog)

    if not all_programs:
        return CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[],
            rewritten_frontiers=frontiers if refactor_programs else None,
            rewrite_stats={'total_replacements': 0, 'programs_changed': 0,
                          'total_size_reduction': 0} if refactor_programs else None
        )

    # Working copies that get updated as we add inventions
    current_frontiers = [list(f) for f in frontiers]  # Deep copy
    current_grammar = grammar

    # Track all inventions and their targets (for rewriting)
    new_inventions: List[Invented] = []
    invention_targets: List[Tuple[Program, int]] = []  # (target, n_args) pairs
    total_savings = 0.0

    # Aggregate rewrite statistics
    aggregate_rewrite_stats = {
        'total_replacements': 0,
        'programs_changed': 0,
        'total_size_reduction': 0,
        'inventions_applied': 0
    }

    # Initial subtree analysis (before any rewriting)
    initial_common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    while len(new_inventions) < max_inventions:
        # Get current programs from current frontiers
        current_programs = []
        for frontier in current_frontiers:
            for prog, _ in frontier:
                current_programs.append(prog)

        # Find common subtrees in CURRENT (possibly rewritten) programs
        common = find_common_subtrees(current_programs, min_size=2, min_count=2)

        # Also find anti-unified patterns
        anti_unified_patterns = []
        if use_anti_unification and len(current_programs) >= 2:
            anti_unified_patterns = find_anti_unified_patterns(
                current_programs, min_uses=2
            )

        # Find the best candidate (highest savings)
        best_candidate = None
        best_savings = min_savings
        best_type = None  # 'exact' or 'anti'

        # Check exact subtree matches
        for occ in common:
            if occ.savings > best_savings:
                invention, n_args = abstract_subtree(occ.subtree)
                if current_grammar.get_production(invention) is None:
                    # Quality check: reject trivial or eta-reducible abstractions
                    if not passes_abstraction_quality_checks(invention):
                        continue
                    # Verify type inference works
                    try:
                        ctx = TypeContext()
                        invention.infer_type(ctx, [])
                        best_candidate = (invention, n_args, occ.subtree, occ.savings)
                        best_savings = occ.savings
                        best_type = 'exact'
                    except Exception:
                        continue

        # Check anti-unified patterns
        for pattern, count, savings in anti_unified_patterns:
            if savings > best_savings:
                free_indices = pattern.free_indices()
                n_vars = len(free_indices)
                if n_vars == 0:
                    continue

                # Create invention
                body = pattern
                for _ in range(n_vars):
                    body = Abstraction(body)
                invention = Invented(body)

                if current_grammar.get_production(invention) is None:
                    # Quality check: reject trivial or eta-reducible abstractions
                    if not passes_abstraction_quality_checks(invention):
                        continue
                    try:
                        ctx = TypeContext()
                        invention.infer_type(ctx, [])
                        best_candidate = (invention, n_vars, pattern, savings)
                        best_savings = savings
                        best_type = 'anti'
                    except Exception:
                        continue

        # If no good candidate found, we're done
        if best_candidate is None:
            break

        # Unpack the best candidate
        invention, n_args, target, savings = best_candidate

        # Add invention to grammar
        ctx = TypeContext()
        tp = invention.infer_type(ctx, [])
        log_prob = math.log(0.1)  # 10% prior probability
        current_grammar = current_grammar.with_production(
            Production(invention, tp, log_prob)
        )

        new_inventions.append(invention)
        invention_targets.append((target, n_args))
        total_savings += savings

        # Refactor programs if enabled
        if refactor_programs:
            rewritten_frontiers, stats = rewrite_all_frontiers(
                current_frontiers, target, invention, n_args
            )
            current_frontiers = rewritten_frontiers

            aggregate_rewrite_stats['total_replacements'] += stats['total_replacements']
            aggregate_rewrite_stats['programs_changed'] += stats['programs_changed']
            aggregate_rewrite_stats['total_size_reduction'] += stats['total_size_reduction']
            aggregate_rewrite_stats['inventions_applied'] += 1

    # Build final result
    return CompressionResult(
        new_inventions=new_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar.normalize_probabilities(),
        total_savings=total_savings,
        subtree_analysis=initial_common,
        rewritten_frontiers=current_frontiers if refactor_programs else None,
        rewrite_stats=aggregate_rewrite_stats if refactor_programs else None
    )


def compress_frontiers_mdl(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    request_type: Type,
    max_inventions: int = 5,
    grammar_weight: float = 1.0,
    min_mdl_improvement: float = 0.0,
    refactor_programs: bool = True
) -> CompressionResult:
    """
    Compression using full MDL (Minimum Description Length) scoring.

    This is the principled alternative to heuristic-based compression.
    Instead of using (size-1)x(count-1) as a proxy for savings, it
    computes the actual description length change.

    Args:
        grammar: Current grammar
        frontiers: List of frontiers, each is [(program, log_likelihood), ...]
        request_type: Type of the programs (needed for DL computation)
        max_inventions: Maximum number of new abstractions to add
        grammar_weight: Lambda parameter for grammar complexity penalty (default 1.0)
        min_mdl_improvement: Minimum MDL decrease required to accept invention
        refactor_programs: If True, rewrite programs after each invention

    Returns:
        CompressionResult with new grammar, inventions, and rewritten frontiers

    MDL OBJECTIVE:
        MDL = lambda * DL(grammar) + Sum DL(program_i | grammar)

        We accept an invention if:
            MDL(old) - MDL(new) >= min_mdl_improvement
    """
    # Collect all programs from frontiers
    all_programs = []
    for frontier in frontiers:
        for prog, _ in frontier:
            all_programs.append(prog)

    if not all_programs:
        return CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[],
            rewritten_frontiers=frontiers if refactor_programs else None,
            rewrite_stats={'total_mdl_improvement': 0.0} if refactor_programs else None
        )

    # Working copies
    current_frontiers = [list(f) for f in frontiers]  # Deep copy
    current_programs = list(all_programs)
    current_grammar = grammar

    # Track inventions and statistics
    new_inventions: List[Invented] = []
    invention_targets: List[Tuple[Program, int]] = []
    total_mdl_improvement = 0.0

    # Initial MDL
    initial_mdl = compute_mdl(grammar, all_programs, request_type, grammar_weight)

    # Aggregate statistics
    mdl_stats = {
        'initial_mdl': initial_mdl,
        'final_mdl': initial_mdl,
        'total_mdl_improvement': 0.0,
        'inventions_evaluated': 0,
        'inventions_accepted': 0,
        'inventions_rejected': 0,
        'per_invention_stats': []
    }

    # Initial subtree analysis
    initial_common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    while len(new_inventions) < max_inventions:
        # Find candidate subtrees in current programs
        common = find_common_subtrees(current_programs, min_size=2, min_count=2)

        if not common:
            break

        # Evaluate all candidates by MDL
        best_candidate = None
        best_improvement = min_mdl_improvement
        best_rewritten = None
        best_stats = None

        for occ in common:
            # Create invention
            invention, n_args = abstract_subtree(occ.subtree)

            # Skip if already in grammar
            if current_grammar.get_production(invention) is not None:
                continue

            # Quality check: reject trivial or eta-reducible abstractions
            if not passes_abstraction_quality_checks(invention):
                continue

            # Check type inference
            try:
                ctx = TypeContext()
                invention.infer_type(ctx, [])
            except Exception:
                continue

            mdl_stats['inventions_evaluated'] += 1

            # Evaluate MDL change
            old_mdl, new_mdl, rewritten, stats = evaluate_invention_mdl(
                current_grammar, current_programs, invention,
                occ.subtree, n_args, request_type, grammar_weight
            )

            improvement = old_mdl - new_mdl

            if improvement > best_improvement:
                best_candidate = (invention, n_args, occ.subtree, improvement)
                best_improvement = improvement
                best_rewritten = rewritten
                best_stats = stats

        # If no good candidate found, we're done
        if best_candidate is None:
            break

        # Unpack best candidate
        invention, n_args, target, improvement = best_candidate

        # Add invention to grammar
        ctx = TypeContext()
        tp = invention.infer_type(ctx, [])
        log_prob = math.log(0.1)
        current_grammar = current_grammar.with_production(
            Production(invention, tp, log_prob)
        )
        current_grammar = current_grammar.normalize_probabilities()

        new_inventions.append(invention)
        invention_targets.append((target, n_args))
        total_mdl_improvement += improvement

        mdl_stats['inventions_accepted'] += 1
        mdl_stats['per_invention_stats'].append(best_stats)

        # Update programs if refactoring
        if refactor_programs and best_rewritten is not None:
            current_programs = best_rewritten

            # Also update frontiers
            current_frontiers, _ = rewrite_all_frontiers(
                current_frontiers, target, invention, n_args
            )

    # Final MDL
    final_mdl = compute_mdl(current_grammar, current_programs, request_type, grammar_weight)
    mdl_stats['final_mdl'] = final_mdl
    mdl_stats['total_mdl_improvement'] = initial_mdl - final_mdl
    mdl_stats['inventions_rejected'] = mdl_stats['inventions_evaluated'] - mdl_stats['inventions_accepted']

    return CompressionResult(
        new_inventions=new_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar.normalize_probabilities(),
        total_savings=total_mdl_improvement,  # Use MDL improvement as "savings"
        subtree_analysis=initial_common,
        rewritten_frontiers=current_frontiers if refactor_programs else None,
        rewrite_stats=mdl_stats if refactor_programs else None
    )


def beam_search_compression(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    request_type: Type,
    beam_width: int = 10,
    max_inventions: int = 5,
    grammar_weight: float = 1.0,
    candidates_per_state: int = 20,
    use_arity_search: bool = True,
    max_args: int = 4
) -> CompressionResult:
    """
    Beam search over compression choices.

    Maintains beam_width best states, explores adding each candidate
    invention to each state, keeps best results. This avoids local
    optima that greedy compression can get stuck in.

    Args:
        grammar: Current grammar
        frontiers: List of frontiers, each is [(program, log_likelihood), ...]
        request_type: Type of the programs
        beam_width: Number of best states to keep (default 10)
        max_inventions: Maximum total inventions to add (default 5)
        grammar_weight: MDL grammar weight (default 1.0)
        candidates_per_state: Max candidates to try per state (default 20)
        use_arity_search: If True, use arity-aware factorization
        max_args: Maximum arity for factorization (if use_arity_search=True)

    Returns:
        CompressionResult with best compression found
    """
    # Collect all programs from frontiers
    all_programs = []
    for frontier in frontiers:
        for prog, _ in frontier:
            all_programs.append(prog)

    if not all_programs:
        return CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[],
            rewritten_frontiers=frontiers,
            rewrite_stats={'beam_search': True, 'states_explored': 0}
        )

    # Initial MDL and state
    initial_mdl = compute_mdl(grammar, all_programs, request_type, grammar_weight)

    initial_state = CompressionState(
        grammar=grammar,
        programs=list(all_programs),
        inventions=[],
        targets=[],
        mdl=initial_mdl,
        history=["Initial state"]
    )

    # Track statistics
    stats = {
        'beam_search': True,
        'beam_width': beam_width,
        'initial_mdl': initial_mdl,
        'states_explored': 0,
        'candidates_evaluated': 0,
        'iterations': 0,
        'beam_history': []
    }

    # Initial subtree analysis
    initial_common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    # Beam starts with just the initial state
    beam = [initial_state]
    stats['beam_history'].append(initial_mdl)

    for iteration in range(max_inventions):
        stats['iterations'] += 1
        candidates = []

        for state in beam:
            stats['states_explored'] += 1

            # Find candidate subtrees in this state's programs
            common = find_common_subtrees(state.programs, min_size=2, min_count=2)

            # Limit candidates per state
            for occ in common[:candidates_per_state]:
                stats['candidates_evaluated'] += 1

                # Try different factorizations if arity search is enabled
                if use_arity_search:
                    fact_result = best_factorization(
                        occ.subtree, state.grammar, state.programs,
                        request_type, grammar_weight, max_args
                    )

                    if fact_result is None:
                        continue

                    fact, improvement, rewritten, inv_stats = fact_result
                    invention = fact.invention
                    n_args = fact.n_args

                    # Quality check for factorizations
                    if not passes_abstraction_quality_checks(invention):
                        continue

                    if improvement <= 0:
                        continue

                    new_mdl = state.mdl - improvement

                else:
                    # Standard abstraction (full arity)
                    invention, n_args = abstract_subtree(occ.subtree)

                    # Skip if already in grammar
                    if state.grammar.get_production(invention) is not None:
                        continue

                    # Quality check: reject trivial or eta-reducible abstractions
                    if not passes_abstraction_quality_checks(invention):
                        continue

                    # Check type inference
                    try:
                        ctx = TypeContext()
                        invention.infer_type(ctx, [])
                    except Exception:
                        continue

                    # Evaluate MDL change
                    try:
                        old_mdl, new_mdl, rewritten, inv_stats = evaluate_invention_mdl(
                            state.grammar, state.programs, invention,
                            occ.subtree, n_args, request_type, grammar_weight
                        )
                    except Exception:
                        continue

                    improvement = old_mdl - new_mdl

                    if improvement <= 0:
                        continue

                # Create new grammar with invention
                ctx = TypeContext()
                tp = invention.infer_type(ctx, [])
                log_prob = math.log(0.1)
                new_grammar = state.grammar.with_production(
                    Production(invention, tp, log_prob)
                )
                new_grammar = new_grammar.normalize_probabilities()

                # Create new state
                new_state = CompressionState(
                    grammar=new_grammar,
                    programs=rewritten,
                    inventions=state.inventions + [invention],
                    targets=state.targets + [(occ.subtree, n_args)],
                    mdl=new_mdl,
                    history=state.history + [
                        f"Added {invention} (improvement: {improvement:.2f})"
                    ]
                )

                candidates.append(new_state)

        if not candidates:
            # No improving candidates found
            break

        # Deduplicate candidates (same invention set = same state)
        seen = set()
        unique_candidates = []
        for c in candidates:
            inv_key = tuple(sorted(str(inv) for inv in c.inventions))
            if inv_key not in seen:
                seen.add(inv_key)
                unique_candidates.append(c)

        # Sort by MDL (lower is better) and keep top beam_width
        unique_candidates.sort(key=lambda s: s.mdl)
        beam = unique_candidates[:beam_width]

        # Track best MDL at this iteration
        stats['beam_history'].append(beam[0].mdl if beam else initial_mdl)

    # Return best state found
    best_state = min(beam, key=lambda s: s.mdl) if beam else initial_state

    stats['final_mdl'] = best_state.mdl
    stats['total_mdl_improvement'] = initial_mdl - best_state.mdl

    # Reconstruct frontiers with rewritten programs
    rewritten_frontiers = [list(f) for f in frontiers]
    for (target, n_args), invention in zip(best_state.targets, best_state.inventions):
        rewritten_frontiers, _ = rewrite_all_frontiers(
            rewritten_frontiers, target, invention, n_args
        )

    return CompressionResult(
        new_inventions=best_state.inventions,
        old_grammar=grammar,
        new_grammar=best_state.grammar.normalize_probabilities(),
        total_savings=initial_mdl - best_state.mdl,
        subtree_analysis=initial_common,
        rewritten_frontiers=rewritten_frontiers,
        rewrite_stats=stats
    )


def beam_search_compression_with_arity(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    request_type: Type,
    beam_width: int = 10,
    max_inventions: int = 5,
    grammar_weight: float = 1.0,
    candidates_per_state: int = 20,
    max_args: int = 4
) -> CompressionResult:
    """
    Convenience wrapper: beam search with arity-aware factorization enabled.

    Combines Phase 3 (Beam Search) and Phase 4 (Arity-Aware Search)
    for the most thorough compression.
    """
    return beam_search_compression(
        grammar=grammar,
        frontiers=frontiers,
        request_type=request_type,
        beam_width=beam_width,
        max_inventions=max_inventions,
        grammar_weight=grammar_weight,
        candidates_per_state=candidates_per_state,
        use_arity_search=True,
        max_args=max_args
    )


def compress_frontiers_legacy(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_inventions: int = 5,
    min_savings: float = 2.0,
    use_anti_unification: bool = True
) -> CompressionResult:
    """
    Legacy version of compress_frontiers WITHOUT program refactoring.

    Preserved for backwards compatibility and comparison testing.
    For new code, use compress_frontiers(refactor_programs=False) instead.
    """
    # Collect all programs from frontiers
    all_programs = []
    for frontier in frontiers:
        for prog, _ in frontier:
            all_programs.append(prog)

    if not all_programs:
        return CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[]
        )

    # Find common subtrees (exact matches)
    common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    # Also find anti-unified patterns (structural similarity)
    anti_unified_patterns = []
    if use_anti_unification and len(all_programs) >= 2:
        anti_unified_patterns = find_anti_unified_patterns(all_programs, min_uses=2)

    # Greedily select inventions
    new_inventions = []
    total_savings = 0.0
    current_grammar = grammar

    # First try exact subtree matches (usually higher confidence)
    for occ in common:
        if len(new_inventions) >= max_inventions:
            break

        if occ.savings < min_savings:
            break

        # Create the invention
        invention, n_args = abstract_subtree(occ.subtree)

        # Check if this is already in the grammar
        if current_grammar.get_production(invention) is not None:
            continue

        # Quality check: reject trivial or eta-reducible abstractions
        if not passes_abstraction_quality_checks(invention):
            continue

        # Add to grammar
        try:
            ctx = TypeContext()
            tp = invention.infer_type(ctx, [])
            log_prob = math.log(0.1)
            current_grammar = current_grammar.with_production(
                Production(invention, tp, log_prob)
            )
            new_inventions.append(invention)
            total_savings += occ.savings

        except Exception:
            continue

    # Then try anti-unified patterns (if we have room for more inventions)
    for pattern, count, savings in anti_unified_patterns:
        if len(new_inventions) >= max_inventions:
            break

        if savings < min_savings:
            break

        free_indices = pattern.free_indices()
        n_vars = len(free_indices)

        if n_vars == 0:
            continue

        body = pattern
        for _ in range(n_vars):
            body = Abstraction(body)

        invention = Invented(body)

        if current_grammar.get_production(invention) is not None:
            continue

        # Quality check: reject trivial or eta-reducible abstractions
        if not passes_abstraction_quality_checks(invention):
            continue

        try:
            ctx = TypeContext()
            tp = invention.infer_type(ctx, [])
            log_prob = math.log(0.1)
            current_grammar = current_grammar.with_production(
                Production(invention, tp, log_prob)
            )
            new_inventions.append(invention)
            total_savings += savings

        except Exception:
            continue

    return CompressionResult(
        new_inventions=new_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar.normalize_probabilities(),
        total_savings=total_savings,
        subtree_analysis=common
    )


def iterative_compression(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_rounds: int = 3,
    max_inventions_per_round: int = 3,
    refactor_programs: bool = True
) -> CompressionResult:
    """
    Perform multiple rounds of compression.

    Each round finds new abstractions based on the current (possibly rewritten) programs.
    Between rounds, frontiers are updated to use the newly learned abstractions,
    enabling the discovery of HIERARCHICAL ABSTRACTIONS.

    Args:
        grammar: Starting grammar
        frontiers: Task solutions
        max_rounds: Maximum compression iterations
        max_inventions_per_round: Cap on new abstractions per round
        refactor_programs: If True, use rewritten programs from each round as input
                          to the next round. This enables hierarchical abstractions.

    Returns:
        Aggregated CompressionResult with:
        - All inventions from all rounds
        - Final grammar with all inventions
        - Total savings across all rounds
        - Final rewritten frontiers (if refactor_programs=True)

    WHY MULTIPLE ROUNDS?
        Round 1 finds: #inc = (lambda (+ $0 1))
        Programs are rewritten: (+ (+ x 1) 1) becomes (#inc (#inc x))
        Round 2 finds: #add2 = (lambda (#inc (#inc $0)))

        Hierarchical abstractions build on each other.
    """
    current_grammar = grammar
    current_frontiers = [list(f) for f in frontiers]  # Deep copy
    all_inventions: List[Invented] = []
    total_savings = 0.0

    # Aggregate rewrite statistics across all rounds
    aggregate_rewrite_stats = {
        'total_replacements': 0,
        'programs_changed': 0,
        'total_size_reduction': 0,
        'inventions_applied': 0,
        'rounds_completed': 0
    }

    for round_num in range(max_rounds):
        result = compress_frontiers(
            current_grammar,
            current_frontiers,
            max_inventions=max_inventions_per_round,
            refactor_programs=refactor_programs
        )

        if not result.new_inventions:
            # No more compression opportunities
            break

        current_grammar = result.new_grammar
        all_inventions.extend(result.new_inventions)
        total_savings += result.total_savings
        aggregate_rewrite_stats['rounds_completed'] += 1

        # Update frontiers for next round (if refactoring is enabled)
        if refactor_programs and result.rewritten_frontiers is not None:
            current_frontiers = result.rewritten_frontiers

            # Aggregate rewrite stats
            if result.rewrite_stats:
                aggregate_rewrite_stats['total_replacements'] += result.rewrite_stats.get('total_replacements', 0)
                aggregate_rewrite_stats['programs_changed'] += result.rewrite_stats.get('programs_changed', 0)
                aggregate_rewrite_stats['total_size_reduction'] += result.rewrite_stats.get('total_size_reduction', 0)
                aggregate_rewrite_stats['inventions_applied'] += result.rewrite_stats.get('inventions_applied', 0)

    return CompressionResult(
        new_inventions=all_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar,
        total_savings=total_savings,
        subtree_analysis=[],
        rewritten_frontiers=current_frontiers if refactor_programs else None,
        rewrite_stats=aggregate_rewrite_stats if refactor_programs else None
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _has_index(program: Program) -> bool:
    """Check if program contains any Index nodes."""
    from ..program import Index, Primitive, Application, Abstraction

    if isinstance(program, Index):
        return True
    if isinstance(program, (Primitive, Invented)):
        return False
    if isinstance(program, Application):
        return _has_index(program.f) or _has_index(program.x)
    if isinstance(program, Abstraction):
        return _has_index(program.body)
    return False


def _count_max_index(program: Program) -> int:
    """Find the maximum Index value in a program."""
    from ..program import Index, Primitive, Application, Abstraction

    if isinstance(program, Index):
        return program.i
    if isinstance(program, (Primitive, Invented)):
        return -1
    if isinstance(program, Application):
        return max(_count_max_index(program.f), _count_max_index(program.x))
    if isinstance(program, Abstraction):
        return _count_max_index(program.body)
    return -1


def compute_compression_ratio(
    programs: List[Program],
    grammar: Grammar,
    new_grammar: Grammar,
    request_type: Type
) -> Tuple[float, float]:
    """
    Compute the compression ratio achieved.

    Returns:
        (old_total_dl, new_total_dl) description lengths in bits

    NOTE: This computes DL of ORIGINAL programs with old vs new grammar.
    For true MDL comparison, we should also rewrite programs and
    measure DL of rewritten programs with new grammar.
    """
    old_dl = sum(grammar.description_length(p, request_type) for p in programs)
    new_dl = sum(new_grammar.description_length(p, request_type) for p in programs)
    return old_dl, new_dl


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def format_invention(invention: Invented) -> str:
    """Format an invention for display."""
    if invention.name:
        return f"#{invention.name}: {invention.body}"
    return f"#({invention.body})"


def compression_report(result: CompressionResult) -> str:
    """Generate a human-readable compression report."""
    lines = ["=" * 60]
    lines.append("COMPRESSION REPORT")
    lines.append("=" * 60)

    lines.append(f"\nNew inventions: {len(result.new_inventions)}")
    for i, inv in enumerate(result.new_inventions):
        lines.append(f"  {i+1}. {format_invention(inv)}")

    lines.append(f"\nTotal savings: {result.total_savings:.2f} (heuristic, not true MDL)")

    if result.subtree_analysis:
        lines.append("\nTop common subtrees:")
        for occ in result.subtree_analysis[:10]:
            lines.append(f"  {occ}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
