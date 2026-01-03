"""
Recognition-Guided Compression (DreamDecompiler).

Based on DreamDecompiler (Palmarini et al., ICML 2024).

The key insight: Instead of scoring candidate abstractions only by how much they
compress SOLVED programs (backward-looking), we also consider how useful they
might be for UNSOLVED tasks (forward-looking).

This is done by using the recognition model's predictions: if the recognition model
predicts that the primitives in a candidate abstraction are useful for unsolved tasks,
that abstraction should be scored higher.

Reference:
    Palmarini, A.B., Lucas, C.G., & Siddharth, N. (2024).
    "Bayesian Program Learning by Decompiling Amortized Knowledge."
    ICML 2024, PMLR 235:39042-39055.
    arXiv: https://arxiv.org/abs/2306.07856

EXTRACTED FROM: compression.py lines 3667-4100
"""

import math
from typing import List, Optional, Tuple, TYPE_CHECKING

from ..program import Program, Abstraction, Invented
from ..grammar import Grammar, Production
from ..type_system import TypeContext

# Import from sibling modules
from .data_structures import CompressionResult
from .quality_filters import passes_abstraction_quality_checks
from .anti_unification import find_anti_unified_patterns
from .subtree_finding import find_common_subtrees, abstract_subtree
from .rewriting import rewrite_all_frontiers
from .compress import compress_frontiers

if TYPE_CHECKING:
    from ..contrastive_recognition import ContrastiveRecognitionModel
    from ..task import Task


def compute_recognition_score(
    candidate_body: Program,
    unsolved_tasks: List['Task'],
    recognition_model: 'ContrastiveRecognitionModel',
    aggregation: str = 'mean'
) -> float:
    """
    Compute forward-looking recognition score for a candidate abstraction.

    This measures how useful the recognition model predicts this abstraction
    would be for unsolved tasks. Higher score = more useful.

    The score is based on the recognition model's predictions for the primitives
    that make up the candidate abstraction.

    Args:
        candidate_body: The body of the proposed invention (a Program)
        unsolved_tasks: List of tasks that haven't been solved yet
        recognition_model: Trained ContrastiveRecognitionModel
        aggregation: How to aggregate across tasks ('mean', 'max', 'sum')

    Returns:
        Recognition score (higher = more promising for unsolved tasks)

    Algorithm:
        1. Extract primitive names from candidate_body
        2. For each unsolved task:
           a. Get recognition predictions: Dict[str, float]
           b. Sum prediction scores for primitives in candidate
        3. Aggregate across tasks using specified method

    Reference:
        This is a simplified version of DreamDecompiler's caching benefit measure.
        Full DreamDecompiler also considers:
        - Program-level likelihoods
        - Bigram context (parent, arg position)
        - Multiple occurrences weighting

        Our simplified version just averages primitive predictions across tasks.
    """
    from ..program import collect_primitive_names

    if not unsolved_tasks or recognition_model is None:
        return 0.0

    # Extract primitive names from the candidate body
    primitives_in_candidate = collect_primitive_names(candidate_body)

    if not primitives_in_candidate:
        return 0.0

    task_scores = []
    for task in unsolved_tasks:
        try:
            # Get recognition model predictions for this task
            preds = recognition_model.predict_primitives_dict(task)

            # Sum predictions for primitives that appear in the candidate
            task_score = sum(preds.get(p, 0.0) for p in primitives_in_candidate)

            # Normalize by number of primitives (so larger candidates aren't unfairly advantaged)
            normalized_score = task_score / len(primitives_in_candidate)
            task_scores.append(normalized_score)

        except Exception:
            # If prediction fails for a task, skip it
            continue

    if not task_scores:
        return 0.0

    # Aggregate scores across tasks
    if aggregation == 'mean':
        return sum(task_scores) / len(task_scores)
    elif aggregation == 'max':
        return max(task_scores)
    elif aggregation == 'sum':
        return sum(task_scores)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_combined_score(
    backward_savings: float,
    forward_recognition_score: float,
    alpha: float = 0.7,
    backward_scale: float = 1.0,
    forward_scale: float = 1.0
) -> float:
    """
    Combine backward (MDL) and forward (recognition) scores.

    Args:
        backward_savings: MDL savings or heuristic savings (higher = better compression)
        forward_recognition_score: Recognition score (higher = more useful for unsolved)
        alpha: Weight for backward score (1-alpha for forward)
               - alpha=1.0: Pure backward (original DreamCoder)
               - alpha=0.7: Mostly backward, some forward (recommended)
               - alpha=0.5: Equal weighting
               - alpha=0.0: Pure forward (experimental)
        backward_scale: Normalization factor for backward score
        forward_scale: Normalization factor for forward score

    Returns:
        Combined score (higher = better candidate)
    """
    # Normalize scores to comparable ranges
    norm_backward = backward_savings / backward_scale if backward_scale > 0 else 0.0
    norm_forward = forward_recognition_score / forward_scale if forward_scale > 0 else 0.0

    return alpha * norm_backward + (1 - alpha) * norm_forward


def compress_frontiers_recognition(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    unsolved_tasks: Optional[List['Task']] = None,
    recognition_model: Optional['ContrastiveRecognitionModel'] = None,
    max_inventions: int = 5,
    min_savings: float = 2.0,
    use_anti_unification: bool = True,
    refactor_programs: bool = True,
    alpha: float = 0.7,
    recognition_threshold: float = 0.0
) -> CompressionResult:
    """
    Compress frontiers using combined backward (MDL) and forward (recognition) scoring.

    This implements recognition-guided compression based on DreamDecompiler
    (Palmarini et al., ICML 2024). Instead of selecting abstractions purely by
    compression savings, we also consider how useful they might be for unsolved tasks.

    Args:
        grammar: Current grammar
        frontiers: List of frontiers, each is [(program, log_likelihood), ...]
        unsolved_tasks: List of tasks that haven't been solved yet (for forward scoring)
        recognition_model: Trained recognition model (for forward scoring)
        max_inventions: Maximum number of new abstractions to add
        min_savings: Minimum backward savings to consider a candidate
        use_anti_unification: Whether to also find patterns via anti-unification
        refactor_programs: If True, rewrite programs after each invention
        alpha: Weight for backward vs forward scoring (0.7 = mostly backward)
        recognition_threshold: Minimum recognition score to boost a candidate

    Returns:
        CompressionResult with new grammar, inventions, and optionally rewritten frontiers

    ALGORITHM:
        1. Same as compress_frontiers for candidate generation
        2. For each candidate, compute:
           - Backward score: savings heuristic (compression)
           - Forward score: recognition model predictions for unsolved tasks
        3. Combine scores: alpha * backward + (1-alpha) * forward
        4. Select candidates by combined score
        5. Standard refactoring if enabled

    FALLBACK BEHAVIOR:
        If unsolved_tasks or recognition_model is None, falls back to
        pure backward scoring (equivalent to original compress_frontiers).

    Reference:
        Palmarini, A.B., Lucas, C.G., & Siddharth, N. (2024).
        "Bayesian Program Learning by Decompiling Amortized Knowledge."
        ICML 2024.
    """
    # If no recognition guidance available, fall back to standard compression
    use_recognition = (
        unsolved_tasks is not None and
        len(unsolved_tasks) > 0 and
        recognition_model is not None and
        alpha < 1.0  # alpha=1.0 means pure backward
    )

    if not use_recognition:
        # Fall back to standard compression
        return compress_frontiers(
            grammar=grammar,
            frontiers=frontiers,
            max_inventions=max_inventions,
            min_savings=min_savings,
            use_anti_unification=use_anti_unification,
            refactor_programs=refactor_programs
        )

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
                          'total_size_reduction': 0, 'recognition_guided': True}
                          if refactor_programs else None
        )

    # Working copies that get updated as we add inventions
    current_frontiers = [list(f) for f in frontiers]
    current_grammar = grammar

    # Track all inventions
    new_inventions: List[Invented] = []
    total_savings = 0.0

    # For computing scale factors
    all_backward_scores = []
    all_forward_scores = []

    # Aggregate rewrite statistics
    aggregate_rewrite_stats = {
        'total_replacements': 0,
        'programs_changed': 0,
        'total_size_reduction': 0,
        'inventions_applied': 0,
        'recognition_guided': True,
        'alpha': alpha
    }

    # Initial subtree analysis
    initial_common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    while len(new_inventions) < max_inventions:
        # Get current programs
        current_programs = []
        for frontier in current_frontiers:
            for prog, _ in frontier:
                current_programs.append(prog)

        # Find candidates (same as compress_frontiers)
        common = find_common_subtrees(current_programs, min_size=2, min_count=2)

        anti_unified_patterns = []
        if use_anti_unification and len(current_programs) >= 2:
            anti_unified_patterns = find_anti_unified_patterns(
                current_programs, min_uses=2
            )

        # Score all candidates with both backward and forward scores
        scored_candidates = []

        # Score exact subtree matches
        for occ in common:
            if occ.savings < min_savings:
                continue

            invention, n_args = abstract_subtree(occ.subtree)
            if current_grammar.get_production(invention) is not None:
                continue

            # Quality check: reject trivial or eta-reducible abstractions
            if not passes_abstraction_quality_checks(invention):
                continue

            try:
                ctx = TypeContext()
                invention.infer_type(ctx, [])
            except Exception:
                continue

            # Backward score (savings)
            backward = occ.savings

            # Forward score (recognition)
            forward = compute_recognition_score(
                occ.subtree,
                unsolved_tasks,
                recognition_model,
                aggregation='mean'
            )

            all_backward_scores.append(backward)
            all_forward_scores.append(forward)

            scored_candidates.append({
                'invention': invention,
                'n_args': n_args,
                'target': occ.subtree,
                'backward': backward,
                'forward': forward,
                'type': 'exact'
            })

        # Score anti-unified patterns
        for pattern, count, savings in anti_unified_patterns:
            if savings < min_savings:
                continue

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
                invention.infer_type(ctx, [])
            except Exception:
                continue

            backward = savings
            forward = compute_recognition_score(
                pattern,
                unsolved_tasks,
                recognition_model,
                aggregation='mean'
            )

            all_backward_scores.append(backward)
            all_forward_scores.append(forward)

            scored_candidates.append({
                'invention': invention,
                'n_args': n_vars,
                'target': pattern,
                'backward': backward,
                'forward': forward,
                'type': 'anti'
            })

        if not scored_candidates:
            break

        # Compute scale factors for normalization
        backward_scale = max(all_backward_scores) if all_backward_scores else 1.0
        forward_scale = max(all_forward_scores) if all_forward_scores else 1.0

        # Compute combined scores and select best
        best_candidate = None
        best_combined_score = -float('inf')

        for cand in scored_candidates:
            combined = compute_combined_score(
                cand['backward'],
                cand['forward'],
                alpha=alpha,
                backward_scale=backward_scale,
                forward_scale=forward_scale
            )

            # Only boost candidates with meaningful recognition scores
            if cand['forward'] < recognition_threshold:
                # If recognition score is below threshold, use pure backward
                combined = cand['backward'] / backward_scale if backward_scale > 0 else 0.0

            if combined > best_combined_score:
                best_combined_score = combined
                best_candidate = cand

        if best_candidate is None:
            break

        # Extract best candidate info
        invention = best_candidate['invention']
        n_args = best_candidate['n_args']
        target = best_candidate['target']
        backward_savings_val = best_candidate['backward']

        # Add invention to grammar
        ctx = TypeContext()
        tp = invention.infer_type(ctx, [])
        log_prob = math.log(0.1)
        current_grammar = current_grammar.with_production(
            Production(invention, tp, log_prob)
        )

        new_inventions.append(invention)
        total_savings += backward_savings_val

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

    return CompressionResult(
        new_inventions=new_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar.normalize_probabilities(),
        total_savings=total_savings,
        subtree_analysis=initial_common,
        rewritten_frontiers=current_frontiers if refactor_programs else None,
        rewrite_stats=aggregate_rewrite_stats if refactor_programs else None
    )
