"""
Compression (library learning) package.

This package provides program compression functionality for DreamCoder-style
library learning. It discovers common abstractions across programs and adds
them to the grammar.

REFACTORED FROM: The original monolithic compression.py (4,099 lines)

MODULE STRUCTURE:
-----------------
- data_structures.py:   SubtreeOccurrence, CompressionResult, CompressionState
- quality_filters.py:   Abstraction quality checks (nontrivial, eta-reducible, etc.)
- anti_unification.py:  Pattern discovery via anti-unification (LGG)
- subtree_finding.py:   Exact subtree matching
- arity_search.py:      Arity-aware factorization (Phase 4)
- rewriting.py:         Program rewriting with inventions
- mdl_scoring.py:       MDL (Minimum Description Length) scoring
- compress.py:          Main compression functions
- recognition_guided.py: DreamDecompiler recognition-guided compression (Phase 5)

BACKWARDS COMPATIBILITY:
------------------------
All existing imports continue to work:
    from dreamcoder_core.compression import compress_frontiers
    from dreamcoder_core.compression import SubtreeOccurrence, CompressionResult

TYPICAL USAGE:
--------------
    from dreamcoder_core.compression import compress_frontiers

    result = compress_frontiers(grammar, frontiers, max_inventions=5)
    print(f"Found {len(result.new_inventions)} new abstractions")

    # For MDL-based compression:
    from dreamcoder_core.compression import compress_frontiers_mdl
    result = compress_frontiers_mdl(grammar, frontiers, request_type)

    # For beam search (more thorough but slower):
    from dreamcoder_core.compression import beam_search_compression
    result = beam_search_compression(grammar, frontiers, request_type)

    # For recognition-guided compression (DreamDecompiler):
    from dreamcoder_core.compression import compress_frontiers_recognition
    result = compress_frontiers_recognition(
        grammar, frontiers,
        unsolved_tasks=unsolved,
        recognition_model=model,
        alpha=0.7
    )
"""

# Data structures
from .data_structures import (
    SubtreeOccurrence,
    CompressionResult,
    CompressionState,
)

# Quality filters
from .quality_filters import (
    is_nontrivial,
    is_eta_reducible,
    is_nested_eta_reducible,
    is_single_task_abstraction,
    passes_abstraction_quality_checks,
)

# Anti-unification
from .anti_unification import (
    anti_unify,
    find_anti_unified_patterns,
    create_abstraction_from_pattern,
)

# Subtree finding
from .subtree_finding import (
    find_common_subtrees,
    abstract_subtree,
)

# Arity-aware search
from .arity_search import (
    Factorization,
    enumerate_factorizations,
    abstract_subtree_partial,
    best_factorization,
    rank_factorizations_by_mdl,
)

# Program rewriting
from .rewriting import (
    RewriteResult,
    rewrite_with_invention,
    rewrite_with_invention_detailed,
    verify_rewrite_semantics,
    rewrite_and_verify,
    rewrite_frontier,
    rewrite_all_frontiers,
)

# MDL scoring
from .mdl_scoring import (
    compute_mdl,
    compute_mdl_detailed,
    evaluate_invention_mdl,
    rank_inventions_by_mdl,
)

# Main compression functions
from .compress import (
    compress_frontiers,
    compress_frontiers_mdl,
    beam_search_compression,
    beam_search_compression_with_arity,
    compress_frontiers_legacy,
    iterative_compression,
    compute_compression_ratio,
    format_invention,
    compression_report,
)

# Recognition-guided compression (DreamDecompiler)
from .recognition_guided import (
    compute_recognition_score,
    compute_combined_score,
    compress_frontiers_recognition,
)


# Define __all__ for explicit public API
__all__ = [
    # Data structures
    'SubtreeOccurrence',
    'CompressionResult',
    'CompressionState',

    # Quality filters
    'is_nontrivial',
    'is_eta_reducible',
    'is_nested_eta_reducible',
    'is_single_task_abstraction',
    'passes_abstraction_quality_checks',

    # Anti-unification
    'anti_unify',
    'find_anti_unified_patterns',
    'create_abstraction_from_pattern',

    # Subtree finding
    'find_common_subtrees',
    'abstract_subtree',

    # Arity-aware search
    'Factorization',
    'enumerate_factorizations',
    'abstract_subtree_partial',
    'best_factorization',
    'rank_factorizations_by_mdl',

    # Program rewriting
    'RewriteResult',
    'rewrite_with_invention',
    'rewrite_with_invention_detailed',
    'verify_rewrite_semantics',
    'rewrite_and_verify',
    'rewrite_frontier',
    'rewrite_all_frontiers',

    # MDL scoring
    'compute_mdl',
    'compute_mdl_detailed',
    'evaluate_invention_mdl',
    'rank_inventions_by_mdl',

    # Main compression functions
    'compress_frontiers',
    'compress_frontiers_mdl',
    'beam_search_compression',
    'beam_search_compression_with_arity',
    'compress_frontiers_legacy',
    'iterative_compression',
    'compute_compression_ratio',
    'format_invention',
    'compression_report',

    # Recognition-guided compression
    'compute_recognition_score',
    'compute_combined_score',
    'compress_frontiers_recognition',
]
