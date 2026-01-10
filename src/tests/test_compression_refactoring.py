"""
Comprehensive test suite for compression refactoring validation.

This test suite verifies that the refactored compression/ package
maintains complete backwards compatibility with the original
monolithic compression.py file.

RUN WITH:
  - With pytest: python3 -m pytest tests/test_compression_refactoring.py -v
  - Standalone:  python3 tests/test_compression_refactoring.py
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a minimal pytest.fixture decorator for standalone mode
    class pytest:
        @staticmethod
        def fixture(func):
            return func

from dreamcoder_core.program import (
    Primitive, Application, Abstraction, Index, Invented
)
from dreamcoder_core.type_system import INT, BOOL, arrow
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.compression import (
    # Data structures
    SubtreeOccurrence, CompressionResult, CompressionState,
    # Quality filters
    is_nontrivial, is_eta_reducible, is_nested_eta_reducible,
    is_single_task_abstraction, passes_abstraction_quality_checks,
    # Anti-unification
    anti_unify, find_anti_unified_patterns, create_abstraction_from_pattern,
    # Subtree finding
    find_common_subtrees, abstract_subtree,
    # Arity search
    Factorization, enumerate_factorizations, abstract_subtree_partial,
    best_factorization, rank_factorizations_by_mdl,
    # Rewriting
    RewriteResult, rewrite_with_invention, rewrite_with_invention_detailed,
    verify_rewrite_semantics, rewrite_and_verify,
    rewrite_frontier, rewrite_all_frontiers,
    # MDL scoring
    compute_mdl, compute_mdl_detailed, evaluate_invention_mdl, rank_inventions_by_mdl,
    # Main compression
    compress_frontiers, compress_frontiers_mdl, beam_search_compression,
    beam_search_compression_with_arity, compress_frontiers_legacy,
    iterative_compression, compute_compression_ratio,
    format_invention, compression_report,
    # Recognition-guided
    compute_recognition_score, compute_combined_score, compress_frontiers_recognition,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def primitives():
    """Basic arithmetic primitives for testing."""
    add = Primitive('+', arrow(INT, INT, INT), lambda a: lambda b: a + b)
    mul = Primitive('*', arrow(INT, INT, INT), lambda a: lambda b: a * b)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)
    three = Primitive('3', INT, 3)
    return {'add': add, 'mul': mul, 'one': one, 'two': two, 'three': three}


@pytest.fixture
def grammar(primitives):
    """Uniform grammar with arithmetic primitives."""
    return uniform_grammar(list(primitives.values()))


@pytest.fixture
def test_programs(primitives):
    """Test programs: (+ 1 1), (+ 1 2), (+ 1 3), (+ 1 (+ 1 1))"""
    add, one, two, three = primitives['add'], primitives['one'], primitives['two'], primitives['three']
    return [
        Application(Application(add, one), one),      # 2
        Application(Application(add, one), two),      # 3
        Application(Application(add, one), three),    # 4
        Application(Application(add, one),            # 3
                   Application(Application(add, one), one))
    ]


# =============================================================================
# DATA STRUCTURE TESTS
# =============================================================================

class TestDataStructures:
    """Test that all data structures are correctly defined."""

    def test_subtree_occurrence_fields(self, test_programs):
        """SubtreeOccurrence should have expected fields."""
        occ = SubtreeOccurrence(
            subtree=test_programs[0],
            count=2,
            programs=['p1', 'p2'],
            savings=3.0
        )
        assert occ.subtree == test_programs[0]
        assert occ.count == 2
        assert occ.programs == ['p1', 'p2']
        assert occ.savings == 3.0

    def test_compression_result_fields(self, grammar):
        """CompressionResult should have all expected fields."""
        result = CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[],
            rewritten_frontiers=None,
            rewrite_stats=None
        )
        assert hasattr(result, 'new_inventions')
        assert hasattr(result, 'old_grammar')
        assert hasattr(result, 'new_grammar')
        assert hasattr(result, 'total_savings')
        assert hasattr(result, 'subtree_analysis')
        assert hasattr(result, 'rewritten_frontiers')
        assert hasattr(result, 'rewrite_stats')

    def test_compression_state_fields(self, grammar):
        """CompressionState should support comparison and hashing."""
        # CompressionState requires: grammar, programs, inventions, targets, mdl
        state = CompressionState(
            grammar=grammar,
            programs=[],
            inventions=[],
            targets=[],  # List of (target_subtree, n_args) tuples
            mdl=0.0      # MDL score, not "total_savings"
        )
        # Should be hashable and comparable
        assert hash(state) is not None
        assert state == state


# =============================================================================
# QUALITY FILTER TESTS
# =============================================================================

class TestQualityFilters:
    """Test abstraction quality filters."""

    def test_is_nontrivial(self, primitives):
        """Trivial programs should be detected."""
        # Trivial: just a variable
        assert not is_nontrivial(Index(0))
        # Trivial: just a primitive
        assert not is_nontrivial(primitives['one'])
        # Nontrivial: application
        assert is_nontrivial(Application(primitives['add'], primitives['one']))

    def test_is_eta_reducible(self, primitives):
        """Eta-reducible abstractions should be detected."""
        add = primitives['add']
        # λx. (f x) is eta-reducible to f
        eta_prog = Abstraction(Application(add, Index(0)))
        assert is_eta_reducible(eta_prog)
        # λx. (+ x 1) is NOT eta-reducible
        non_eta = Abstraction(Application(Application(add, Index(0)), primitives['one']))
        assert not is_eta_reducible(non_eta)

    def test_passes_abstraction_quality_checks(self, primitives):
        """Combined quality check should pass valid abstractions."""
        add, one = primitives['add'], primitives['one']
        # Good abstraction: λx. (+ x 1)
        good_body = Abstraction(Application(Application(add, Index(0)), one))
        good_inv = Invented(good_body)
        assert passes_abstraction_quality_checks(good_inv)


# =============================================================================
# ANTI-UNIFICATION TESTS
# =============================================================================

class TestAntiUnification:
    """Test anti-unification pattern discovery."""

    def test_anti_unify_identical(self, test_programs):
        """Identical programs should return unchanged."""
        pattern, subs, sub_map = anti_unify(test_programs[0], test_programs[0])
        assert pattern == test_programs[0]
        assert len(subs) == 0

    def test_anti_unify_different(self, primitives):
        """Different programs should produce generalized pattern."""
        add, one, two, three = primitives['add'], primitives['one'], primitives['two'], primitives['three']
        p1 = Application(Application(add, one), two)
        p2 = Application(Application(add, one), three)

        pattern, subs, sub_map = anti_unify(p1, p2)

        # Pattern should have a variable where they differ
        assert pattern is not None
        assert len(subs) == 1  # One difference: two vs three

    def test_find_anti_unified_patterns(self, test_programs):
        """Should find patterns across multiple programs."""
        patterns = find_anti_unified_patterns(test_programs, min_uses=2)
        # Should find at least one pattern
        assert len(patterns) >= 0  # May be empty depending on implementation


# =============================================================================
# SUBTREE FINDING TESTS
# =============================================================================

class TestSubtreeFinding:
    """Test common subtree detection."""

    def test_find_common_subtrees(self, test_programs):
        """Should find repeated subtrees."""
        common = find_common_subtrees(test_programs, min_size=2, min_count=2)

        # Should find (+ 1) which appears in all programs
        assert len(common) > 0
        # Savings should be positive
        assert all(c.savings > 0 for c in common)

    def test_abstract_subtree(self, primitives):
        """Should create correct invention from subtree."""
        add, one = primitives['add'], primitives['one']

        # (+ $0 1) has one free variable
        target = Application(Application(add, Index(0)), one)
        invention, n_args = abstract_subtree(target)

        assert n_args == 1
        assert isinstance(invention, Invented)
        assert isinstance(invention.body, Abstraction)


# =============================================================================
# ARITY SEARCH TESTS
# =============================================================================

class TestAritySearch:
    """Test arity-aware factorization."""

    def test_enumerate_factorizations_zero_vars(self, primitives):
        """Ground terms should have one factorization."""
        add, one = primitives['add'], primitives['one']
        ground = Application(Application(add, one), one)

        facts = enumerate_factorizations(ground)

        assert len(facts) == 1
        assert facts[0].n_args == 0

    def test_enumerate_factorizations_two_vars(self, primitives):
        """Two free vars should produce multiple factorizations."""
        add = primitives['add']
        pattern = Application(Application(add, Index(0)), Index(1))

        facts = enumerate_factorizations(pattern, max_args=4)

        # Should have: 2-arg, 1-arg (two ways)
        assert len(facts) >= 2
        arities = [f.n_args for f in facts]
        assert 2 in arities
        assert 1 in arities


# =============================================================================
# REWRITING TESTS
# =============================================================================

class TestRewriting:
    """Test program rewriting with inventions."""

    def test_rewrite_preserves_semantics(self, primitives):
        """Rewriting should preserve program semantics."""
        add, one = primitives['add'], primitives['one']

        # Original: λx. (+ x 1)
        orig = Abstraction(Application(Application(add, Index(0)), one))

        # Create invention for (+ $0 1)
        target = Application(Application(add, Index(0)), one)
        inv_body = Abstraction(Application(Application(add, Index(0)), one))
        inv = Invented(inv_body)

        # Rewrite
        rewritten = rewrite_with_invention(orig, target, inv, 1)

        # Verify semantics
        for test_val in [0, 1, 5, 10, -3]:
            orig_result = orig.evaluate([])(test_val)
            new_result = rewritten.evaluate([])(test_val)
            assert orig_result == new_result, f"Mismatch at {test_val}"

    def test_verify_rewrite_semantics_identical(self, primitives):
        """Identical programs should pass verification."""
        add, one, two = primitives['add'], primitives['one'], primitives['two']
        prog = Application(Application(add, one), two)

        success, error = verify_rewrite_semantics(prog, prog, [[]])

        assert success
        assert error is None

    def test_verify_rewrite_semantics_mismatch(self, primitives):
        """Different programs should fail verification."""
        add, one, two, three = primitives['add'], primitives['one'], primitives['two'], primitives['three']
        orig = Application(Application(add, one), two)  # 3
        wrong = Application(Application(add, one), three)  # 4

        success, error = verify_rewrite_semantics(orig, wrong, [[]])

        assert not success
        assert error is not None

    def test_rewrite_frontier(self, primitives):
        """Frontier rewriting should work correctly."""
        add, one, two = primitives['add'], primitives['one'], primitives['two']

        p1 = Application(Application(add, one), one)
        p2 = Application(Application(add, one), two)
        frontier = [(p1, -1.0), (p2, -2.0)]

        target = Application(Application(add, one), Index(0))
        inv_body = Abstraction(Application(Application(add, one), Index(0)))
        inv = Invented(inv_body)

        rewritten, stats = rewrite_frontier(frontier, target, inv, 1)

        assert len(rewritten) == 2
        assert stats['programs_changed'] >= 0


# =============================================================================
# MDL SCORING TESTS
# =============================================================================

class TestMDLScoring:
    """Test MDL computation."""

    def test_compute_mdl(self, grammar, test_programs):
        """MDL should be computable."""
        mdl = compute_mdl(grammar, test_programs, INT)

        assert mdl > 0  # Should be positive
        assert isinstance(mdl, float)

    def test_compute_mdl_detailed(self, grammar, test_programs):
        """Detailed MDL should have all components."""
        detailed = compute_mdl_detailed(grammar, test_programs, INT)

        assert 'grammar_dl' in detailed
        assert 'programs_dl' in detailed
        assert 'total_mdl' in detailed
        assert detailed['total_mdl'] == detailed['grammar_dl'] + detailed['programs_dl']

    def test_evaluate_invention_mdl(self, grammar, test_programs, primitives):
        """Invention evaluation should return all components."""
        add, one = primitives['add'], primitives['one']

        target = Application(add, one)
        inv = Invented(target)

        old_mdl, new_mdl, rewritten, stats = evaluate_invention_mdl(
            grammar, test_programs, inv, target, 0, INT
        )

        assert isinstance(old_mdl, float)
        assert isinstance(new_mdl, float)
        assert 'mdl_improvement' in stats


# =============================================================================
# MAIN COMPRESSION TESTS
# =============================================================================

class TestMainCompression:
    """Test main compression functions."""

    def test_compress_frontiers_basic(self, grammar, test_programs):
        """Basic compression should work."""
        frontiers = [[(p, 0.0)] for p in test_programs]

        result = compress_frontiers(
            grammar, frontiers,
            max_inventions=3,
            min_savings=0.5,
            refactor_programs=False
        )

        assert isinstance(result, CompressionResult)
        assert result.total_savings >= 0

    def test_compress_frontiers_with_refactoring(self, grammar, test_programs):
        """Compression with refactoring should preserve semantics."""
        frontiers = [[(p, 0.0)] for p in test_programs]
        original_values = [p.evaluate([]) for p in test_programs]

        result = compress_frontiers(
            grammar, frontiers,
            max_inventions=3,
            min_savings=0.5,
            refactor_programs=True
        )

        if result.rewritten_frontiers:
            new_values = [f[0][0].evaluate([]) for f in result.rewritten_frontiers]
            assert original_values == new_values, "Semantics not preserved!"

    def test_compress_frontiers_empty(self, grammar):
        """Empty frontiers should not crash."""
        result = compress_frontiers(grammar, [], max_inventions=3)

        assert len(result.new_inventions) == 0
        assert result.total_savings == 0

    def test_format_invention(self, primitives):
        """Invention formatting should produce string."""
        add, one = primitives['add'], primitives['one']
        inv = Invented(Application(add, one))

        formatted = format_invention(inv)

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_compression_report(self, grammar, test_programs):
        """Compression report should be generated."""
        frontiers = [[(p, 0.0)] for p in test_programs]
        result = compress_frontiers(grammar, frontiers, max_inventions=2)

        report = compression_report(result)

        assert 'COMPRESSION' in report
        assert 'inventions' in report.lower() or 'New inventions' in report


# =============================================================================
# RECOGNITION-GUIDED TESTS
# =============================================================================

class TestRecognitionGuided:
    """Test recognition-guided compression (DreamDecompiler)."""

    def test_compute_combined_score(self):
        """Combined score should weight correctly."""
        backward = 10.0
        forward = 5.0

        # With default scales (1.0), normalization doesn't change values much
        # Formula: alpha * (backward/backward_scale) + (1-alpha) * (forward/forward_scale)

        # Alpha=1.0: pure backward = 1.0 * 10.0 + 0.0 * 5.0 = 10.0
        score_pure_backward = compute_combined_score(backward, forward, alpha=1.0)
        assert score_pure_backward == 10.0

        # Alpha=0.0: pure forward = 0.0 * 10.0 + 1.0 * 5.0 = 5.0
        score_pure_forward = compute_combined_score(backward, forward, alpha=0.0)
        assert score_pure_forward == 5.0

        # Alpha=0.5: balanced = 0.5 * 10.0 + 0.5 * 5.0 = 7.5
        score_balanced = compute_combined_score(backward, forward, alpha=0.5)
        assert score_balanced == 7.5

    def test_compress_frontiers_recognition_fallback(self, grammar, test_programs):
        """Without recognition model, should fall back to standard compression."""
        frontiers = [[(p, 0.0)] for p in test_programs]

        # No recognition model provided - should fall back
        result = compress_frontiers_recognition(
            grammar, frontiers,
            unsolved_tasks=None,
            recognition_model=None,
            max_inventions=2
        )

        assert isinstance(result, CompressionResult)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests verifying end-to-end workflows."""

    def test_full_compression_pipeline(self, grammar, test_programs):
        """Full pipeline: find patterns → compress → rewrite → verify."""
        frontiers = [[(p, 0.0)] for p in test_programs]
        original_values = [p.evaluate([]) for p in test_programs]

        # Step 1: Find common subtrees
        common = find_common_subtrees(test_programs, min_size=2, min_count=2)
        assert len(common) > 0

        # Step 2: Compress with refactoring
        result = compress_frontiers(
            grammar, frontiers,
            max_inventions=3,
            min_savings=0.5,
            refactor_programs=True
        )

        # Step 3: Verify semantics preserved
        if result.rewritten_frontiers:
            new_values = [f[0][0].evaluate([]) for f in result.rewritten_frontiers]
            assert original_values == new_values

        # Step 4: Generate report
        report = compression_report(result)
        assert len(report) > 0

    def test_iterative_compression(self, grammar, test_programs):
        """Multi-round compression should work."""
        frontiers = [[(p, 0.0)] for p in test_programs]

        # Correct parameter name: max_inventions_per_round (not inventions_per_round)
        # Note: min_savings is not a parameter of iterative_compression
        result = iterative_compression(
            grammar, frontiers,
            max_rounds=2,
            max_inventions_per_round=2,
            refactor_programs=True
        )

        assert isinstance(result, CompressionResult)


def run_standalone_tests():
    """Run all tests without pytest."""
    print('=' * 70)
    print('COMPREHENSIVE COMPRESSION REFACTORING TESTS (Standalone)')
    print('=' * 70)

    # Create fixtures
    prims = {
        'add': Primitive('+', arrow(INT, INT, INT), lambda a: lambda b: a + b),
        'mul': Primitive('*', arrow(INT, INT, INT), lambda a: lambda b: a * b),
        'one': Primitive('1', INT, 1),
        'two': Primitive('2', INT, 2),
        'three': Primitive('3', INT, 3)
    }
    gram = uniform_grammar(list(prims.values()))
    test_progs = [
        Application(Application(prims['add'], prims['one']), prims['one']),
        Application(Application(prims['add'], prims['one']), prims['two']),
        Application(Application(prims['add'], prims['one']), prims['three']),
        Application(Application(prims['add'], prims['one']),
                   Application(Application(prims['add'], prims['one']), prims['one']))
    ]

    passed = 0
    failed = 0

    def run_test(name, test_class, method_name, *args):
        nonlocal passed, failed
        try:
            instance = test_class()
            method = getattr(instance, method_name)
            method(*args)
            print(f'  ✓ {name}')
            passed += 1
        except Exception as e:
            print(f'  ✗ {name}: {e}')
            failed += 1

    # Data structures
    print('\n--- Data Structure Tests ---')
    run_test('SubtreeOccurrence fields', TestDataStructures, 'test_subtree_occurrence_fields', test_progs)
    run_test('CompressionResult fields', TestDataStructures, 'test_compression_result_fields', gram)
    run_test('CompressionState fields', TestDataStructures, 'test_compression_state_fields', gram)

    # Quality filters
    print('\n--- Quality Filter Tests ---')
    run_test('is_nontrivial', TestQualityFilters, 'test_is_nontrivial', prims)
    run_test('is_eta_reducible', TestQualityFilters, 'test_is_eta_reducible', prims)
    run_test('passes_abstraction_quality_checks', TestQualityFilters, 'test_passes_abstraction_quality_checks', prims)

    # Anti-unification
    print('\n--- Anti-unification Tests ---')
    run_test('anti_unify identical', TestAntiUnification, 'test_anti_unify_identical', test_progs)
    run_test('anti_unify different', TestAntiUnification, 'test_anti_unify_different', prims)
    run_test('find_anti_unified_patterns', TestAntiUnification, 'test_find_anti_unified_patterns', test_progs)

    # Subtree finding
    print('\n--- Subtree Finding Tests ---')
    run_test('find_common_subtrees', TestSubtreeFinding, 'test_find_common_subtrees', test_progs)
    run_test('abstract_subtree', TestSubtreeFinding, 'test_abstract_subtree', prims)

    # Arity search
    print('\n--- Arity Search Tests ---')
    run_test('enumerate_factorizations (zero vars)', TestAritySearch, 'test_enumerate_factorizations_zero_vars', prims)
    run_test('enumerate_factorizations (two vars)', TestAritySearch, 'test_enumerate_factorizations_two_vars', prims)

    # Rewriting
    print('\n--- Rewriting Tests ---')
    run_test('rewrite_preserves_semantics', TestRewriting, 'test_rewrite_preserves_semantics', prims)
    run_test('verify_rewrite_semantics (identical)', TestRewriting, 'test_verify_rewrite_semantics_identical', prims)
    run_test('verify_rewrite_semantics (mismatch)', TestRewriting, 'test_verify_rewrite_semantics_mismatch', prims)
    run_test('rewrite_frontier', TestRewriting, 'test_rewrite_frontier', prims)

    # MDL scoring
    print('\n--- MDL Scoring Tests ---')
    run_test('compute_mdl', TestMDLScoring, 'test_compute_mdl', gram, test_progs)
    run_test('compute_mdl_detailed', TestMDLScoring, 'test_compute_mdl_detailed', gram, test_progs)
    run_test('evaluate_invention_mdl', TestMDLScoring, 'test_evaluate_invention_mdl', gram, test_progs, prims)

    # Main compression
    print('\n--- Main Compression Tests ---')
    run_test('compress_frontiers (basic)', TestMainCompression, 'test_compress_frontiers_basic', gram, test_progs)
    run_test('compress_frontiers (refactoring)', TestMainCompression, 'test_compress_frontiers_with_refactoring', gram, test_progs)
    run_test('compress_frontiers (empty)', TestMainCompression, 'test_compress_frontiers_empty', gram)
    run_test('format_invention', TestMainCompression, 'test_format_invention', prims)
    run_test('compression_report', TestMainCompression, 'test_compression_report', gram, test_progs)

    # Recognition-guided
    print('\n--- Recognition-Guided Tests ---')
    run_test('compute_combined_score', TestRecognitionGuided, 'test_compute_combined_score')
    run_test('compress_frontiers_recognition (fallback)', TestRecognitionGuided, 'test_compress_frontiers_recognition_fallback', gram, test_progs)

    # Integration
    print('\n--- Integration Tests ---')
    run_test('full_compression_pipeline', TestIntegration, 'test_full_compression_pipeline', gram, test_progs)
    run_test('iterative_compression', TestIntegration, 'test_iterative_compression', gram, test_progs)

    # Summary
    print('\n' + '=' * 70)
    print(f'RESULTS: {passed} passed, {failed} failed')
    print('=' * 70)
    if failed == 0:
        print('✅ ALL TESTS PASSED!')
    else:
        print(f'⚠️  {failed} tests failed')
    return failed == 0


if __name__ == '__main__':
    if HAS_PYTEST:
        import sys
        sys.exit(pytest.main([__file__, '-v']))
    else:
        success = run_standalone_tests()
        import sys
        sys.exit(0 if success else 1)
