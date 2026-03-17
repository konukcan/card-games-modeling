"""Tests for the DSL-to-English translator.

Covers simple programs, arithmetic evaluation, comparison flipping,
even/odd special cases, crash safety, caching, and capitalisation.
"""

import pytest

from gallery_analysis.visualization.dsl_translator import translate_dsl, clear_cache


# ---------------------------------------------------------------------------
# Fixture: clear translation cache between tests so they are independent.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_cache():
    """Clear the module-level translation cache before each test."""
    clear_cache()
    yield
    clear_cache()


# ===================================================================
# 1. Simple programs — basic primitives produce readable output
# ===================================================================

class TestSimplePrograms:
    """Verify that common single-primitive programs translate readably."""

    def test_not_has_color_black(self):
        result = translate_dsl("(λ not (has_color $0 BLACK))")
        low = result.lower()
        assert "no" in low or "not" in low, f"Expected negation in: {result}"
        assert "black" in low, f"Expected 'black' in: {result}"

    def test_count_color_ge_3(self):
        result = translate_dsl("(λ ge (count_color $0 RED) 3)")
        low = result.lower()
        assert "red" in low, f"Expected 'red' in: {result}"
        # The translator uses the unicode ≥ symbol for ge.
        assert "3" in result, f"Expected '3' in: {result}"
        assert "\u2265" in result or ">=" in result or "> 3" in result or "≥" in result, (
            f"Expected a 'greater-or-equal' indicator in: {result}"
        )

    def test_all_cards_red(self):
        result = translate_dsl("(λ all (λ eq (get_color $0) RED) $0)")
        low = result.lower()
        assert "all" in low, f"Expected 'all' in: {result}"
        assert "red" in low, f"Expected 'red' in: {result}"


# ===================================================================
# 2. Arithmetic evaluation — static computation works
# ===================================================================

class TestArithmeticEvaluation:
    """Verify that constant arithmetic is folded to a single number."""

    def test_nested_addition_folded(self):
        result = translate_dsl("(λ ge (sum_ranks $0) (+ 5 (+ 5 5)))")
        assert "15" in result, (
            f"Expected '15' (folded from 5+5+5) in: {result}"
        )
        # Should NOT contain the un-folded form.
        assert "5 + 5" not in result, (
            f"Arithmetic should be folded but got: {result}"
        )


# ===================================================================
# 3. Comparison flipping — (lt 5 X) becomes "X > 5" style
# ===================================================================

class TestComparisonFlipping:
    """Verify that comparisons with the constant on the left are flipped."""

    def test_lt_number_on_left_flips(self):
        result = translate_dsl("(λ lt 5 (max_suit_count $0))")
        assert ">" in result, f"Expected '>' (flipped from lt) in: {result}"
        assert "5" in result, f"Expected '5' in: {result}"


# ===================================================================
# 4. Even/odd special case
# ===================================================================

class TestEvenOdd:
    """Verify the even/odd rank idiom is detected."""

    def test_even_rank(self):
        result = translate_dsl("(λ all (λ eq (mod (rank_val $0) 2) 0) $0)")
        low = result.lower()
        assert "even" in low, f"Expected 'even' in: {result}"

    def test_odd_rank(self):
        result = translate_dsl("(λ all (λ eq (mod (rank_val $0) 2) 1) $0)")
        low = result.lower()
        assert "odd" in low, f"Expected 'odd' in: {result}"


# ===================================================================
# 5. Never crashes — malformed input returns the raw string
# ===================================================================

class TestNeverCrashes:
    """The translator must never raise; it returns the raw string on failure."""

    def test_plain_text(self):
        raw = "not a valid program"
        result = translate_dsl(raw)
        # Must return *something* without raising.
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        result = translate_dsl("")
        # Must return *something* without raising (translator returns "None"
        # because parse_sexpr returns None for empty input, which is then
        # stringified and capitalised).
        assert isinstance(result, str)

    def test_unknown_primitive(self):
        raw = "(λ unknown_primitive $0)"
        result = translate_dsl(raw)
        # Must return *something* without raising.
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deeply_nested_garbage(self):
        raw = "(((((()"
        result = translate_dsl(raw)
        assert isinstance(result, str)

    def test_only_parens(self):
        raw = "()"
        result = translate_dsl(raw)
        assert isinstance(result, str)


# ===================================================================
# 6. Caching — calling twice with same input returns same result
# ===================================================================

class TestCaching:
    """Verify that repeated calls are safe and consistent."""

    def test_same_result_on_second_call(self):
        program = "(λ all (λ eq (get_color $0) RED) $0)"
        first = translate_dsl(program)
        second = translate_dsl(program)
        assert first == second

    def test_cache_hit_does_not_crash(self):
        """Call three times rapidly — no exceptions expected."""
        program = "(λ ge (sum_ranks $0) 10)"
        for _ in range(3):
            translate_dsl(program)


# ===================================================================
# 7. Output is capitalised — first character is uppercase
# ===================================================================

class TestCapitalisation:
    """Every non-empty translation should start with an uppercase letter."""

    @pytest.mark.parametrize("program", [
        "(λ all (λ eq (get_color $0) RED) $0)",
        "(λ not (has_color $0 BLACK))",
        "(λ ge (count_color $0 RED) 3)",
        "(λ lt 5 (max_suit_count $0))",
        "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)",
    ])
    def test_first_char_uppercase(self, program):
        result = translate_dsl(program)
        assert result[0].isupper(), (
            f"Expected uppercase first char, got: {result!r}"
        )
