"""
Complete catalogue of 55 card game rules for the DreamCoder wake-sleep experiments.

Each rule is implemented as a predicate (Hand → bool) using the compositional primitives,
with CompositionNode trees describing how they are built from the DSL.
This module demonstrates the compositional structure and enables analysis of
shared subprograms across rules.

NOTE: This catalogue is used by dreamcoder_core/, experiments/, and analysis/ code.
It is NOT the rule set used by the MCMC/Bayesian analysis in gallery_analysis/.
For the MCMC gallery rules (61 rules ported from the behavioral experiment),
see gallery_analysis/gallery_rules.py.

COMPOSITIONAL NOTATION:
- We express each rule as a composition of primitives
- Format: rule = primitive1 ∘ primitive2 ∘ ... or rule = combinator(primitive)(args)
- This reveals shared structure for transfer learning analysis
"""

from typing import Callable, List, Set, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .cards import Hand, Suit, Rank, Color, AltColor1, AltColor2, Parity, RANK_VALUES
from .primitives import *


@dataclass
class CompositionNode:
    """
    Represents a node in the compositional structure of a rule.

    This allows us to express rules as trees of primitive compositions,
    enabling analysis of shared substructure.
    """
    primitive: str  # Name of the primitive or combinator
    args: List['CompositionNode'] = field(default_factory=list)  # Child nodes
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters (e.g., threshold=3)

    def __str__(self) -> str:
        """Generate human-readable composition string."""
        if not self.args and not self.params:
            return self.primitive

        param_str = ""
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())

        if self.args:
            args_str = ", ".join(str(a) for a in self.args)
            if param_str:
                return f"{self.primitive}({args_str}, {param_str})"
            return f"{self.primitive}({args_str})"

        return f"{self.primitive}({param_str})"

    def to_lambda_notation(self) -> str:
        """Generate lambda calculus notation."""
        return f"λh. {self._to_lambda_body()}"

    def _to_lambda_body(self) -> str:
        if not self.args:
            if self.params:
                return f"{self.primitive}({', '.join(f'{v}' for v in self.params.values())})"
            return self.primitive

        args_str = ", ".join(a._to_lambda_body() for a in self.args)
        return f"{self.primitive}({args_str})"


@dataclass
class Rule:
    """
    A rule with metadata, evaluation function, and compositional decomposition.

    Attributes:
        id: Unique identifier (e.g., "Sorted_by_rank")
        token: Short token for experiment (e.g., "r1x")
        name: Human-readable name
        predicate: The evaluation function (Hand → bool)
        family: Rule family classification
        description: Detailed explanation for participants
        composition: Compositional decomposition as tree
        primitives_used: Flat list of primitive names
        level: Compositional depth (max level of primitives used)
    """
    id: str
    token: str
    name: str
    predicate: Callable[[Hand], bool]
    family: str
    description: str
    composition: CompositionNode
    primitives_used: List[str]
    level: int = 0  # Max level of primitives (0-4)

    def eval(self, hand: Hand) -> bool:
        """Evaluate rule on a hand."""
        return self.predicate(hand)

    def composition_str(self) -> str:
        """Get string representation of composition."""
        return str(self.composition)

    def lambda_str(self) -> str:
        """Get lambda calculus notation."""
        return self.composition.to_lambda_notation()


# ============================================================================
# HELPER: Create composition nodes
# ============================================================================

def C(name: str, *args, **params) -> CompositionNode:
    """Shorthand for creating composition nodes."""
    return CompositionNode(
        primitive=name,
        args=list(args),
        params=params
    )


# ============================================================================
# RULE CATALOG - CORE 45 RULES
# Organized by family with full compositional decomposition
# Matches card-games/js/rules.js exactly
# ============================================================================

def create_all_rules() -> List[Rule]:
    """Create all 45 core experimental rules with full metadata and compositions."""
    rules = []

    # =========================================================================
    # FAMILY: LOCAL (Simple positional/ordering rules)
    # =========================================================================

    # r1x: Sorted by rank
    rules.append(Rule(
        id="Sorted_by_rank",
        token="r1x",
        name="Ranks are in non-decreasing order",
        predicate=lambda h: is_sorted(h, get_rank_val, strict=False),
        family="LOCAL",
        description="Read ranks left-to-right; they never go down (e.g., 5,7,7,9,J,…).",
        composition=C("is_sorted", C("map", C("get_rank_val")), strict=False),
        primitives_used=["is_sorted", "map", "get_rank_val"],
        level=1
    ))

    # r4x: S before H
    def spade_before_heart(hand: Hand) -> bool:
        seen_spade = False
        for card in hand:
            if get_suit(card) == Suit.SPADES:
                seen_spade = True
            if seen_spade and get_suit(card) == Suit.HEARTS:
                return True
        return False

    rules.append(Rule(
        id="S_before_H",
        token="r4x",
        name="Some ♠ appears before some ♥",
        predicate=spade_before_heart,
        family="LOCAL",
        description="Scan left-to-right; at least one ♠ is left of at least one ♥.",
        composition=C("exists_ordered", C("get_suit"), C("eq", suit1="SPADES"), C("eq", suit2="HEARTS")),
        primitives_used=["exists_ordered", "get_suit", "eq"],
        level=1
    ))

    # r44x: Ends same suit
    rules.append(Rule(
        id="Ends_same_suit",
        token="r44x",
        name="First and last share the suit",
        predicate=lambda h: len(h) >= 2 and get_suit(h[0]) == get_suit(h[-1]),
        family="LOCAL",
        description="The first and last cards have the same suit.",
        composition=C("terminals_equal", C("get_suit")),
        primitives_used=["terminals_equal", "get_suit", "first", "last", "eq"],
        level=4
    ))

    # r45x: Ends same color
    rules.append(Rule(
        id="Ends_same_color",
        token="r45x",
        name="First and last share the color",
        predicate=lambda h: len(h) >= 2 and get_color(h[0]) == get_color(h[-1]),
        family="LOCAL",
        description="The first and last cards are both red or both black.",
        composition=C("terminals_equal", C("get_color")),
        primitives_used=["terminals_equal", "get_color", "first", "last", "eq"],
        level=4
    ))

    # =========================================================================
    # FAMILY: COUNT (Counting and set cardinality rules)
    # =========================================================================

    # r2x: Has pair ranks
    def has_pair(hand: Hand) -> bool:
        ranks_seen = set()
        for card in hand:
            r = get_rank(card)
            if r in ranks_seen:
                return True
            ranks_seen.add(r)
        return False

    rules.append(Rule(
        id="Has_pair_ranks",
        token="r2x",
        name="At least one pair (same rank)",
        predicate=has_pair,
        family="COUNT",
        description="Some two cards share the same rank, e.g., two 9s.",
        composition=C("lt", C("unique_count", C("get_rank")), C("length")),
        primitives_used=["unique_count", "get_rank", "length", "lt"],
        level=1
    ))

    # r3x: Uniform color
    rules.append(Rule(
        id="Uniform_color",
        token="r3x",
        name="All cards have the same color",
        predicate=uniform_property(get_color),
        family="COUNT",
        description="All cards are either black (♣/♠) or red (♦/♥).",
        composition=C("uniform", C("get_color")),
        primitives_used=["uniform", "get_color", "unique_count", "eq"],
        level=1
    ))

    # p7x: Exactly two suits
    rules.append(Rule(
        id="Exactly_two_suits",
        token="p7x",
        name="Exactly two suits appear",
        predicate=lambda h: unique_count(get_suit)(h) == 2,
        family="COUNT",
        description="Across the whole hand there are exactly two distinct suits.",
        composition=C("eq", C("unique_count", C("get_suit")), value=2),
        primitives_used=["unique_count", "get_suit", "eq"],
        level=1
    ))

    # r11x: Half or more same suit
    def half_or_more_same_suit(hand: Hand) -> bool:
        n = len(hand)
        threshold = (n + 1) // 2
        for suit in Suit:
            if count_equal(suit, get_suit)(hand) >= threshold:
                return True
        return False

    rules.append(Rule(
        id="Half_or_more_same_suit",
        token="r11x",
        name="At least half the cards share a suit",
        predicate=half_or_more_same_suit,
        family="COUNT",
        description="≥ N/2 cards (rounded up) are the same suit.",
        composition=C("gte", C("max_count", C("get_suit")), C("div", C("length"), divisor=2)),
        primitives_used=["max_count", "get_suit", "length", "div", "gte"],
        level=1
    ))

    # r55x: At most three suits
    rules.append(Rule(
        id="At_most_three_suits",
        token="r55x",
        name="At most three suits appear",
        predicate=lambda h: unique_count(get_suit)(h) <= 3,
        family="COUNT",
        description="Across the hand there are no more than three distinct suits.",
        composition=C("lte", C("unique_count", C("get_suit")), value=3),
        primitives_used=["unique_count", "get_suit", "lte"],
        level=1
    ))

    # r43x: Exactly one club
    rules.append(Rule(
        id="Exactly_one_club",
        token="r43x",
        name="Exactly one club (♣)",
        predicate=lambda h: count_equal(Suit.CLUBS, get_suit)(h) == 1,
        family="COUNT",
        description="The hand contains exactly one ♣.",
        composition=C("eq", C("count_equal", C("get_suit"), value="CLUBS"), value=1),
        primitives_used=["count_equal", "get_suit", "eq"],
        level=1
    ))

    # =========================================================================
    # FAMILY: POSITION
    # NOTE: r7x (Pos3_is_JQK), r8x (Pos4_is_2_5_7) - ARCHIVED (non-contiguous rank sets)
    # NOTE: r9x (Has_Ace_of_Spades), r10x (Has_6_of_Diamonds) - ARCHIVED (rank constants)
    # =========================================================================

    # =========================================================================
    # FAMILY: AP (Arithmetic Progression rules)
    # =========================================================================

    # r5x: AP len 3 anywhere any step
    rules.append(Rule(
        id="AP_len3_anywhere_anyk",
        token="r5x",
        name="There is a 3-term rank pattern with equal steps (any step)",
        predicate=has_arithmetic_progression(3, None, False),
        family="AP",
        description="Ignoring order, some three ranks differ by the same amount (e.g., 4,6,8 or 7,10,13).",
        composition=C("has_AP", length=3, step="any", aligned=False),
        primitives_used=["has_AP"],
        level=3
    ))

    # r51x: AP len 3 step 2 anywhere
    rules.append(Rule(
        id="AP_len3_step2_anywhere",
        token="r51x",
        name="3-term rank pattern with step 2",
        predicate=has_arithmetic_progression(3, 2, False),
        family="AP",
        description="Ignoring order, some three ranks form a step-2 progression (e.g., 5,7,9).",
        composition=C("has_AP", length=3, step=2, aligned=False),
        primitives_used=["has_AP"],
        level=3
    ))

    # r52x: AP len 4 step 2 anywhere
    rules.append(Rule(
        id="AP_len4_step2_anywhere",
        token="r52x",
        name="4-term rank pattern with step 2",
        predicate=has_arithmetic_progression(4, 2, False),
        family="AP",
        description="Ignoring order, some four ranks form a step-2 progression (e.g., 4,6,8,10).",
        composition=C("has_AP", length=4, step=2, aligned=False),
        primitives_used=["has_AP"],
        level=3
    ))

    # =========================================================================
    # FAMILY: SCORE (Scoring formulas)
    # NOTE: r6x (Score_threshold_Rstar) - ARCHIVED (complex compound scoring)
    # =========================================================================

    # r42x: Half sum diff >= N
    def half_sum_diff_geN(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_sum = sum(get_rank_val(hand[i]) for i in range(k))
        right_sum = sum(get_rank_val(hand[i]) for i in range(k, 2*k))
        return (left_sum - right_sum) >= n

    rules.append(Rule(
        id="Half_sum_diff_geN",
        token="r42x",
        name="Left half beats right by at least N points",
        predicate=half_sum_diff_geN,
        family="SCORE",
        description="Sum ranks left minus right is ≥ the hand size (N).",
        composition=C("gte",
            C("sub", C("sum", C("left_half")), C("sum", C("right_half"))),
            C("length")
        ),
        primitives_used=["halves", "sum", "map", "get_rank_val", "sub", "gte", "length"],
        level=2
    ))

    # r56x: Half sum one side >= 2x other
    def half_sum_one_side_ge_2x(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_sum = sum(get_rank_val(hand[i]) for i in range(k))
        right_sum = sum(get_rank_val(hand[i]) for i in range(k, 2*k))
        return (left_sum >= 2 * right_sum) or (right_sum >= 2 * left_sum)

    rules.append(Rule(
        id="Half_sum_one_side_ge_2x_other",
        token="r56x",
        name="One half ≥ 2× the other (sum of ranks)",
        predicate=half_sum_one_side_ge_2x,
        family="SCORE",
        description="Split the hand into left and right halves; the sum of ranks in one half is at least double the other.",
        composition=C("or",
            C("gte", C("sum", C("left_half")), C("mul", C("sum", C("right_half")), factor=2)),
            C("gte", C("sum", C("right_half")), C("mul", C("sum", C("left_half")), factor=2))
        ),
        primitives_used=["halves", "sum", "map", "get_rank_val", "mul", "gte", "or"],
        level=2
    ))

    # =========================================================================
    # FAMILY: HIER (Hierarchical: both halves share boolean property)
    # =========================================================================

    # r12x: Halves uniform color equal
    def halves_uniform_color_equal(hand: Hand) -> bool:
        left, right = halves(hand)
        left_uniform = uniform_property(get_color)(left)
        right_uniform = uniform_property(get_color)(right)
        return left_uniform == right_uniform

    rules.append(Rule(
        id="Halves_uniform_color_equal",
        token="r12x",
        name="Both halves are uniform in color (or both not)",
        predicate=halves_uniform_color_equal,
        family="HIER",
        description="Left and right halves either both have a single color, or both mix colors.",
        composition=C("eq",
            C("uniform", C("get_color"), C("left_half")),
            C("uniform", C("get_color"), C("right_half"))
        ),
        primitives_used=["halves", "uniform", "get_color", "eq"],
        level=2
    ))

    # NOTE: r13x (Halves_uniform_parity_equal) - ARCHIVED (double nesting too complex)

    # r14x: Halves AP step 1 equal
    def halves_ap_step1_equal(hand: Hand) -> bool:
        def is_run(h: Hand) -> bool:
            if len(h) < 2:
                return False
            vals = [get_rank_val(c) for c in h]
            for i in range(len(vals) - 1):
                if vals[i+1] - vals[i] != 1:
                    return False
            return True
        left, right = halves(hand)
        return is_run(left) == is_run(right)

    rules.append(Rule(
        id="Halves_AP_step1_equal",
        token="r14x",
        name="Both halves are runs (+1) or both aren't",
        predicate=halves_ap_step1_equal,
        family="HIER",
        description="In each half, ranks strictly increase by 1 — either in both halves or in neither.",
        composition=C("eq",
            C("is_run", C("left_half")),
            C("is_run", C("right_half"))
        ),
        primitives_used=["halves", "is_run", "eq"],
        level=2
    ))

    # r26x: Halves hearts presence equal
    def halves_hearts_presence_equal(hand: Hand) -> bool:
        left, right = halves(hand)
        has_heart_left = any(get_suit(c) == Suit.HEARTS for c in left)
        has_heart_right = any(get_suit(c) == Suit.HEARTS for c in right)
        return has_heart_left == has_heart_right

    rules.append(Rule(
        id="Halves_hearts_presence_equal",
        token="r26x",
        name="Both halves either have a ♥ or neither does",
        predicate=halves_hearts_presence_equal,
        family="HIER",
        description="Look at each half separately; either both contain at least one heart, or neither does.",
        composition=C("eq",
            C("any", C("eq", C("get_suit"), value="HEARTS"), C("left_half")),
            C("any", C("eq", C("get_suit"), value="HEARTS"), C("right_half"))
        ),
        primitives_used=["halves", "any", "get_suit", "eq"],
        level=2
    ))

    # r53x: Halves AP len3 any equal
    def halves_ap_len3_any_equal(hand: Hand) -> bool:
        left, right = halves(hand)
        left_has_ap = has_arithmetic_progression(3, None, False)(left)
        right_has_ap = has_arithmetic_progression(3, None, False)(right)
        return left_has_ap == right_has_ap

    rules.append(Rule(
        id="Halves_AP_len3_any_equal",
        token="r53x",
        name="Both halves have a 3-term rank pattern (or both don't)",
        predicate=halves_ap_len3_any_equal,
        family="HIER",
        description="Within each half, ignoring order, some three ranks differ by the same amount — either in both halves or in neither.",
        composition=C("eq",
            C("has_AP", C("left_half"), params={"length": 3, "step": "any", "aligned": False}),
            C("has_AP", C("right_half"), params={"length": 3, "step": "any", "aligned": False})
        ),
        primitives_used=["halves", "has_AP", "eq"],
        level=3
    ))

    # r54x: Halves AP len2 step1 equal
    def halves_ap_len2_step1_equal(hand: Hand) -> bool:
        def has_pair_step1(h: Hand) -> bool:
            vals = set(get_rank_val(c) for c in h)
            for v in vals:
                if v+1 in vals or v-1 in vals:
                    return True
            return False
        left, right = halves(hand)
        return has_pair_step1(left) == has_pair_step1(right)

    rules.append(Rule(
        id="Halves_AP_len2_step1_equal",
        token="r54x",
        name="Both halves have some ±1 rank pair (or both don't)",
        predicate=halves_ap_len2_step1_equal,
        family="HIER",
        description="Within each half there exists a pair of ranks that differ by exactly 1 — either in both halves or in neither.",
        composition=C("eq",
            C("has_adjacent_pair", C("left_half")),
            C("has_adjacent_pair", C("right_half"))
        ),
        primitives_used=["halves", "has_adjacent_pair", "eq"],
        level=2
    ))

    # =========================================================================
    # FAMILY: LANG (Language/Grammar rules - bracket matching)
    # =========================================================================

    # r15x: Well-formed brackets by suit
    rules.append(Rule(
        id="Well_formed_brackets_by_suit",
        token="r15x",
        name="Suits form matched brackets",
        predicate=bracket_match_suits,
        family="LANG",
        description="Treat ♠/♥ as openers and ♣/♦ as closers of two bracket types; properly nested.",
        composition=C("bracket_match",
            openers={"SPADES": "(", "HEARTS": "["},
            closers={"CLUBS": ")", "DIAMONDS": "]"}
        ),
        primitives_used=["bracket_match"],
        level=3
    ))

    # r16x: Even opens next closes
    rules.append(Rule(
        id="Even_opens_next_closes",
        token="r16x",
        name="Even opens, next odd closes",
        predicate=lambda h: bracket_match_ranks_even_odd(h, even_opens=True),
        family="LANG",
        description="Even ranks push; only the next odd closes (e.g., 6 opens; 7 closes).",
        composition=C("bracket_match_parity", even_opens=True),
        primitives_used=["bracket_match_parity", "get_rank_val", "parity"],
        level=3
    ))

    # r17x: Odd opens next closes
    rules.append(Rule(
        id="Odd_opens_next_closes",
        token="r17x",
        name="Odd opens, next even closes",
        predicate=lambda h: bracket_match_ranks_even_odd(h, even_opens=False),
        family="LANG",
        description="Odd ranks push; only the next even closes (e.g., 5 opens; 6 closes).",
        composition=C("bracket_match_parity", even_opens=False),
        primitives_used=["bracket_match_parity", "get_rank_val", "parity"],
        level=3
    ))

    # =========================================================================
    # FAMILY: PAL (Palindrome rules)
    # =========================================================================

    # r18x: Suits palindrome
    rules.append(Rule(
        id="Suits_palindrome",
        token="r18x",
        name="Suits read the same forward/back",
        predicate=seq_palindrome(get_suit),
        family="PAL",
        description="The sequence of suits is palindromic.",
        composition=C("seq_palindrome", C("get_suit")),
        primitives_used=["seq_palindrome", "map", "get_suit", "reverse", "eq"],
        level=4
    ))

    # r19x: Colors palindrome
    rules.append(Rule(
        id="Colors_palindrome",
        token="r19x",
        name="Colors read the same forward/back",
        predicate=seq_palindrome(get_color),
        family="PAL",
        description="The sequence of RED/BLACK repeats symmetrically.",
        composition=C("seq_palindrome", C("get_color")),
        primitives_used=["seq_palindrome", "map", "get_color", "reverse", "eq"],
        level=4
    ))

    # r20x: Ranks palindrome
    rules.append(Rule(
        id="Ranks_palindrome",
        token="r20x",
        name="Ranks read the same forward/back",
        predicate=seq_palindrome(get_rank),
        family="PAL",
        description="The sequence of ranks is palindromic.",
        composition=C("seq_palindrome", C("get_rank")),
        primitives_used=["seq_palindrome", "map", "get_rank", "reverse", "eq"],
        level=4
    ))

    # =========================================================================
    # FAMILY: ALTCLR (Alternative color groupings)
    # =========================================================================

    # r27x: AltColor1 palindrome
    rules.append(Rule(
        id="AltColor1_palindrome",
        token="r27x",
        name="Pointy/Round pattern is a palindrome",
        predicate=seq_palindrome(get_altcolor1),
        family="ALTCLR",
        description="Group suits into Pointy (♠♦) and Round (♥♣). That two-group pattern reads the same forward/back.",
        composition=C("seq_palindrome", C("get_altcolor1")),
        primitives_used=["seq_palindrome", "map", "get_altcolor1", "reverse", "eq"],
        level=4
    ))

    # r28x: AltColor2 palindrome
    rules.append(Rule(
        id="AltColor2_palindrome",
        token="r28x",
        name="SH/DC pattern is a palindrome",
        predicate=seq_palindrome(get_altcolor2),
        family="ALTCLR",
        description="Group suits into SH (♠♥) and DC (♦♣). That two-group pattern reads the same forward/back.",
        composition=C("seq_palindrome", C("get_altcolor2")),
        primitives_used=["seq_palindrome", "map", "get_altcolor2", "reverse", "eq"],
        level=4
    ))

    # r31x: Ends same altcolor1
    rules.append(Rule(
        id="Ends_same_altcolor1",
        token="r31x",
        name="First and last are both Pointy or both Round",
        predicate=lambda h: len(h) >= 2 and get_altcolor1(h[0]) == get_altcolor1(h[-1]),
        family="ALTCLR",
        description="Map suits to Pointy (♠♦) vs Round (♥♣); first and last belong to the same group.",
        composition=C("terminals_equal", C("get_altcolor1")),
        primitives_used=["terminals_equal", "get_altcolor1", "first", "last", "eq"],
        level=4
    ))

    # =========================================================================
    # FAMILY: COPY (Halves copy sequence)
    # =========================================================================

    # r21x: Halves copy suits
    def halves_copy_suits(hand: Hand) -> bool:
        left, right = halves(hand)
        return [get_suit(c) for c in left] == [get_suit(c) for c in right]

    rules.append(Rule(
        id="Halves_copy_suits",
        token="r21x",
        name="Halves have same suit sequence",
        predicate=halves_copy_suits,
        family="COPY",
        description="Split hand in half; left suits match right suits in order.",
        composition=C("halves_equal", C("map", C("get_suit"))),
        primitives_used=["halves_equal", "halves", "map", "get_suit", "eq"],
        level=4
    ))

    # r22x: Halves copy colors
    def halves_copy_colors(hand: Hand) -> bool:
        left, right = halves(hand)
        return [get_color(c) for c in left] == [get_color(c) for c in right]

    rules.append(Rule(
        id="Halves_copy_colors",
        token="r22x",
        name="Halves have same color sequence",
        predicate=halves_copy_colors,
        family="COPY",
        description="Split hand in half; left colors match right colors in order.",
        composition=C("halves_equal", C("map", C("get_color"))),
        primitives_used=["halves_equal", "halves", "map", "get_color", "eq"],
        level=4
    ))

    # r23x: Halves copy ranks
    def halves_copy_ranks(hand: Hand) -> bool:
        left, right = halves(hand)
        return [get_rank(c) for c in left] == [get_rank(c) for c in right]

    rules.append(Rule(
        id="Halves_copy_ranks",
        token="r23x",
        name="Halves have same rank sequence",
        predicate=halves_copy_ranks,
        family="COPY",
        description="Split hand in half; left ranks match right ranks in order.",
        composition=C("halves_equal", C("map", C("get_rank"))),
        primitives_used=["halves_equal", "halves", "map", "get_rank", "eq"],
        level=4
    ))

    # r29x: Halves copy altcolor1
    def halves_copy_altcolor1(hand: Hand) -> bool:
        left, right = halves(hand)
        return [get_altcolor1(c) for c in left] == [get_altcolor1(c) for c in right]

    rules.append(Rule(
        id="Halves_copy_altcolor1",
        token="r29x",
        name="Right half copies left (Pointy/Round)",
        predicate=halves_copy_altcolor1,
        family="COPY",
        description="Map suits to Pointy (♠♦) vs Round (♥♣); the right half matches the left half exactly.",
        composition=C("halves_equal", C("map", C("get_altcolor1"))),
        primitives_used=["halves_equal", "halves", "map", "get_altcolor1", "eq"],
        level=4
    ))

    # r30x: Halves copy altcolor2
    def halves_copy_altcolor2(hand: Hand) -> bool:
        left, right = halves(hand)
        return [get_altcolor2(c) for c in left] == [get_altcolor2(c) for c in right]

    rules.append(Rule(
        id="Halves_copy_altcolor2",
        token="r30x",
        name="Right half copies left (SH/DC)",
        predicate=halves_copy_altcolor2,
        family="COPY",
        description="Map suits to SH (♠♥) vs DC (♦♣); the right half matches the left half exactly.",
        composition=C("halves_equal", C("map", C("get_altcolor2"))),
        primitives_used=["halves_equal", "halves", "map", "get_altcolor2", "eq"],
        level=4
    ))

    # r46x: Halves same suit set
    def halves_same_suit_set(hand: Hand) -> bool:
        left, right = halves(hand)
        return set(get_suit(c) for c in left) == set(get_suit(c) for c in right)

    rules.append(Rule(
        id="Halves_same_suit_set",
        token="r46x",
        name="Both halves contain the same suits (set)",
        predicate=halves_same_suit_set,
        family="COPY",
        description="Ignoring how many of each suit, both halves have the same set of suits.",
        composition=C("eq", C("unique", C("map", C("get_suit")), C("left_half")), C("unique", C("map", C("get_suit")), C("right_half"))),
        primitives_used=["halves", "unique", "map", "get_suit", "eq"],
        level=2
    ))

    # =========================================================================
    # FAMILY: SHIFT (Positional rank differences)
    # =========================================================================

    # r24x: Shift half plus two
    def shift_half_plus_two(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        pairs = shifted_pairs(k)(hand)
        for c1, c2 in pairs:
            if get_rank_val(c2) != get_rank_val(c1) + 2:
                return False
        return len(pairs) > 0

    rules.append(Rule(
        id="Shift_half_plus_two",
        token="r24x",
        name="Half-shift positions differ by +2 rank",
        predicate=shift_half_plus_two,
        family="SHIFT",
        description="Position i and position i+floor(n/2) differ by exactly 2 ranks.",
        composition=C("all", C("shifted_pairs", k="n/2"), C("eq", C("diff", C("get_rank_val")), value=2)),
        primitives_used=["shifted_pairs", "all", "get_rank_val", "diff", "eq"],
        level=2
    ))

    # NOTE: r25x (Shift2_plus3) - ARCHIVED (overlapping constraints too complex)

    # r41x: Shift half ge (right >= left)
    def shift_half_ge(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        for i in range(k):
            if get_rank_val(hand[i + k]) < get_rank_val(hand[i]):
                return False
        return True

    rules.append(Rule(
        id="Shift_half_ge",
        token="r41x",
        name="Right half ≥ left half (ranks)",
        predicate=shift_half_ge,
        family="SHIFT",
        description="For each position i, the right-half rank is at least the left-half rank.",
        composition=C("all", C("shifted_pairs", k="n/2"), C("gte", C("second"), C("first"))),
        primitives_used=["shifted_pairs", "all", "get_rank_val", "gte"],
        level=2
    ))

    # =========================================================================
    # FAMILY: MAP - ARCHIVED
    # NOTE: r32x-r37x (suit cycle rules) - ARCHIVED (arbitrary suit cycles)
    # =========================================================================

    # =========================================================================
    # FAMILY: ADJ (Adjacent/skip constraints)
    # =========================================================================

    # r38x: Adjacent same rank or suit
    def adj_same_rank_or_suit(hand: Hand) -> bool:
        for i in range(len(hand) - 1):
            if not (get_rank(hand[i]) == get_rank(hand[i+1]) or get_suit(hand[i]) == get_suit(hand[i+1])):
                return False
        return True

    rules.append(Rule(
        id="Adj_same_rank_or_suit",
        token="r38x",
        name="Neighbors share rank or suit",
        predicate=adj_same_rank_or_suit,
        family="ADJ",
        description="For every adjacent pair, either ranks match or suits match.",
        composition=C("all", C("adjacent_pairs"), C("or", C("eq", C("get_rank")), C("eq", C("get_suit")))),
        primitives_used=["adjacent_pairs", "all", "get_rank", "get_suit", "eq", "or"],
        level=2
    ))

    # r39x: Skip2 same rank or suit
    def skip2_same_rank_or_suit(hand: Hand) -> bool:
        for i in range(len(hand) - 2):
            if not (get_rank(hand[i]) == get_rank(hand[i+2]) or get_suit(hand[i]) == get_suit(hand[i+2])):
                return False
        return True

    rules.append(Rule(
        id="Skip2_same_rank_or_suit",
        token="r39x",
        name="Every 3rd card matches rank or suit",
        predicate=skip2_same_rank_or_suit,
        family="ADJ",
        description="For each i, cards i and i+2 share rank or suit.",
        composition=C("all", C("shifted_pairs", k=2), C("or", C("eq", C("get_rank")), C("eq", C("get_suit")))),
        primitives_used=["shifted_pairs", "all", "get_rank", "get_suit", "eq", "or"],
        level=2
    ))

    # r40x: Adjacent rank gap <= 3
    def adj_rank_gap_le3(hand: Hand) -> bool:
        for i in range(len(hand) - 1):
            if abs(get_rank_val(hand[i]) - get_rank_val(hand[i+1])) > 3:
                return False
        return True

    rules.append(Rule(
        id="Adj_rank_gap_le3",
        token="r40x",
        name="Neighbors differ by ≤3 ranks",
        predicate=adj_rank_gap_le3,
        family="ADJ",
        description="For each adjacent pair, rank distance is at most 3.",
        composition=C("all", C("adjacent_pairs"), C("lte", C("abs", C("diff", C("get_rank_val"))), value=3)),
        primitives_used=["adjacent_pairs", "all", "get_rank_val", "diff", "abs", "lte"],
        level=2
    ))

    # r57x: Adjacent same rank or color
    def adj_same_rank_or_color(hand: Hand) -> bool:
        for i in range(len(hand) - 1):
            if not (get_rank(hand[i]) == get_rank(hand[i+1]) or get_color(hand[i]) == get_color(hand[i+1])):
                return False
        return True

    rules.append(Rule(
        id="Adj_same_rank_or_color",
        token="r57x",
        name="Neighbors share rank or color",
        predicate=adj_same_rank_or_color,
        family="ADJ",
        description="For every adjacent pair, either ranks match or colors match.",
        composition=C("all", C("adjacent_pairs"), C("or", C("eq", C("get_rank")), C("eq", C("get_color")))),
        primitives_used=["adjacent_pairs", "all", "get_rank", "get_color", "eq", "or"],
        level=2
    ))

    # =========================================================================
    # FAMILY: PARITY (Odd/even rules)
    # =========================================================================

    # r47x: Only one odd rank
    def only_one_odd_rank(hand: Hand) -> bool:
        odd_count = sum(1 for c in hand if get_rank_val(c) % 2 == 1)
        return odd_count == 1

    rules.append(Rule(
        id="Only_one_odd_rank",
        token="r47x",
        name="Exactly one odd rank",
        predicate=only_one_odd_rank,
        family="PARITY",
        description="Among all cards there is exactly one whose rank is odd.",
        composition=C("eq", C("count", C("filter", C("is_odd", C("get_rank_val")))), value=1),
        primitives_used=["count", "filter", "get_rank_val", "is_odd", "eq"],
        level=1
    ))

    # r48x: Uniform rank parity
    rules.append(Rule(
        id="Uniform_rank_parity",
        token="r48x",
        name="All ranks are same parity",
        predicate=uniform_property(get_parity),
        family="PARITY",
        description="Every rank is odd, or every rank is even.",
        composition=C("uniform", C("get_parity")),
        primitives_used=["uniform", "get_parity"],
        level=1
    ))

    # =========================================================================
    # FAMILY: CENTER (Distance from center rules)
    # =========================================================================

    # r49x: Halves radial nonincreasing
    def halves_radial_nonincreasing(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left = hand[:k]
        right = hand[n-k:]
        # Left: from center outward (k-1, k-2, ..., 0) non-increasing
        for i in range(k-1, 0, -1):
            if get_rank_val(left[i]) < get_rank_val(left[i-1]):
                return False
        # Right: from center outward (0,1,...,k-1) non-increasing when read outward
        for i in range(k-1):
            if get_rank_val(right[i]) < get_rank_val(right[i+1]):
                return False
        return True

    rules.append(Rule(
        id="Halves_radial_nonincreasing",
        token="r49x",
        name="Outward from center, ranks don't go up (per half)",
        predicate=halves_radial_nonincreasing,
        family="CENTER",
        description="In each half, as you move away from the center, ranks never increase.",
        composition=C("and",
            C("sorted_outward", C("left_half"), direction="decreasing"),
            C("sorted_outward", C("right_half"), direction="decreasing")
        ),
        primitives_used=["halves", "sorted_outward", "get_rank_val", "and"],
        level=2
    ))

    # r50x: Global radial no dominance
    def global_radial_no_dominance(hand: Hand) -> bool:
        n = len(hand)
        center_left = n // 2 - 1
        center_right = n // 2
        def dist(i):
            return (center_left - i) if i <= center_left else (i - center_right)
        for i in range(n):
            for j in range(n):
                if dist(j) > dist(i) and get_rank_val(hand[j]) > get_rank_val(hand[i]):
                    return False
        return True

    rules.append(Rule(
        id="Global_radial_no_dominance",
        token="r50x",
        name="No farther-from-center card outranks a nearer one",
        predicate=global_radial_no_dominance,
        family="CENTER",
        description="Consider distance to the center; no card farther away may have higher rank than a nearer card.",
        composition=C("forall_pairs", C("implies", C("gt", C("distance"), C("distance")), C("lte", C("get_rank_val"), C("get_rank_val")))),
        primitives_used=["forall_pairs", "distance_from_center", "get_rank_val", "gt", "lte", "implies"],
        level=2
    ))

    # =========================================================================
    # 8-FAMILY 16-RULE TRANSFER DESIGN (from experimental-methodology.tex)
    # These rules were added in JS rules.js but not yet in Python catalogue.
    # =========================================================================

    # r58x: Halves face card equal — both halves have J/Q/K or neither does
    def halves_face_card_equal(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_half = hand[:k]
        right_half = hand[n - k:]
        has_face_left = any_fn(lambda c: get_rank(c) in (Rank.JACK, Rank.QUEEN, Rank.KING), left_half)
        has_face_right = any_fn(lambda c: get_rank(c) in (Rank.JACK, Rank.QUEEN, Rank.KING), right_half)
        return has_face_left == has_face_right

    rules.append(Rule(
        id="Halves_face_card_equal",
        token="r58x",
        name="Both halves have a face card, or neither does",
        predicate=halves_face_card_equal,
        family="HIER",
        description="Either both halves contain at least one face card (J/Q/K), or neither half has any.",
        composition=C("eq", C("any", C("is_face"), C("left_half")), C("any", C("is_face"), C("right_half"))),
        primitives_used=["halves", "any", "is_face", "eq"],
        level=2
    ))

    # r60x: AP len3 step1 anywhere — 3 consecutive ranks anywhere
    def ap_len3_step1_anywhere(hand: Hand) -> bool:
        vals = set(get_rank_val(c) for c in hand)
        for v in vals:
            if (v + 1) in vals and (v + 2) in vals:
                return True
        return False

    rules.append(Rule(
        id="AP_len3_step1_anywhere",
        token="r60x",
        name="Three consecutive ranks (anywhere)",
        predicate=ap_len3_step1_anywhere,
        family="AP",
        description="Somewhere in the hand, three cards have consecutive ranks like 5, 6, 7 or J, Q, K.",
        composition=C("has_AP", length=3, step=1, aligned=False),
        primitives_used=["has_AP", "get_rank_val"],
        level=3
    ))

    # r61x: All but one same color
    def all_but_one_same_color(hand: Hand) -> bool:
        red = sum(1 for c in hand if get_color(c) == Color.RED)
        black = len(hand) - red
        return min(red, black) <= 1

    rules.append(Rule(
        id="All_but_one_same_color",
        token="r61x",
        name="All but one same color",
        predicate=all_but_one_same_color,
        family="GLOBAL",
        description="All cards except at most one are the same color (all red or all black, with at most one exception).",
        composition=C("le", C("min", C("count_color", color="RED"), C("count_color", color="BLACK")), value=1),
        primitives_used=["count_color", "min", "le"],
        level=1
    ))

    # r62x: Three or more same suit
    def three_or_more_same_suit(hand: Hand) -> bool:
        counts = {}
        for c in hand:
            s = get_suit(c)
            counts[s] = counts.get(s, 0) + 1
        return max(counts.values()) >= 3

    rules.append(Rule(
        id="Three_or_more_same_suit",
        token="r62x",
        name="At least 3 cards same suit",
        predicate=three_or_more_same_suit,
        family="GLOBAL",
        description="At least 3 cards share the same suit (e.g., 3 hearts).",
        composition=C("ge", C("max", C("count_per_suit")), value=3),
        primitives_used=["count_per_suit", "max", "ge"],
        level=1
    ))

    # r63x: Two pairs ranks
    def two_pairs_ranks(hand: Hand) -> bool:
        counts = {}
        for c in hand:
            r = get_rank(c)
            counts[r] = counts.get(r, 0) + 1
        pairs = sum(1 for v in counts.values() if v >= 2)
        return pairs >= 2

    rules.append(Rule(
        id="Two_pairs_ranks",
        token="r63x",
        name="Two pairs (different ranks)",
        predicate=two_pairs_ranks,
        family="COUNT",
        description="Two different ranks each appear at least twice (e.g., two 5s and two Kings).",
        composition=C("ge", C("count_pairs", C("group_by_rank")), value=2),
        primitives_used=["group_by_rank", "count_pairs", "ge"],
        level=1
    ))

    # r64x: Two pairs suits
    def two_pairs_suits(hand: Hand) -> bool:
        counts = {}
        for c in hand:
            s = get_suit(c)
            counts[s] = counts.get(s, 0) + 1
        pairs = sum(1 for v in counts.values() if v >= 2)
        return pairs >= 2

    rules.append(Rule(
        id="Two_pairs_suits",
        token="r64x",
        name="Two pairs (different suits)",
        predicate=two_pairs_suits,
        family="COUNT",
        description="Two different suits each appear at least twice (e.g., 2+ hearts and 2+ spades).",
        composition=C("ge", C("count_pairs", C("group_by_suit")), value=2),
        primitives_used=["group_by_suit", "count_pairs", "ge"],
        level=1
    ))

    # r65x: Halves uniform color equal (biconditional)
    # NOTE: This is extensionally identical to r12x (same predicate, same behavior).
    # In JS rules.js both r12x and r65x map to Halves_uniform_color_equal.
    # We keep r65x as an alias with a distinct Python id for the 8-family design.
    def halves_uniform_color_equal_v2(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_h = hand[:k]
        right_h = hand[n - k:]
        left_uni = len(set(get_color(c) for c in left_h)) <= 1
        right_uni = len(set(get_color(c) for c in right_h)) <= 1
        return left_uni == right_uni

    rules.append(Rule(
        id="Halves_uniform_color_equal_r65x",
        token="r65x",
        name="Both halves uniform color (or neither)",
        predicate=halves_uniform_color_equal_v2,
        family="HIER",
        description="Either both halves are all one color, OR neither half is uniform. Mixed fails. (Alias of r12x for 8-family design.)",
        composition=C("eq", C("uniform_color", C("left_half")), C("uniform_color", C("right_half"))),
        primitives_used=["halves", "uniform_color", "eq"],
        level=2
    ))

    # r66x: Halves majority suit equal (biconditional)
    def halves_majority_suit_equal(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_h = hand[:k]
        right_h = hand[n - k:]
        def has_majority_suit(h):
            counts = {}
            for c in h:
                s = get_suit(c)
                counts[s] = counts.get(s, 0) + 1
            return max(counts.values()) >= 2 if counts else False
        return has_majority_suit(left_h) == has_majority_suit(right_h)

    rules.append(Rule(
        id="Halves_majority_suit_equal",
        token="r66x",
        name="Both halves have 2+ same suit (or neither)",
        predicate=halves_majority_suit_equal,
        family="HIER",
        description="Either both halves have at least 2 cards of the same suit, OR neither does.",
        composition=C("eq", C("has_pair_suit", C("left_half")), C("has_pair_suit", C("right_half"))),
        primitives_used=["halves", "has_pair_suit", "eq"],
        level=2
    ))

    # r67x: Halves both AP len3 step1 — both halves contain 3 consecutive ranks
    def halves_both_ap_len3_step1(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_h = hand[:k]
        right_h = hand[n - k:]
        def has_consec3(h):
            vals = set(get_rank_val(c) for c in h)
            for v in vals:
                if (v + 1) in vals and (v + 2) in vals:
                    return True
            return False
        return has_consec3(left_h) and has_consec3(right_h)

    rules.append(Rule(
        id="Halves_both_AP_len3_step1",
        token="r67x",
        name="Both halves have 3 consecutive ranks",
        predicate=halves_both_ap_len3_step1,
        family="HIER",
        description="Each half must contain three consecutive ranks (like 4, 5, 6).",
        composition=C("and", C("has_AP", C("left_half"), length=3, step=1), C("has_AP", C("right_half"), length=3, step=1)),
        primitives_used=["halves", "has_AP", "and", "get_rank_val"],
        level=3
    ))

    # r68x: Halves both adj rank or suit — both halves satisfy adjacency constraint
    def halves_both_adj_rank_or_suit(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        left_h = hand[:k]
        right_h = hand[n - k:]
        def satisfies_adj(h):
            for i in range(len(h) - 1):
                if not (get_rank(h[i]) == get_rank(h[i+1]) or get_suit(h[i]) == get_suit(h[i+1])):
                    return False
            return True
        return satisfies_adj(left_h) and satisfies_adj(right_h)

    rules.append(Rule(
        id="Halves_both_adj_rank_or_suit",
        token="r68x",
        name="Both halves: neighbors share rank or suit",
        predicate=halves_both_adj_rank_or_suit,
        family="HIER",
        description="In each half, every adjacent pair shares either rank or suit.",
        composition=C("and", C("adj_check", C("left_half")), C("adj_check", C("right_half"))),
        primitives_used=["halves", "adjacent_pairs", "all", "get_rank", "get_suit", "eq", "or", "and"],
        level=2
    ))

    return rules


# ============================================================================
# DISCOVERED ABSTRACTIONS
# These are compositional patterns that appear in multiple rules
# ============================================================================

DISCOVERED_ABSTRACTIONS = {
    "halves_equal": {
        "name": "Halves Equal (Property)",
        "composition": "λF. λh. F(left_half(h)) = F(right_half(h))",
        "description": "Check if applying a property function to both halves gives equal results",
        "used_by": ["Halves_copy_suits", "Halves_copy_colors", "Halves_copy_ranks",
                    "Halves_copy_altcolor1", "Halves_copy_altcolor2"],
        "level": 4,
        "frequency": 5
    },
    "halves_property_equal": {
        "name": "Halves Property Equal (Boolean)",
        "composition": "λP. λh. P(left_half(h)) = P(right_half(h))",
        "description": "Check if a boolean property holds equally in both halves (both true or both false)",
        "used_by": ["Halves_same_color", "Halves_AP_step1_equal", "Halves_hearts_presence_equal",
                    "Halves_AP_len3_any_equal", "Halves_AP_len2_step1_equal"],
        "level": 4,
        "frequency": 5
    },
    "seq_palindrome": {
        "name": "Sequence Palindrome",
        "composition": "λF. λh. map(F, h) = reverse(map(F, h))",
        "description": "Check if the sequence of property values is a palindrome",
        "used_by": ["Suits_palindrome", "Colors_palindrome", "Ranks_palindrome",
                    "AltColor1_palindrome", "AltColor2_palindrome"],
        "level": 4,
        "frequency": 5
    },
    "terminals_equal": {
        "name": "Terminals Equal",
        "composition": "λF. λh. F(first(h)) = F(last(h))",
        "description": "Check if first and last cards share a property",
        "used_by": ["Ends_same_suit", "Ends_same_color", "Ends_same_altcolor1"],
        "level": 4,
        "frequency": 3
    },
    "shifted_pairs_check": {
        "name": "Shifted Pairs Check",
        "composition": "λk. λR. λh. all(R, shifted_pairs(k, h))",
        "description": "Check if all pairs at offset k satisfy a relation",
        "used_by": ["Shift_half_plus_two", "Shift_half_ge", "Skip2_same_rank_or_suit"],
        "level": 2,
        "frequency": 3
    },
    "adjacent_check": {
        "name": "Adjacent Pairs Check",
        "composition": "λR. λh. all(R, adjacent_pairs(h))",
        "description": "Check if all adjacent pairs satisfy a relation",
        "used_by": ["Adj_same_rank_or_suit", "Adj_rank_gap_le3", "Adj_same_rank_or_color"],
        "level": 2,
        "frequency": 3
    },
    "uniform_property": {
        "name": "Uniform Property",
        "composition": "λF. λh. unique_count(map(F, h)) = 1",
        "description": "Check if all cards share the same property value",
        "used_by": ["Uniform_color", "Uniform_rank_parity"],
        "level": 1,
        "frequency": 2
    },
    "bracket_match": {
        "name": "Bracket Match (PDA)",
        "composition": "λopeners. λclosers. λh. PDA_accept(h, openers, closers)",
        "description": "Check if sequence forms well-matched brackets",
        "used_by": ["Well_formed_brackets_by_suit", "Even_opens_next_closes", "Odd_opens_next_closes"],
        "level": 3,
        "frequency": 3
    },
    "has_AP": {
        "name": "Has Arithmetic Progression",
        "composition": "λlen. λstep. λaligned. λh. exists_AP(h, len, step, aligned)",
        "description": "Check if hand contains an arithmetic progression of ranks",
        "used_by": ["AP_len3_anywhere_anyk", "AP_len3_step2_anywhere", "AP_len4_step2_anywhere"],
        "level": 3,
        "frequency": 3
    }
}


# ============================================================================
# INSTANTIATE ALL RULES
# ============================================================================

ALL_RULES = create_all_rules()
RULE_DICT = {rule.id: rule for rule in ALL_RULES}
RULE_BY_TOKEN = {rule.token: rule for rule in ALL_RULES}


def get_rule(rule_id: str) -> Rule:
    """Get rule by ID."""
    if rule_id not in RULE_DICT:
        raise ValueError(f"Unknown rule ID: {rule_id}")
    return RULE_DICT[rule_id]


def get_rule_by_token(token: str) -> Rule:
    """Get rule by token (e.g., 'r1x')."""
    if token not in RULE_BY_TOKEN:
        raise ValueError(f"Unknown rule token: {token}")
    return RULE_BY_TOKEN[token]


def get_rules_by_family(family: str) -> List[Rule]:
    """Get all rules in a given family."""
    return [r for r in ALL_RULES if r.family == family]


def get_all_families() -> List[str]:
    """Get list of all rule families."""
    return sorted(set(r.family for r in ALL_RULES))


if __name__ == "__main__":
    print(f"=== Rule Catalogue ===")
    print(f"Total rules: {len(ALL_RULES)}")
    print()

    # Count by family
    families = {}
    for rule in ALL_RULES:
        families.setdefault(rule.family, []).append(rule)

    print("Rules by family:")
    for family in sorted(families.keys()):
        print(f"  {family}: {len(families[family])}")

    print()
    print("First 5 rules with compositions:")
    for rule in ALL_RULES[:5]:
        print(f"  {rule.id}:")
        print(f"    Composition: {rule.composition_str()}")
        print(f"    Lambda: {rule.lambda_str()}")
        print()
