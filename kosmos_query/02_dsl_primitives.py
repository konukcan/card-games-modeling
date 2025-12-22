"""
DSL Primitives for Card Game Rule Learning

This file describes the ~60 primitives available in the DSL.
Each primitive has:
- A name (used in programs)
- A type signature
- A description of what it does

The type notation uses:
- 'a, 'b = type variables (polymorphic)
- -> = function arrow
- list(t) = list of type t
- Base types: bool, int, card, suit, rank, color, hand (= list(card))
"""

PRIMITIVES = {
    # =========================================================================
    # LEVEL 0: CONSTANTS (19 primitives)
    # =========================================================================

    # Suit constants
    "CLUBS": ("suit", "The clubs suit (♣)"),
    "DIAMONDS": ("suit", "The diamonds suit (♦)"),
    "HEARTS": ("suit", "The hearts suit (♥)"),
    "SPADES": ("suit", "The spades suit (♠)"),

    # Color constants
    "RED": ("color", "Red color (hearts, diamonds)"),
    "BLACK": ("color", "Black color (clubs, spades)"),

    # Numeric constants (0-5, 10-14 for face cards, 17/21 for blackjack)
    "0": ("int", "Zero"),
    "1": ("int", "One"),
    "2": ("int", "Two"),
    "3": ("int", "Three"),
    "4": ("int", "Four"),
    "5": ("int", "Five"),
    "10": ("int", "Ten"),
    "11": ("int", "Jack value"),
    "12": ("int", "Queen value"),
    "13": ("int", "King value"),
    "14": ("int", "Ace value"),

    # Boolean constants
    "true": ("bool", "Boolean true"),
    "false": ("bool", "Boolean false"),

    # =========================================================================
    # LEVEL 1: CARD ACCESSORS (4 primitives)
    # =========================================================================

    "get_suit": ("card -> suit", "Extract the suit from a card"),
    "get_rank": ("card -> rank", "Extract the rank from a card"),
    "rank_val": ("card -> int", "Extract numeric rank value (2-14)"),
    "get_color": ("card -> color", "Extract color (RED or BLACK)"),

    # =========================================================================
    # LEVEL 2: POSITION ACCESS (5 primitives)
    # =========================================================================

    "head": ("list('a) -> 'a", "Get first element of a list"),
    "last": ("list('a) -> 'a", "Get last element of a list"),
    "at": ("list('a) -> int -> 'a", "Get element at index"),
    "length": ("list('a) -> int", "Get length of a list"),
    "reverse": ("list('a) -> list('a)", "Reverse a list"),

    # =========================================================================
    # LEVEL 2b: LIST SLICING (7 primitives)
    # =========================================================================

    "take": ("int -> list('a) -> list('a)", "Take first n elements"),
    "drop": ("int -> list('a) -> list('a)", "Drop first n elements"),
    "zip_with": ("('a -> 'b -> bool) -> list('a) -> list('b) -> list(bool)",
                 "Combine two lists element-wise with a function"),
    "adjacent_pairs": ("list('a) -> list(list('a))",
                       "Get consecutive pairs: [a,b,c] -> [[a,b], [b,c]]"),
    "half_len": ("list('a) -> int", "Get half the length of a list"),
    "first_half": ("list('a) -> list('a)", "Get first half of list"),
    "second_half": ("list('a) -> list('a)", "Get second half of list"),

    # =========================================================================
    # LEVEL 3: DIRECT PROPERTY QUERIES (9 primitives)
    # =========================================================================

    "has_suit": ("hand -> suit -> bool", "Check if hand has any card of suit"),
    "has_color": ("hand -> color -> bool", "Check if hand has any card of color"),
    "count_suit": ("hand -> suit -> int", "Count cards of a specific suit"),
    "count_color": ("hand -> color -> int", "Count cards of a specific color"),
    "all_same_suit": ("hand -> bool", "Check if all cards have same suit (flush)"),
    "all_same_color": ("hand -> bool", "Check if all cards have same color"),
    "n_unique_suits": ("hand -> int", "Count distinct suits in hand"),
    "n_unique_ranks": ("hand -> int", "Count distinct ranks in hand"),
    "n_unique_colors": ("hand -> int", "Count distinct colors in hand"),

    # =========================================================================
    # LEVEL 4: AGGREGATES (3 primitives)
    # =========================================================================

    "sum_ranks": ("hand -> int", "Sum of all rank values in hand"),
    "max_rank": ("hand -> int", "Maximum rank value in hand"),
    "min_rank": ("hand -> int", "Minimum rank value in hand"),

    # =========================================================================
    # LEVEL 5: COMPARISONS (6 primitives)
    # =========================================================================

    "eq": ("'a -> 'a -> bool", "Equality test"),
    "neq": ("'a -> 'a -> bool", "Not-equal test"),
    "lt": ("int -> int -> bool", "Less than"),
    "le": ("int -> int -> bool", "Less than or equal"),
    "gt": ("int -> int -> bool", "Greater than"),
    "ge": ("int -> int -> bool", "Greater than or equal"),

    # =========================================================================
    # LEVEL 6: BOOLEAN OPERATIONS (4 primitives)
    # =========================================================================

    "and": ("bool -> bool -> bool", "Logical AND"),
    "or": ("bool -> bool -> bool", "Logical OR"),
    "not": ("bool -> bool", "Logical NOT"),
    "if": ("bool -> 'a -> 'a -> 'a", "If-then-else"),

    # =========================================================================
    # LEVEL 7: HIGHER-ORDER FUNCTIONS (5 primitives)
    # =========================================================================

    "map": ("('a -> 'b) -> list('a) -> list('b)", "Apply function to each element"),
    "filter": ("('a -> bool) -> list('a) -> list('a)", "Keep elements matching predicate"),
    "all": ("('a -> bool) -> list('a) -> bool", "Check all elements satisfy predicate"),
    "any": ("('a -> bool) -> list('a) -> bool", "Check any element satisfies predicate"),
    "unique": ("list('a) -> list('a)", "Remove duplicates, preserving order"),

    # =========================================================================
    # LEVEL 8: ARITHMETIC (3 primitives)
    # =========================================================================

    "+": ("int -> int -> int", "Addition"),
    "-": ("int -> int -> int", "Subtraction"),
    "mod": ("int -> int -> int", "Modulo (remainder)"),
}

# =========================================================================
# EXAMPLE PROGRAMS
# =========================================================================

EXAMPLE_PROGRAMS = {
    "all_same_suit": "(λ all_same_suit $0)",
    # Description: Check if all cards have same suit
    # Primitives used: all_same_suit

    "first_and_last_same_suit": "(λ eq (get_suit (head $0)) (get_suit (last $0)))",
    # Description: Check if first and last cards share suit
    # Primitives used: eq, get_suit, head, last

    "halves_copy_suits": "(λ eq (map get_suit (first_half $0)) (map get_suit (second_half $0)))",
    # Description: Check if left half suits equal right half suits (in order)
    # Primitives used: eq, map, get_suit, first_half, second_half

    "sorted_by_rank": "(λ all (λ le (rank_val (head $0)) (rank_val (last $0))) (adjacent_pairs $0))",
    # Description: Check if ranks are in non-decreasing order
    # Primitives used: all, le, rank_val, head, last, adjacent_pairs

    "at_least_3_hearts": "(λ ge (count_suit $0 HEARTS) 3)",
    # Description: Check if hand has at least 3 hearts
    # Primitives used: ge, count_suit, HEARTS, 3

    "suits_palindrome": "(λ eq (map get_suit $0) (reverse (map get_suit $0)))",
    # Description: Check if suit sequence is palindromic
    # Primitives used: eq, map, get_suit, reverse
}

# =========================================================================
# DISCOVERED ABSTRACTIONS
# =========================================================================

DISCOVERED_ABSTRACTIONS = {
    "halves_equal_F": {
        "definition": "(λ λ eq (map $1 (first_half $0)) (map $1 (second_half $0)))",
        "type": "('a -> 'b) -> list('a) -> bool",
        "description": "Check if property F is equal for left and right halves",
        "used_by": ["Halves_copy_suits", "Halves_copy_colors", "Halves_copy_ranks"],
    },

    "terminals_equal_F": {
        "definition": "(λ λ eq ($1 (head $0)) ($1 (last $0)))",
        "type": "('a -> 'b) -> list('a) -> bool",
        "description": "Check if first and last elements have equal property F",
        "used_by": ["Ends_same_suit", "Ends_same_color"],
    },

    "seq_palindrome_F": {
        "definition": "(λ λ eq (map $1 $0) (reverse (map $1 $0)))",
        "type": "('a -> 'b) -> list('a) -> bool",
        "description": "Check if sequence of property F values is palindromic",
        "used_by": ["Suits_palindrome", "Colors_palindrome", "Ranks_palindrome"],
    },
}

# =========================================================================
# PRIMITIVE USAGE BY RULE FAMILY
# =========================================================================

FAMILY_PRIMITIVES = {
    "LOCAL": ["head", "last", "at", "get_suit", "get_rank", "eq", "is_sorted"],
    "COUNT": ["count_suit", "count_color", "n_unique_suits", "eq", "ge", "le"],
    "PAL": ["map", "reverse", "eq", "get_suit", "get_color", "get_rank"],
    "COPY": ["first_half", "second_half", "map", "eq", "get_suit", "get_color"],
    "SHIFT": ["first_half", "second_half", "zip_with", "ge", "rank_val"],
    "ADJ": ["adjacent_pairs", "all", "any", "or", "eq", "get_suit", "get_rank"],
    "PARITY": ["mod", "all", "eq", "rank_val"],
    "AP": ["adjacent_pairs", "-", "eq", "all", "rank_val"],
}
