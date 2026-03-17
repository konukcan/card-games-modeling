"""Rule-based DSL-to-English translator for card game programs.

Converts S-expression programs like ``(λ all (λ eq (get_color $0) RED) $0)``
into readable English like ``"All cards are red"``.

Design goals:
  - SHORT, human-readable output suitable for chart labels
  - Never crash -- always return *something* (raw DSL string as fallback)
  - Cache results so repeated translations are free

The translator works in three stages:
  1. **Parse** the S-expression into a nested Python list (AST).
  2. **Translate** the AST recursively into English, using pattern
     matching for common idioms (even/odd rank, color counts, etc.).
  3. **Polish** the result (capitalise first letter, trim whitespace).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Module-level translation cache: raw program string -> English string.
# ---------------------------------------------------------------------------
_cache: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Constants -- map DSL tokens to readable names
# ---------------------------------------------------------------------------
_SUIT_NAMES = {"HEARTS": "hearts", "DIAMONDS": "diamonds",
               "CLUBS": "clubs", "SPADES": "spades"}
_COLOR_NAMES = {"RED": "red", "BLACK": "black"}
_CONST_NAMES = {**_SUIT_NAMES, **_COLOR_NAMES}

# Comparison operator symbols for compact display.
_CMP_SYMBOLS = {"eq": "=", "lt": "<", "le": "\u2264", "gt": ">", "ge": "\u2265"}

# Inverse comparisons (used when operand order is flipped for readability).
_CMP_INVERSE = {"lt": ">", "le": "\u2265", "gt": "<", "ge": "\u2264", "eq": "="}

# Aggregate helpers -- these take a hand and return a number.
_AGGREGATE_LABELS: Dict[str, str] = {
    "n_unique_suits": "unique suits",
    "n_unique_ranks": "unique ranks",
    "n_unique_colors": "unique colors",
    "max_suit_count": "max suit count",
    "n_repeated_ranks": "repeated ranks",
    "n_repeated_suits": "repeated suits",
    "sum_ranks": "sum of ranks",
    "max_rank": "max rank",
    "min_rank": "min rank",
    "length": "hand size",
    "half_len": "half hand size",
}


# ===================================================================
# 1. S-expression parser
# ===================================================================

def _tokenize(s: str) -> List[str]:
    """Split an S-expression string into tokens.

    Parentheses and the lambda symbol ``λ`` become their own tokens.
    Everything else is split on whitespace.

    Example::

        >>> _tokenize("(λ all (λ eq x y) $0)")
        ['(', 'λ', 'all', '(', 'λ', 'eq', 'x', 'y', ')', '$0', ')']
    """
    # Pad parens with spaces so split() isolates them.
    s = s.replace("(", " ( ").replace(")", " ) ")
    return s.split()


def _parse_tokens(tokens: List[str], pos: int = 0):
    """Recursively parse tokens starting at *pos*.

    Returns ``(ast_node, next_pos)`` where *ast_node* is either a string
    (atom) or a list of nodes (compound expression).
    """
    if pos >= len(tokens):
        return None, pos

    tok = tokens[pos]

    if tok == "(":
        # Collect children until matching ")".
        children: List[Any] = []
        pos += 1  # skip '('
        while pos < len(tokens) and tokens[pos] != ")":
            child, pos = _parse_tokens(tokens, pos)
            if child is not None:
                children.append(child)
        pos += 1  # skip ')'
        return children, pos
    elif tok == ")":
        # Shouldn't happen if input is well-formed, but be safe.
        return None, pos + 1
    else:
        return tok, pos + 1


def parse_sexpr(program: str) -> Any:
    """Parse a complete S-expression string into nested lists.

    Atoms (identifiers, numbers, ``$0``, ``λ``) are plain strings.
    Compound expressions are Python lists.

    Example::

        >>> parse_sexpr("(λ all (λ eq x y) $0)")
        ['λ', 'all', ['λ', 'eq', 'x', 'y'], '$0']
    """
    tokens = _tokenize(program)
    ast, _ = _parse_tokens(tokens, 0)
    return ast


# ===================================================================
# 2. Static evaluator (for constant arithmetic)
# ===================================================================

def _try_eval_number(node) -> Optional[int]:
    """Try to statically evaluate *node* to an integer.

    Handles literal ints and arithmetic expressions built from
    ``+``, ``-``, ``mod`` over literal ints.  Returns ``None`` if the
    node contains variables or unrecognised forms.
    """
    if isinstance(node, str):
        try:
            return int(node)
        except ValueError:
            return None

    if isinstance(node, list) and len(node) == 3:
        op, a, b = node
        va = _try_eval_number(a)
        vb = _try_eval_number(b)
        if va is not None and vb is not None:
            if op == "+":
                return va + vb
            if op == "-":
                return va - vb
            if op == "mod" and vb != 0:
                return va % vb
    return None


# ===================================================================
# 3. Recursive translator
# ===================================================================

def _is_var(node) -> bool:
    """Return True if *node* is a lambda variable reference like ``$0``."""
    return isinstance(node, str) and node.startswith("$")


def _is_hand(node) -> bool:
    """Return True if *node* refers to the hand (outermost ``$0``)."""
    return node == "$0"


def _translate(node, card_var: str = "card") -> str:
    """Recursively translate an AST node to English.

    *card_var* is a label for the current element inside a HOF context
    (all, any, filter, map).  At the top level it defaults to ``"card"``.

    The function tries a sequence of pattern matches from most specific
    (idiomatic) to most generic.  The fallback is always the raw string
    representation.
    """

    # --- Atoms ---------------------------------------------------------
    if isinstance(node, str):
        # Constant names
        if node in _CONST_NAMES:
            return _CONST_NAMES[node]
        # Lambda variable -- use context label
        if node == "$0":
            return card_var
        # Numeric literal
        try:
            int(node)
            return node
        except ValueError:
            pass
        return node

    if not isinstance(node, list) or len(node) == 0:
        return str(node)

    head = node[0]

    # --- Strip outer lambda(s) ----------------------------------------
    # The outermost ``λ`` binds the hand.  We strip it and translate the
    # body, mapping ``$0`` -> ``hand``.  But we don't literally say
    # "hand" everywhere -- downstream helpers know the context.
    if head == "λ" and len(node) >= 2:
        body = node[1] if len(node) == 2 else node[1:]
        # If body is itself a list with one element, unwrap it.
        if isinstance(body, list) and len(body) == 1:
            body = body[0]
        return _translate_top(body)

    # --- everything else in generic form ------------------------------
    return _translate_expr(node, card_var)


def _translate_top(node, hand_label: str = "hand") -> str:
    """Translate the body of the outermost lambda (hand-level expression).

    At this level ``$0`` means *the hand*.
    """

    if isinstance(node, str):
        if node in _CONST_NAMES:
            return _CONST_NAMES[node]
        if node == "$0":
            return hand_label
        return node

    if not isinstance(node, list) or len(node) == 0:
        return str(node)

    head = node[0]

    # -- Boolean combinators -------------------------------------------
    if head == "and" and len(node) == 3:
        left = _translate_top(node[1], hand_label)
        right = _translate_top(node[2], hand_label)
        return f"{left} and {right}"

    if head == "or" and len(node) == 3:
        left = _translate_top(node[1], hand_label)
        right = _translate_top(node[2], hand_label)
        return f"{left} or {right}"

    if head == "not" and len(node) == 2:
        inner = _translate_top(node[1], hand_label)
        return _negate(inner)

    if head == "if" and len(node) == 4:
        cond = _translate_top(node[1], hand_label)
        then_ = _translate_top(node[2], hand_label)
        else_ = _translate_top(node[3], hand_label)
        return f"if {cond} then {then_} else {else_}"

    # -- Quantifiers (all / any) with inline lambda --------------------
    if head in ("all", "any") and len(node) == 3:
        return _translate_quantifier(head, node[1], node[2], hand_label)

    # -- Shortcut predicates -------------------------------------------
    if head == "has_suit" and len(node) == 3:
        suit = _const_label(node[2])
        return f"has {suit}"

    if head == "has_color" and len(node) == 3:
        color = _const_label(node[2])
        return f"has {color} cards"

    # -- Aggregates with comparisons -----------------------------------
    # Pattern: (cmp (agg $0) n) or (cmp n (agg $0))
    if head in _CMP_SYMBOLS and len(node) == 3:
        return _translate_comparison_top(head, node[1], node[2], hand_label)

    # -- count_suit / count_color --------------------------------------
    if head == "count_suit" and len(node) == 3:
        suit = _const_label(node[2])
        return f"count of {suit}"

    if head == "count_color" and len(node) == 3:
        color = _const_label(node[2])
        return f"count of {color} cards"

    # -- Aggregate applied to hand (standalone, without comparison) -----
    if head in _AGGREGATE_LABELS and len(node) == 2 and _is_hand(node[1]):
        return _AGGREGATE_LABELS[head]

    # -- Fallback: generic expression translation ----------------------
    return _translate_expr(node, hand_label)


# -----------------------------------------------------------------------
# Quantifiers
# -----------------------------------------------------------------------

def _translate_quantifier(quant: str, pred, collection, hand_label: str) -> str:
    """Translate ``(all <pred> <collection>)`` and ``(any <pred> <collection>)``.

    *pred* is usually a lambda over a single card.
    """
    # The predicate is typically ``(λ <body>)`` where ``$0`` = card.
    if isinstance(pred, list) and len(pred) >= 2 and pred[0] == "λ":
        body = pred[1] if len(pred) == 2 else pred[1:]
        card_desc = _translate_card_pred(body)
        if quant == "all":
            return f"all cards {card_desc}"
        else:
            # "some card" needs singular verb -- rewrite "are X" -> "is X"
            if card_desc.startswith("are "):
                return f"some card is {card_desc[4:]}"
            if card_desc.startswith("have "):
                return f"some card has {card_desc[5:]}"
            return f"some card {card_desc}"

    # pred is a bare function name
    if isinstance(pred, str):
        q_word = "all cards" if quant == "all" else "some card"
        return f"{q_word} satisfy {pred}"

    q_word = "all cards" if quant == "all" else "some card"
    return f"{q_word} satisfy {_translate_expr(pred, 'card')}"


def _translate_card_pred(node) -> str:
    """Translate a predicate body where ``$0`` = a single card.

    Tries to produce compact descriptions like "are red" or "have even rank".
    """
    if isinstance(node, str):
        if node == "$0":
            return "card"
        if node in _CONST_NAMES:
            return _CONST_NAMES[node]
        return node

    if not isinstance(node, list) or len(node) == 0:
        return str(node)

    head = node[0]

    # -- even/odd rank detection ---------------------------------------
    # (eq (mod (rank_val $0) 2) 0) -> "have even rank"
    # (eq (mod (rank_val $0) 2) 1) -> "have odd rank"
    if head == "eq" and len(node) == 3:
        left, right = node[1], node[2]
        parity = _detect_parity(left, right)
        if parity:
            return parity

    # -- color check: (eq (get_color $0) RED) -> "are red" ------------
    if head == "eq" and len(node) == 3:
        if _is_color_check(node[1], node[2]):
            color = _const_label(node[2])
            return f"are {color}"
        if _is_color_check(node[2], node[1]):
            color = _const_label(node[1])
            return f"are {color}"

    # -- suit check: (eq (get_suit $0) HEARTS) -> "are hearts" --------
    if head == "eq" and len(node) == 3:
        if _is_suit_check(node[1], node[2]):
            suit = _const_label(node[2])
            return f"are {suit}"
        if _is_suit_check(node[2], node[1]):
            suit = _const_label(node[1])
            return f"are {suit}"

    # -- rank comparison: (ge (rank_val $0) 5) -> "have rank >= 5" ----
    if head in _CMP_SYMBOLS and len(node) == 3:
        return _translate_card_comparison(head, node[1], node[2])

    # -- boolean combinators inside card predicate ---------------------
    if head == "and" and len(node) == 3:
        left = _translate_card_pred(node[1])
        right = _translate_card_pred(node[2])
        return f"{left} and {right}"

    if head == "or" and len(node) == 3:
        left = _translate_card_pred(node[1])
        right = _translate_card_pred(node[2])
        return f"{left} or {right}"

    if head == "not" and len(node) == 2:
        inner = _translate_card_pred(node[1])
        return f"not ({inner})"

    # -- generic fallback for card-level expressions -------------------
    return _translate_expr(node, "card")


# -----------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------

def _translate_comparison_top(op: str, left, right, hand_label: str) -> str:
    """Translate a comparison at the hand/top level.

    Tries to put the "meaningful" side (aggregate, count) on the left
    and the number on the right for readability.
    """
    sym = _CMP_SYMBOLS[op]

    # -- even/odd parity at hand level ---------------------------------
    parity = _detect_parity(left, right)
    if parity:
        return parity

    # Try to identify aggregate on left, number on right.
    left_str = _translate_top(left, hand_label)
    right_str = _translate_top(right, hand_label)

    # Static evaluation for either side.
    lv = _try_eval_number(left)
    rv = _try_eval_number(right)

    # If both are numbers and the whole thing can be evaluated, do so.
    if lv is not None and rv is not None:
        result = _eval_cmp(op, lv, rv)
        if result is not None:
            return "always" if result else "never"

    # If the number is on the LEFT (e.g., (lt 5 (max_suit_count $0))),
    # flip to put the aggregate first with the inverse operator.
    if lv is not None and rv is None:
        inv_sym = _CMP_INVERSE[op]
        return f"{right_str} {inv_sym} {lv}"

    # Normal order: aggregate on left.
    if rv is not None:
        return f"{left_str} {sym} {rv}"

    return f"{left_str} {sym} {right_str}"


def _translate_card_comparison(op: str, left, right) -> str:
    """Translate a comparison inside a card predicate."""
    sym = _CMP_SYMBOLS[op]

    lv = _try_eval_number(left)
    rv = _try_eval_number(right)

    left_str = _translate_card_accessor(left)
    right_str = _translate_card_accessor(right)

    # Number on left -- flip.
    if lv is not None and rv is None:
        inv_sym = _CMP_INVERSE[op]
        return f"have {right_str} {inv_sym} {lv}"

    if rv is not None:
        return f"have {left_str} {sym} {rv}"

    return f"have {left_str} {sym} {right_str}"


def _translate_card_accessor(node) -> str:
    """Translate a card accessor like ``(rank_val $0)`` to ``'rank'``."""
    if isinstance(node, str):
        v = _try_eval_number(node)
        if v is not None:
            return str(v)
        if node == "$0":
            return "card"
        if node in _CONST_NAMES:
            return _CONST_NAMES[node]
        return node

    if isinstance(node, list) and len(node) == 2:
        head, arg = node
        if head == "rank_val" and _is_var(arg):
            return "rank"
        if head == "get_suit" and _is_var(arg):
            return "suit"
        if head == "get_color" and _is_var(arg):
            return "color"
        if head == "get_rank" and _is_var(arg):
            return "rank symbol"

    # Fallback to generic.
    return _translate_expr(node, "card")


# -----------------------------------------------------------------------
# Generic expression translator (fallback)
# -----------------------------------------------------------------------

def _translate_expr(node, ctx_var: str = "hand") -> str:
    """Generic expression translator -- least specific, used as fallback.

    Produces slightly verbose but always-correct translations.
    """
    if isinstance(node, str):
        v = _try_eval_number(node)
        if v is not None:
            return str(v)
        if node == "$0":
            return ctx_var
        if node in _CONST_NAMES:
            return _CONST_NAMES[node]
        return node

    if not isinstance(node, list) or len(node) == 0:
        return str(node)

    head = node[0]

    # -- Lambda -- strip and translate body ----------------------------
    if head == "λ" and len(node) >= 2:
        body = node[1] if len(node) == 2 else node[1:]
        return _translate_expr(body, ctx_var)

    # -- Static arithmetic ---------------------------------------------
    v = _try_eval_number(node)
    if v is not None:
        return str(v)

    # -- Arithmetic (non-static) ---------------------------------------
    if head == "+" and len(node) == 3:
        return f"({_translate_expr(node[1], ctx_var)} + {_translate_expr(node[2], ctx_var)})"
    if head == "-" and len(node) == 3:
        return f"({_translate_expr(node[1], ctx_var)} - {_translate_expr(node[2], ctx_var)})"
    if head == "mod" and len(node) == 3:
        return f"({_translate_expr(node[1], ctx_var)} mod {_translate_expr(node[2], ctx_var)})"

    # -- Comparisons ---------------------------------------------------
    if head in _CMP_SYMBOLS and len(node) == 3:
        sym = _CMP_SYMBOLS[head]
        return f"{_translate_expr(node[1], ctx_var)} {sym} {_translate_expr(node[2], ctx_var)}"

    # -- Boolean -------------------------------------------------------
    if head == "and" and len(node) == 3:
        return f"{_translate_expr(node[1], ctx_var)} and {_translate_expr(node[2], ctx_var)}"
    if head == "or" and len(node) == 3:
        return f"{_translate_expr(node[1], ctx_var)} or {_translate_expr(node[2], ctx_var)}"
    if head == "not" and len(node) == 2:
        return f"not ({_translate_expr(node[1], ctx_var)})"

    # -- Quantifiers ---------------------------------------------------
    if head in ("all", "any") and len(node) == 3:
        return _translate_quantifier(head, node[1], node[2], ctx_var)

    # -- if ------------------------------------------------------------
    if head == "if" and len(node) == 4:
        c = _translate_expr(node[1], ctx_var)
        t = _translate_expr(node[2], ctx_var)
        e = _translate_expr(node[3], ctx_var)
        return f"if {c} then {t} else {e}"

    # -- Shortcut predicates -------------------------------------------
    if head == "has_suit" and len(node) == 3:
        suit = _const_label(node[2])
        return f"has {suit}"
    if head == "has_color" and len(node) == 3:
        color = _const_label(node[2])
        return f"has {color} cards"

    # -- count_suit / count_color --------------------------------------
    if head == "count_suit" and len(node) == 3:
        suit = _const_label(node[2])
        return f"count of {suit}"
    if head == "count_color" and len(node) == 3:
        color = _const_label(node[2])
        return f"count of {color} cards"

    # -- Aggregates applied to a variable (hand) -----------------------
    if head in _AGGREGATE_LABELS and len(node) == 2:
        return _AGGREGATE_LABELS[head]

    # -- Card accessors ------------------------------------------------
    if head == "rank_val" and len(node) == 2:
        inner = _translate_expr(node[1], ctx_var)
        if inner == ctx_var:
            return "rank"
        return f"rank of {inner}"
    if head == "get_suit" and len(node) == 2:
        inner = _translate_expr(node[1], ctx_var)
        if inner == ctx_var:
            return "suit"
        return f"suit of {inner}"
    if head == "get_color" and len(node) == 2:
        inner = _translate_expr(node[1], ctx_var)
        if inner == ctx_var:
            return "color"
        return f"color of {inner}"
    if head == "get_rank" and len(node) == 2:
        inner = _translate_expr(node[1], ctx_var)
        if inner == ctx_var:
            return "rank symbol"
        return f"rank symbol of {inner}"

    # -- List operations -----------------------------------------------
    if head == "at" and len(node) == 3:
        lst = _translate_expr(node[1], ctx_var)
        idx_v = _try_eval_number(node[2])
        idx = str(idx_v) if idx_v is not None else _translate_expr(node[2], ctx_var)
        if lst == ctx_var:
            return f"card at pos {idx}"
        return f"{lst}[{idx}]"

    if head == "head" and len(node) == 2:
        lst = _translate_expr(node[1], ctx_var)
        return f"first of {lst}" if lst != ctx_var else "first card"

    if head == "last" and len(node) == 2:
        lst = _translate_expr(node[1], ctx_var)
        return f"last of {lst}" if lst != ctx_var else "last card"

    if head == "length" and len(node) == 2:
        lst = _translate_expr(node[1], ctx_var)
        return f"length of {lst}" if lst != ctx_var else "hand size"

    if head == "half_len" and len(node) == 2:
        return "half hand size"

    if head == "reverse" and len(node) == 2:
        return f"reversed {_translate_expr(node[1], ctx_var)}"

    if head == "first_half" and len(node) == 2:
        return "first half"

    if head == "second_half" and len(node) == 2:
        return "second half"

    if head == "sort_by_rank" and len(node) == 2:
        return "sorted by rank"

    if head == "unique" and len(node) == 2:
        return "unique cards"

    if head == "take" and len(node) == 3:
        n = _try_eval_number(node[1])
        n_str = str(n) if n is not None else _translate_expr(node[1], ctx_var)
        return f"first {n_str} cards"

    if head == "drop" and len(node) == 3:
        n = _try_eval_number(node[1])
        n_str = str(n) if n is not None else _translate_expr(node[1], ctx_var)
        return f"cards after dropping {n_str}"

    if head == "filter" and len(node) == 3:
        pred = _translate_expr(node[1], ctx_var)
        return f"cards where {pred}"

    if head == "map" and len(node) == 3:
        fn = _translate_expr(node[1], ctx_var)
        return f"map {fn}"

    if head == "zip_with" and len(node) == 4:
        fn = _translate_expr(node[1], ctx_var)
        return f"zip with {fn}"

    if head == "adjacent_pairs" and len(node) == 2:
        return "adjacent pairs"

    if head == "running_sum" and len(node) == 2:
        return "running sum"

    if head == "signum" and len(node) == 2:
        inner = _translate_expr(node[1], ctx_var)
        return f"sign of {inner}"

    if head == "suit_to_int" and len(node) == 2:
        inner = _translate_expr(node[1], ctx_var)
        return f"suit as int of {inner}"

    # -- Catch-all: join tokens ----------------------------------------
    parts = [_translate_expr(child, ctx_var) for child in node]
    return " ".join(parts)


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

def _const_label(node) -> str:
    """Return the human-readable label for a constant, or raw string."""
    if isinstance(node, str) and node in _CONST_NAMES:
        return _CONST_NAMES[node]
    return _translate_expr(node) if not isinstance(node, str) else node


def _detect_parity(left, right) -> Optional[str]:
    """Detect ``(eq (mod (rank_val $0) 2) 0/1)`` -> even/odd."""
    # (eq (mod (rank_val $0) 2) 0)
    if (isinstance(left, list) and len(left) == 3
            and left[0] == "mod"
            and isinstance(left[1], list) and len(left[1]) == 2
            and left[1][0] == "rank_val"
            and _try_eval_number(left[2]) == 2):
        rv = _try_eval_number(right)
        if rv == 0:
            return "have even rank"
        if rv == 1:
            return "have odd rank"
    # Symmetric: number on left.
    if (isinstance(right, list) and len(right) == 3
            and right[0] == "mod"
            and isinstance(right[1], list) and len(right[1]) == 2
            and right[1][0] == "rank_val"
            and _try_eval_number(right[2]) == 2):
        lv = _try_eval_number(left)
        if lv == 0:
            return "have even rank"
        if lv == 1:
            return "have odd rank"
    return None


def _is_color_check(accessor, constant) -> bool:
    """Check if ``accessor`` is ``(get_color $0)`` and ``constant`` is a color."""
    return (isinstance(accessor, list) and len(accessor) == 2
            and accessor[0] == "get_color"
            and isinstance(constant, str) and constant in _COLOR_NAMES)


def _is_suit_check(accessor, constant) -> bool:
    """Check if ``accessor`` is ``(get_suit $0)`` and ``constant`` is a suit."""
    return (isinstance(accessor, list) and len(accessor) == 2
            and accessor[0] == "get_suit"
            and isinstance(constant, str) and constant in _SUIT_NAMES)


def _negate(text: str) -> str:
    """Produce a readable negation of *text*.

    Handles a few common patterns to avoid ugly double-negatives.
    """
    # "has X" -> "no X"
    if text.startswith("has "):
        return f"no {text[4:]}"
    # "has X cards" -> "no X cards"
    # "all cards ..." -> "not all cards ..."
    return f"not ({text})"


def _eval_cmp(op: str, a: int, b: int) -> Optional[bool]:
    """Evaluate a comparison on two known integers."""
    if op == "eq":
        return a == b
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "gt":
        return a > b
    if op == "ge":
        return a >= b
    return None


# ===================================================================
# 4. Public API
# ===================================================================

def translate_dsl(program: str) -> str:
    """Translate a DSL program string to readable English.

    This is the main entry point.  Results are cached so repeated calls
    with the same program string are essentially free.

    Parameters
    ----------
    program : str
        An S-expression like ``"(λ all (λ eq (get_color $0) RED) $0)"``

    Returns
    -------
    str
        A short English description like ``"All cards are red"``.
        If translation fails for any reason, the raw DSL string is
        returned as a fallback (never raises).
    """
    if program in _cache:
        return _cache[program]

    try:
        ast = parse_sexpr(program)
        raw = _translate(ast)
        # Polish: capitalise first letter, strip extra whitespace.
        result = " ".join(raw.split())
        if result:
            result = result[0].upper() + result[1:]
        else:
            result = program
    except Exception:
        # Absolute fallback -- never crash.
        result = program

    _cache[program] = result
    return result


def clear_cache() -> None:
    """Clear the translation cache (mainly for testing)."""
    _cache.clear()
