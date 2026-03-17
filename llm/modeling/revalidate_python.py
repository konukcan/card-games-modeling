#!/usr/bin/env python3
"""Re-validate existing Python hypotheses with fixed sandbox (Counter + normalizer)."""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rules.cards import Card, Suit, Rank, RANK_VALUES, STR_TO_RANK
from rules.cards import Color, suit_to_color, card_color


def _normalize_python_code(code: str) -> str:
    """Normalize already-extracted Python code so it defines a callable `rule`."""
    import re
    stripped = code.strip()
    if stripped.startswith("rule = ") or stripped.startswith("rule="):
        return stripped
    if stripped.startswith("def rule(") or stripped.startswith("def rule ("):
        return stripped
    if stripped.startswith("lambda hand:"):
        return f"rule = {stripped}"
    if "rule = " in stripped or "def rule" in stripped:
        return stripped
    func_match = re.search(r"^def (\w+)\(hand", stripped, re.MULTILINE)
    if func_match:
        func_name = func_match.group(1)
        return f"{stripped}\nrule = {func_name}"
    if "\n" not in stripped and "hand" in stripped:
        return f"rule = lambda hand: {stripped}"
    return stripped


def _make_exec_globals() -> dict:
    from collections import Counter
    return {
        "__builtins__": __builtins__,
        "Card": Card, "Suit": Suit, "Rank": Rank,
        "RANK_VALUES": RANK_VALUES, "STR_TO_RANK": STR_TO_RANK,
        "Color": Color, "suit_to_color": suit_to_color,
        "card_color": card_color,
        "Counter": Counter,
    }


def validate_syntax(code: str) -> bool:
    """Try to compile and exec the code in our sandbox."""
    try:
        compiled = compile(code, "<hypothesis>", "exec")
        g = _make_exec_globals()
        exec(compiled, g)
        # Check that 'rule' is defined and callable
        if "rule" not in g or not callable(g["rule"]):
            return False
        return True
    except Exception:
        return False


def main():
    results_dir = Path(__file__).parent.parent / "results" / "phase1b"
    python_files = sorted(results_dir.glob("gemini-pro__python-freeform__*.json"))

    old_pass = 0
    new_pass = 0
    total = 0
    fixed = []
    still_broken = []

    for f in python_files:
        data = json.loads(f.read_text())
        for hyp in data.get("hypotheses", []):
            total += 1
            old_ok = hyp.get("syntax_ok", False)

            code = hyp.get("code", "")
            normalized = _normalize_python_code(code)
            new_ok = validate_syntax(normalized)

            if old_ok:
                old_pass += 1
            if new_ok:
                new_pass += 1

            if new_ok and not old_ok:
                fixed.append({
                    "rule": data["rule_id"],
                    "rank": hyp["rank"],
                    "desc": hyp["nl_description"][:60]
                })
            elif not new_ok and not old_ok:
                still_broken.append({
                    "rule": data["rule_id"],
                    "rank": hyp["rank"],
                    "code_snippet": code[:80].replace("\n", " | ")
                })

    print(f"\n{'='*60}")
    print(f"Python Re-validation Results")
    print(f"{'='*60}")
    print(f"Total hypotheses: {total}")
    print(f"Old pass count:   {old_pass} ({100*old_pass/total:.1f}%)")
    print(f"New pass count:   {new_pass} ({100*new_pass/total:.1f}%)")
    print(f"Newly fixed:      {len(fixed)}")
    print(f"Still broken:     {len(still_broken)}")

    if fixed:
        print(f"\n--- Newly Fixed ({len(fixed)}) ---")
        for item in fixed:
            print(f"  {item['rule']} rank {item['rank']}: {item['desc']}")

    if still_broken:
        print(f"\n--- Still Broken ({len(still_broken)}) ---")
        for item in still_broken[:20]:
            print(f"  {item['rule']} rank {item['rank']}: {item['code_snippet']}")
        if len(still_broken) > 20:
            print(f"  ... and {len(still_broken) - 20} more")


if __name__ == "__main__":
    main()
