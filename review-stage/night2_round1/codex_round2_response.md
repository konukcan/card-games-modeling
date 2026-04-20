# Night 2 Round 2 — Codex re-review response

- **Reviewer thread**: 019da9c1-da9f-78e1-9c7b-6410ddac7b06 (continued)
- **Received**: 2026-04-20T07:35Z
- **Score**: 8/10 (up from 5)
- **Verdict**: `almost`

## Findings

1. **`hash(rule_id)` is process-randomized** unless PYTHONHASHSEED is fixed
   (PEP 456). Driver claim of cross-run reproducibility is wrong as written.
   `run_adversarial_hands.py:146-150`.

2. **"Lower-bound proxy" overclaim** in retained_mass warnings. The TV bound
   applies to `p_accept`, not directly to entropy. Once `p` is perturbed,
   entropy can move up OR down depending on which side of 0.5 the error
   lands. `adversarial_hands.py:19-22, 267-275, 366-374`.

3. **Stale `__post_init__` claim** on `splitting_hypotheses` field —
   `AdversarialHand` does not have a `__post_init__`; the field is filled
   later inside the search functions. The Night 1 pattern again: prose
   promises a stronger invariant than the implementation provides.
   `adversarial_hands.py:93-95`.

## Recommendations (4 residual)

1. Replace `hash()` with `zlib.crc32` or `hashlib`.
2. Remove "lower-bound" everywhere; say approximation + bounded
   predictive-probability error.
3. Fix the stale `__post_init__` comment OR implement the method.
4. Add an exact-tie splitter test (boundary p_accept == 0.5).

## Codex's answers to my questions

- Empty posterior fails closed tightly enough (no need to extend to
  1-hypothesis case).
- Tie convention is fine as documented.
- Scope-box on uniform-MC sampling is defensible AS LONG AS the report does
  not claim "worst-case". Targeted sampling would be needed before any
  stronger claim about rare-rule disagreement coverage.
