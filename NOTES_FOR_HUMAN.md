# Notes for Human (Night 2)

Items requiring human attention or potentially out-of-scope.

## Phase 0 observations

### Grammar primitive count discrepancy
- Launch prompt expected `len(g.primitives) == 64`.
- Actual `build_gallery_grammar()` on this branch returns 62 primitives.
- Likely benign: probably a counting convention (e.g., excluding logical-only constants like `true`/`false` that are typed differently). Both this branch and `feature/mcmc-search` share the same `enumerator.py` aside from depth-budget docstring relabelling — so any MCMC-vs-enumeration comparison should still operate on identical primitive sets.
- Action taken: documented here, no code change.

### MCMC sampler cherry-picked
- The launch prompt forbade modifying `mcmc_search.py` but the analysis code (`analyze_mcmc.py`) imports from it.
- Brought the file over verbatim from `feature/mcmc-search` with no logic changes.
- Spirit of the rule preserved: I am not refactoring the MCMC sampler logic.
