s# Brainstorming Seed: MCMC-Based Wake Phase for DreamCoder

## Context

The current card-games-modelling project uses a DreamCoder-inspired wake-sleep loop where the "wake" phase uses **top-down best-first enumeration** to find programs that solve tasks. This works well for programs up to depth ~5-6, but many interesting rules (positional patterns, halves-copy, distribution-matching) require depth 7-9 and are effectively unreachable.

An alternative approach, used by Piantadosi's LOTlib3 framework, is **MCMC (tree-regeneration Metropolis-Hastings)** search over the program space. Instead of systematically enumerating all programs in cost order, MCMC performs a random walk through program space, proposing local modifications (subtree regeneration) and accepting/rejecting based on the posterior (prior × likelihood).

## The Question

Could we replace or supplement the enumeration-based wake phase with MCMC-based program search? What would this look like architecturally, and what are the trade-offs?

## Key Advantages of MCMC Over Enumeration

1. **Reaches deep programs.** MCMC can jump to depth-9 programs without exhaustively searching depths 1-8 first. A single proposal step can regenerate a large subtree, landing directly on a complex program.

2. **Guided by likelihood.** Enumeration finds programs in cost order regardless of the task. MCMC concentrates samples in high-posterior regions — programs that are both simple (high prior) AND fit the data well (high likelihood). This is much more efficient when the solution is deep but the likelihood surface is informative.

3. **Naturally handles the constant problem.** MCMC can propose specific constants (rank value 9, index 3) as part of subtree regeneration without needing to enumerate all possible constants at every integer-typed hole.

4. **Compatible with library learning.** The grammar (PCFG) that defines the prior can still be updated via compression, and the MCMC proposals are drawn from the grammar, so learned abstractions automatically guide future search.

## Key Challenges

1. **Mixing time.** MCMC may get stuck in local optima. Tree-regeneration proposals that replace large subtrees help, but convergence guarantees are weak.

2. **No exhaustiveness guarantee.** Enumeration guarantees finding the shortest program up to the cost bound. MCMC may miss it.

3. **Integration with recognition network.** The current recognition network biases the grammar weights for enumeration. How would it interact with MCMC proposals?

4. **Parallelization.** Enumeration is embarrassingly parallel (different cost ranges on different workers). MCMC chains are sequential, though multiple independent chains can run in parallel.

## Literature and References

### LOTlib3 (Piantadosi)
- **Repository:** https://github.com/piantado/LOTlib3
- **Search method:** Tree-regeneration Metropolis-Hastings MCMC
- **Proposal distribution:** Regenerate a randomly selected subtree by sampling from the PCFG prior
- **Key detail:** Uses `maxnodes=25` hard cutoff to prevent chains from wandering into overly complex programs
- **Implementation:** Python 3, untyped PCFG grammar, `eval()` execution

### Piantadosi, Tenenbaum & Goodman (2016) "The Logical Primitives of Thought"
- **MCMC details:** 250,000 steps of Metropolis-Hastings per concept per data point
- **Top 250 hypotheses stored** to form a finite hypothesis space
- **Inference over parameters:** 10,000 MCMC steps over parameters, alternating between likelihood and prior parameters
- **Paper:** https://colala.berkeley.edu/papers/piantadosi2016logical.pdf

### Original DreamCoder (Ellis et al., 2021)
- **Current approach:** Top-down best-first enumeration with PCFG costs
- **Recognition network:** Biases grammar weights per-task to guide enumeration
- **Paper:** https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf

### Hybrid Approaches
- **Stochastic search + enumeration:** Some systems use MCMC to find an approximate solution, then enumerate nearby to find the optimal one
- **Evolutionary program synthesis:** Genetic programming uses population-based stochastic search over programs — related but different from MCMC
- **Neural-guided MCMC:** Use the recognition network to define the proposal distribution (regenerate subtrees using task-conditioned grammar weights)

### Rule et al. (2024) "Symbolic metaprogram search"
- **Key idea:** Search over *metaprograms* (programs that revise programs) rather than programs directly
- **Relevance:** An alternative to both enumeration and MCMC — the search space is over program transformations
- **Paper:** Nature Communications, https://www.nature.com/articles/s41467-024-50966-x

## Current System Architecture (for reference)

### Files
| Component | File |
|-----------|------|
| Enumeration (wake phase) | `src/dreamcoder_core/enumeration.py` |
| Grammar (PCFG) | `src/dreamcoder_core/grammar.py` |
| Wake-sleep loop | `src/dreamcoder_core/wake_sleep.py` |
| Recognition network | `src/dreamcoder_core/contrastive_recognition.py` |
| Compression | `src/dreamcoder_core/compression/` |
| Type system | `src/dreamcoder_core/type_system.py` |
| Program representation | `src/dreamcoder_core/program.py` |
| Reference experiment | `src/experiments/run_reference_wakesleep.py` |

### Key Design Decisions
- **Typed DSL:** Programs are type-checked, unlike LOTlib3's untyped grammar
- **Recognition-guided enumeration:** Neural network biases grammar weights per-task
- **Top-down best-first:** Priority queue ordered by cumulative log-probability
- **Cost = negative log-probability** under the PCFG

## Questions to Explore

1. **Architecture:** Should MCMC replace enumeration entirely, or run as a complementary search strategy (e.g., enumeration for shallow programs, MCMC for deep ones)?

2. **Proposal distribution:** How should subtree regeneration work with our typed grammar? LOTlib3 is untyped; we need type-safe proposals.

3. **Recognition integration:** Can the recognition network define the MCMC proposal distribution? (Sample subtrees from recognition-biased grammar rather than the base grammar.)

4. **Convergence diagnostics:** How many MCMC steps are enough? Piantadosi uses 250,000 per concept — is that feasible for our task set?

5. **Library learning interaction:** After MCMC finds solutions, does compression still work the same way? The solutions found by MCMC may have different structural properties than enumeration-found solutions.

6. **Evaluation:** How do we compare MCMC vs enumeration performance? Same task set, same time budget, compare solutions found?
