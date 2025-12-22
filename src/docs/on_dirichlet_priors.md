# On Dirichlet Priors in Program Synthesis

This document explains the role of Dirichlet priors in DreamCoder-style program synthesis, with particular attention to how they affect learning dynamics and the types of abstractions discovered.

## Table of Contents

1. [What is a Dirichlet Prior?](#what-is-a-dirichlet-prior)
2. [Where Dirichlet Priors Appear in DreamCoder](#where-dirichlet-priors-appear)
3. [The α Parameter and Its Effects](#the-α-parameter)
4. [Concrete Examples](#concrete-examples)
5. [Implications for Self-Explanation Research](#implications-for-self-explanation)
6. [Implementation Details](#implementation-details)

---

## What is a Dirichlet Prior?

The **Dirichlet distribution** is a probability distribution *over* probability distributions. It's the conjugate prior for the categorical (multinomial) distribution.

### Intuition

Imagine you have K primitives and need to assign probabilities to each. The Dirichlet prior encodes your *belief* about what these probabilities should look like *before* seeing any data.

```
θ = (θ₁, θ₂, ..., θₖ)  where  Σθᵢ = 1

θ ~ Dirichlet(α₁, α₂, ..., αₖ)
```

### Symmetric Dirichlet

When all αᵢ = α (same value), we have a **symmetric Dirichlet**:

```
θ ~ Dirichlet(α, α, ..., α)
```

The single parameter α controls the "shape" of the distribution over distributions.

---

## The α Parameter

The α parameter dramatically affects the expected probability distribution:

| α value | Name | Distribution Shape | Effect |
|---------|------|-------------------|--------|
| α → 0 | Sparse | Peaked on few primitives | Winner-take-all |
| α = 0.1 | Sparse | Mostly concentrated | Strong specialization |
| α = 0.5 | Moderate | Somewhat spread | Balanced |
| α = 1 | Uniform (Laplace) | All distributions equally likely | Maximum entropy |
| α > 1 | Dense | Spread across all | Anti-specialization |

### Visual Intuition (for K=3 primitives)

```
α = 0.1 (Sparse)              α = 1 (Uniform)              α = 10 (Dense)
     *                              *                            *
    /|\                           / | \                        /   \
   / | \                         /  |  \                      /     \
  *  *  *                       *   *   *                    *-------*

Probability mass              All points equally           Probability mass
at corners (one               likely in simplex            near center
primitive dominates)                                       (all primitives
                                                           roughly equal)
```

---

## Where Dirichlet Priors Appear in DreamCoder

### 1. Grammar Weight Updates (Inside-Outside Algorithm)

After solving tasks, we update primitive probabilities based on usage counts.

**Bayesian Update:**
```
Prior:      P(θ) = Dirichlet(α, α, ..., α)
Likelihood: P(data | θ) = ∏ θᵢ^(countᵢ)
Posterior:  P(θ | data) = Dirichlet(α + count₁, α + count₂, ...)
```

**Point Estimate (Mean of Posterior):**
```
P(primitiveᵢ) = (countᵢ + α) / (Σcountⱼ + K·α)
```

This is implemented in `grammar.py` → `inside_outside_update()`:
```python
def inside_outside_update(self, frontiers, alpha=1.0):
    counts = defaultdict(lambda: alpha)  # pseudo-counts = α
    # ... count primitive uses ...
    P_i = count_i / total
```

### 2. Grammar Structure Prior (MDL/Compression)

When deciding whether to add a new abstraction, the prior affects the "cost" of grammar complexity:

```
Total Description Length = DL(grammar) + Σ DL(programᵢ | grammar)
```

The DL(grammar) term encodes a prior that:
- Penalizes having too many primitives
- Penalizes complex primitive definitions

### 3. Recognition Model Regularization

The neural recognition model can have priors/regularizers on its output distribution, though this is typically handled through standard neural network techniques (dropout, weight decay) rather than explicit Dirichlet priors.

---

## Concrete Examples

### Example 1: Effect on Primitive Probabilities

After solving 5 card game tasks, primitive usage counts are:

```
has_suit:       4 uses
n_unique_ranks: 3 uses
all_same_color: 2 uses
gt:             1 use
+, *, map, filter, fold, not: 0 uses each
────────────────────────────
Total: 10 uses across 10 primitives
```

**With α = 0.1 (Sparse):**
```
P(has_suit) = (4 + 0.1) / (10 + 10×0.1) = 4.1/11 = 0.373  (37.3%)
P(+)        = (0 + 0.1) / (10 + 10×0.1) = 0.1/11 = 0.009  (0.9%)

Ratio: 41:1
```

**With α = 1.0 (Laplace):**
```
P(has_suit) = (4 + 1) / (10 + 10×1) = 5/20 = 0.250  (25.0%)
P(+)        = (0 + 1) / (10 + 10×1) = 1/20 = 0.050  (5.0%)

Ratio: 5:1
```

**Interpretation:**
- Sparse (α=0.1): Unused primitives become nearly inaccessible (0.9%)
- Laplace (α=1): Unused primitives retain meaningful probability (5%)

### Example 2: Effect on Abstraction Learning

Consider two candidate abstractions:

| Abstraction | Description | Used in |
|-------------|-------------|---------|
| `#all_satisfy` | General "all cards satisfy P" | 8/10 tasks |
| `#has_ace` | Specific "hand contains ace" | 2/10 tasks |

**With α = 0.1 (Sparse):**
```
After iteration 1:
  P(#all_satisfy) ≈ 0.40
  P(#has_ace)     ≈ 0.05

After iteration 2 (assuming similar task distribution):
  P(#all_satisfy) ≈ 0.55  (used more → higher prob → used even more)
  P(#has_ace)     ≈ 0.02  (used less → lower prob → used even less)

After iteration 3:
  P(#all_satisfy) ≈ 0.65
  P(#has_ace)     ≈ 0.008  (effectively dead)
```

**Rich-get-richer dynamics:** General abstractions dominate, specialized ones wither.

**With α = 1.0 (Laplace):**
```
After iteration 1:
  P(#all_satisfy) ≈ 0.30
  P(#has_ace)     ≈ 0.10

After iteration 2:
  P(#all_satisfy) ≈ 0.35
  P(#has_ace)     ≈ 0.08

After iteration 3:
  P(#all_satisfy) ≈ 0.38
  P(#has_ace)     ≈ 0.06  (still usable)
```

**More stable dynamics:** Both abstractions remain viable.

### Example 3: Effect on Task Adaptation

**Scenario:** Tasks 1-10 are about suits/colors. Task 11 requires arithmetic.

**With α = 0.1:**
- Arithmetic primitives (+, -, *, /) have probability ≈ 0.01 each
- Task 11 requires exploring ~100 programs before trying arithmetic
- May timeout before finding solution

**With α = 1.0:**
- Arithmetic primitives have probability ≈ 0.05 each
- Task 11 explores arithmetic within first ~20 programs
- More likely to find solution

---

## Implications for Self-Explanation Research

The Dirichlet prior has interesting connections to theories of learning and explanation:

### 1. Simplicity Bias

Sparse priors (low α) create a strong **simplicity bias**:
- Prefer explanations using few distinct concepts
- Converge to minimal vocabulary
- Related to Occam's Razor / Minimum Description Length

### 2. Entrenchment Effects

Low α creates **entrenchment**:
- Early-learned concepts become entrenched
- Later concepts struggle to gain foothold
- Similar to "functional fixedness" in human cognition

### 3. Transfer vs. Specialization Trade-off

| α value | Learning Style | Analogy |
|---------|---------------|---------|
| Low α | Specialist | Expert in narrow domain |
| High α | Generalist | Flexible across domains |

### 4. Curriculum Effects

With low α, **curriculum order matters**:
- Early tasks shape the grammar strongly
- Later tasks must work with established vocabulary
- Changing task order could yield different final grammar

### 5. Explanation Granularity

The α parameter affects **explanation granularity**:
- Low α → Explanations use few, general concepts
- High α → Explanations can use many, specific concepts

This connects to questions about what makes a "good" self-explanation:
- Should explanations be parsimonious (few concepts)?
- Or should they be precise (specific concepts for each case)?

---

## Implementation Details

### Our Implementation

```python
# In grammar.py

def inside_outside_update(
    self,
    frontiers: List[List[Tuple[Program, float]]],
    alpha: float = 1.0  # Dirichlet concentration parameter
) -> 'Grammar':
    """
    Update grammar weights with Dirichlet prior.

    Args:
        frontiers: Solutions from each task
        alpha: Dirichlet concentration (default 1.0 = Laplace smoothing)
               - α < 1: Sparse (prefer few high-probability primitives)
               - α = 1: Uniform (Laplace smoothing)
               - α > 1: Dense (prefer spread-out probabilities)
    """
    counts = defaultdict(lambda: alpha)  # Initialize with pseudo-counts
    # ... count primitive uses in solutions ...

    total = sum(counts.values())
    for primitive in primitives:
        P[primitive] = counts[primitive] / total
```

### Original DreamCoder (Ellis et al.)

Uses α = 1 (Laplace smoothing) with type-indexed normalization:
- Probabilities normalized separately per return type
- INT-returning primitives compete only with each other
- BOOL-returning primitives compete only with each other

---

## Summary

| Aspect | Sparse (α < 1) | Laplace (α = 1) | Dense (α > 1) |
|--------|----------------|-----------------|---------------|
| **Unused primitives** | Near-zero probability | Retain 1/(K+N) | Stay high |
| **Learning dynamics** | Rich-get-richer | Stable | Averaging |
| **Abstractions** | Few, general | Mixed | Many, specific |
| **Exploration** | Low | Moderate | High |
| **Task assumption** | Homogeneous | Moderate diversity | Heterogeneous |
| **Curriculum sensitivity** | High | Moderate | Low |

For program synthesis with diverse tasks, α = 1 (Laplace) is recommended as a reasonable default that balances specialization and exploration.

---

## References

- Ellis, K., et al. (2021). DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. PLDI.
- Gelman, A., et al. (2013). Bayesian Data Analysis, 3rd ed. Chapter 5 (Hierarchical models and Dirichlet priors).
