# LAPS-Style Language Integration for Card Game Rule Learning

## Research Analysis Document
**Date:** December 2024
**Status:** Research exploration phase
**Related:** DreamCoder architecture, self-explanation research

---

## Executive Summary

Integrating natural language descriptions into the DreamCoder-style card game learning system is **theoretically well-motivated and architecturally feasible**, following the LAPS (Language for Abstraction and Program Search) framework by Wong et al. (2021). The key insight is that language provides a soft constraint that sharpens the posterior over programs, reducing enumeration search space while enabling transfer learning across rules with similar descriptions.

**Critical challenges:** The 45-rule dataset is 1-2 orders of magnitude smaller than LAPS/LILO evaluation sets, raising concerns about overfitting and generalization. The good news is that the codebase already has description fields in the `Rule` dataclass—the infrastructure exists, requiring only neural architecture modifications to leverage them.

---

## I. How LAPS Works: Core Principles

### 1.1 The LAPS Framework

LAPS (Wong et al., ICML 2021) extends DreamCoder by incorporating natural language annotations into two key components:

**Joint Objective:**
```
P(p | D, d, G) ∝ P(D | p) · P(d | p) · P(p | G)
```

Where:
- `P(D | p)` = likelihood (program consistent with examples)
- `P(d | p)` = language compatibility (program matches description)
- `P(p | G)` = grammar prior

**Key Mechanism:** Language enters through the recognition network, which learns to predict primitive probabilities conditioned on **both** input-output examples **and** natural language descriptions:

```
Q(π | D, d) = Recognition(Encode(D), Encode(d))
```

### 1.2 How Descriptions Are Obtained

In LAPS, descriptions are **human-provided annotations** during training. The system works even without language at test time because:

1. Language during training improves the **quality of learned abstractions** (more generalizable)
2. Language during training improves the **recognition network** (better primitive predictions)
3. These improvements persist even when test tasks have no descriptions

### 1.3 LILO's AutoDoc Extension

LILO (Grand et al., ICLR 2024) adds **program-to-language generation**:

1. **Stitch compression** identifies optimal λ-abstractions from solved programs
2. **AutoDoc** uses an LLM to generate human-readable names and docstrings for each abstraction
3. **Feedback loop**: Generated documentation helps the LLM synthesizer deploy learned abstractions in future synthesis

---

## II. Current Architecture Analysis

### 2.1 What Already Exists

The `Rule` dataclass in `src/rules/catalogue.py` **already includes descriptions**:

```python
Rule(
    id="Uniform_color",
    token="r3x",
    name="All cards have the same color",
    predicate=uniform_property(get_color),
    family="COUNT",
    description="All cards are either black (♣/♠) or red (♦/♥).",  # ← HERE
    composition=C("uniform", C("get_color")),
    primitives_used=["uniform", "get_color", "unique_count", "eq"],
    level=1
)
```

### 2.2 Current Encoding Pipeline (Without Language)

```
Hand (8 cards) → CardEncoder (GRU) → hand_embedding (hidden_dim)
                           ↓
Example (hand, label) → ExampleEncoder → example_embedding
                           ↓
Task (100 examples) → TaskEncoder (attention) → task_embedding
                           ↓
                  PrimitivePredictor (MLP)
                           ↓
                  log_probs (60 primitives)
```

**What's NOT currently used:**
- `task.name` (metadata only)
- `task.description` (completely ignored by neural model)
- `task.family` (available but unused)

### 2.3 Where Language Would Be Injected

**Recommended integration point** (Late Fusion):

```
[Task Examples]              [Description d]
      ↓                            ↓
 CardEncoder                 LangEncoder
 (existing GRU)            (new: LSTM/Transformer)
      ↓                            ↓
  TaskEncoder               LanguageEmbedding
      ↓                            ↓
      └────────> FusionMLP <───────┘
                     ↓
            PrimitivePredictor
```

---

## III. Theoretical Framework

### 3.1 What Language Provides (That Examples Don't)

**Proposition 1: Disambiguation**
When multiple programs are consistent with examples, language selects the intended one.

**Proposition 2: Compositional Hints**
Phrases directly suggest primitives:
- "first half" → `first_half`, `take`, `split`
- "same color" → `all_same_color`, `get_color`, `eq`

**Proposition 3: Transfer Cues**
Similar descriptions enable generalization across rules.

### 3.2 Description-Primitive Correspondence

| Description Phrase | Strongly Associated Primitives |
|--------------------|-------------------------------|
| "same" | `eq`, `all_same_suit`, `all_same_color` |
| "pair" | `has_pair_ranks`, `count`, `eq` |
| "first/last" | `head`, `last` |
| "half/halves" | `first_half`, `second_half` |
| "sorted/ascending" | `is_sorted`, `lt` |
| "count/number" | `length`, `count_suit`, `n_unique` |

---

## IV. Critical Challenges

### 4.1 CRITICAL: Dataset Size

**Issue:** Only 45 core rules (89 with pretraining). LAPS/LILO used hundreds to thousands.

**Impact:** High overfitting risk.

**Mitigations:**
1. Use frozen pretrained text encoder (BERT/RoBERTa)
2. Data augmentation via paraphrasing
3. Cross-validation to detect overfitting

### 4.2 HIGH: Semantic Redundancy

**Issue:** With 100 examples per task, the visual pattern may already be learnable.

**Required Control:**
- Compare: (a) examples only, (b) shuffled descriptions, (c) correct descriptions

### 4.3 HIGH: Description-Primitive Alignment

**Issue:** Descriptions are high-level ("Has a pair") but primitives are operational (`count`, `eq`).

### 4.4 MEDIUM: LILO's AutoDoc Not Directly Applicable

**Issue:** LILO uses GPT-4 to generate descriptions. This system uses pure enumeration.

---

## V. Implementation Roadmap

### Phase 1: Foundation (Low Risk)
- Verify descriptions exist for all 45 rules ✓
- Create semantic concept vocabulary (~40-50 concepts)
- Annotate rules with concepts

### Phase 2: Simple Language Integration
- Add LanguageEncoder (GRU-based)
- Modify `encode_task` in `neural_recognition.py`
- Run controlled experiments

### Phase 3: Full LAPS Integration
- Use pretrained text encoder (frozen BERT + adapter)
- Implement cross-attention fusion
- Add language to dreaming phase

### Phase 4: AutoDoc Exploration (Optional)
- Integrate LLM API for description generation
- Generate descriptions for learned abstractions

---

## VI. Key Sources

- [LAPS Paper (Wong et al., ICML 2021)](https://arxiv.org/abs/2106.11053)
- [LILO Paper (Grand et al., ICLR 2024)](https://arxiv.org/abs/2310.19791)
- [DreamCoder (Ellis et al., 2021)](https://dl.acm.org/doi/10.1145/3453483.3454080)
- [Stitch Compression (Bowers et al., POPL 2023)](https://arxiv.org/abs/2211.16605)
- [Wong PhD Thesis "From Words to Worlds" (2024)](https://dspace.mit.edu/handle/1721.1/157326)
- [Chi et al. (1994) Self-explanation effect](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1803_3)

---

## VII. Connection to Self-Explanation

| Self-Explanation Mechanism | DreamCoder Analog | Language Enhancement |
|---------------------------|-------------------|---------------------|
| Verbalizing to learn | Sleep phase training | Train on description-primitive pairs |
| Generating explanations | Dreaming | Generate descriptions for sampled programs |
| Filling knowledge gaps | Library learning | Language-guided abstraction selection |
| Transfer via verbal analogy | Recognition model | Similar descriptions → similar primitives |

---

*Document generated as part of /research-ultrathink analysis*
