# Deep Dive: Language-Program Integration for Card Game Synthesis

## Document Overview

This document provides detailed answers to four key questions about extending the DreamCoder-style card game system with LAPS-style language integration:

1. **How language descriptions algorithmically contribute to inference**
2. **How language descriptions are generated**
3. **Hand-level vs. task-level descriptions**
4. **The translation function between programs and language**

---

# Part I: How Language Contributes to Inference

## The Algorithm: Step-by-Step

### 1.1 Baseline DreamCoder Pipeline (Without Language)

```python
def synthesize_without_language(task, grammar, timeout):
    # Step 1: Encode examples only
    task_embedding = task_encoder(task.examples)

    # Step 2: Predict primitive weights from examples
    weights = primitive_predictor(task_embedding)

    # Step 3: Apply weights to grammar
    weighted_grammar = apply_weights_to_grammar(grammar, weights)

    # Step 4: Enumerate and check programs
    for program, log_prob in enumerate_programs(weighted_grammar, task.type):
        if task.check(program):  # Verify on all examples
            return program
    return None
```

### 1.2 LAPS Pipeline (With Language)

```python
def synthesize_with_language(task, description, grammar, timeout):
    # Step 1: Encode examples (same as before)
    task_embedding = task_encoder(task.examples)

    # Step 2: Encode language description (NEW)
    language_embedding = language_encoder(description)

    # Step 3: Fuse embeddings (NEW)
    fused_embedding = fusion_module(task_embedding, language_embedding)

    # Step 4: Predict primitive weights from fused embedding
    weights = primitive_predictor(fused_embedding)

    # Step 5: Apply weights to grammar
    weighted_grammar = apply_weights_to_grammar(grammar, weights)

    # Step 6: Enumerate and check programs
    for program, log_prob in enumerate_programs(weighted_grammar, task.type):
        if task.check(program):
            return program
    return None
```

### 1.3 The Key Mathematical Operation

**Grammar Reweighting:**

```python
def apply_weights_to_grammar(base_grammar, predicted_weights, alpha=0.5):
    """
    For each primitive:
    new_log_prob = α * base_log_prob + (1-α) * predicted_log_prob
    """
    new_productions = []
    for production in base_grammar.productions:
        primitive_name = production.primitive.name
        base_log_prob = production.log_probability

        if primitive_name in predicted_weights:
            predicted_log_prob = predicted_weights[primitive_name]
            new_log_prob = (alpha * base_log_prob +
                           (1 - alpha) * predicted_log_prob)
        else:
            new_log_prob = base_log_prob

        new_productions.append(Production(production.primitive, new_log_prob))

    return Grammar(new_productions)
```

### 1.4 Fusion Mechanisms

**Option 1: Concatenation Fusion**
```python
f = W_proj · [task_emb; lang_emb] + b
```
- **Pros:** Simple, fast
- **Cons:** No dynamic weighting

**Option 2: Gated Fusion**
```python
g = σ(W_g · [task_emb; lang_emb] + b_g)
f = g ⊙ task_emb + (1-g) ⊙ lang_emb
```
- **Pros:** Adaptive weighting
- **Cons:** More parameters

**Option 3: Cross-Attention Fusion**
```python
t' = CrossAttention(Q=task_emb, KV=lang_emb)
l' = CrossAttention(Q=lang_emb, KV=task_emb)
f = LayerNorm(t' + l')
```
- **Pros:** Most expressive
- **Cons:** Most parameters, may overfit

### 1.5 Worked Example: "All cards same color"

**Without Language:**
```
Primitives by predicted weight:
1. all_same_suit      (log-prob: -1.2)  ← WRONG but high
2. all_same_color     (log-prob: -1.4)  ← CORRECT but second
3. get_suit           (log-prob: -1.8)
...

Enumeration order: all_same_suit first (might pass some examples)
```

**With Language ("all cards same color"):**
```
Primitives by predicted weight:
1. all_same_color     (log-prob: -0.5)  ← CORRECT and first
2. get_color          (log-prob: -1.0)
3. eq_color           (log-prob: -1.2)
4. all_same_suit      (log-prob: -2.5)  ← Pushed down

Enumeration order: all_same_color first (correct immediately)
```

**Search savings:** For complex programs, the speedup compounds multiplicatively across composed primitives.

---

# Part II: How Language Descriptions Are Generated

## 2.1 When Descriptions Are NOT Provided as Input

### The Three Approaches

**Approach A: Template-Based Generation**
```python
TEMPLATES = {
    "all_same_X": "All cards have the same {X}",
    "and(A, B)": "{A} and {B}",
    "or(A, B)": "{A} or {B}",
}

def describe_template(program):
    if matches("all_same_color", program):
        return "All cards have the same color"
    elif matches("and(A, B)", program):
        return f"{describe_template(A)} and {describe_template(B)}"
```
- **Pros:** Precise, compositional, no training
- **Cons:** Rigid, can't handle novel abstractions

**Approach B: Neural Sequence-to-Sequence**
```python
def describe_neural(program):
    program_tokens = tokenize_program(program)
    return seq2seq_model.generate(program_tokens)
```
- **Pros:** Learns from data, handles variations
- **Cons:** Requires training data, may hallucinate

**Approach C: LLM-Based (LILO's AutoDoc)**
```python
def describe_llm(program, examples):
    prompt = f"""
    Given this program: {program}
    And these examples: {examples}
    Write a concise description:
    """
    return llm.complete(prompt)
```
- **Pros:** Zero-shot, handles novel cases
- **Cons:** API cost, non-deterministic

### 2.2 The Bootstrap Problem

**The Dilemma:**
- To learn programs → need descriptions to guide search
- To generate descriptions → need programs to describe
- To learn mapping → need paired examples

**Solution: Hybrid Bootstrap**

```
Phase 1: Start with templates for base primitives
         all_same_suit → "all cards same suit"

Phase 2: Use LLM for compositions with few-shot examples
         New composition → LLM generates description

Phase 3: Fine-tune on accumulated pairs
         As library grows, train specialized model
```

### 2.3 Dreaming with Descriptions (LILO Approach)

In the dream phase:
1. **Sample fantasy program** from grammar
2. **Generate description** using AutoDoc
3. **Train recognition** on (description, examples) → program

```python
def dream_with_language(grammar, description_model, n_dreams):
    dreams = []
    for _ in range(n_dreams):
        # Sample fantasy program
        program = sample_from_grammar(grammar)

        # Execute to get examples
        examples = execute_on_random_inputs(program)

        # Generate description
        description = description_model.describe(program, examples)

        # Create training example
        dreams.append((description, examples, program))

    return dreams
```

**Why This Helps:**
- Infinite training data (fantasy programs)
- Language-program associations learned
- Compositional generalization

**Risks:**
- Bad descriptions → wrong associations
- Mitigations: Verification, confidence filtering, consistency checking

---

# Part III: Hand-Level vs. Task-Level Descriptions

## 3.1 The Two Levels

### Task-Level Descriptions (Current Approach)

**What it looks like:**
```python
Rule(
    id="Uniform_color",
    description="All cards are either black or red.",  # ONE description
    ...
)
```

**Properties:**
- Abstract concept description
- Assumes rule is known
- One annotation per task (45 total)
- Directly names primitives

### Hand-Level Descriptions (Alternative)

**What it looks like:**
```
Hand: [2♥, 5♥, 8♦, K♥, A♥], Label: True
Description: "5 cards, 4 hearts 1 diamond, all red, no pairs, not sorted"

Hand: [2♠, 5♣, 8♠, K♣, A♠], Label: True
Description: "5 cards, 3 spades 2 clubs, all black, no pairs, not sorted"

Hand: [2♥, 5♠, 8♦, K♣, A♥], Label: False
Description: "5 cards, mixed suits and colors, no pairs"
```

**Properties:**
- Concrete observable features
- Available before rule is known
- 100 annotations per task (4,500 total if manual)
- Must INDUCE relevance

## 3.2 Comparison

| Aspect | Task-Level | Hand-Level |
|--------|------------|------------|
| **Information type** | Abstract rule | Observable features |
| **When available** | After rule known | Before rule known |
| **Annotation effort** | Low (45) | High (4,500) if manual |
| **Cognitive realism** | Low | High |
| **Utility for search** | Direct shortcut | Must learn to induce |

## 3.3 The Recommended Hybrid Approach

**Architecture:**
```
Hand-level processing:
  For each hand h_i:
    auto_features_i = GenerateHandFeatures(h_i)  # AUTOMATED
    hand_emb_i = Encode(cards_i, auto_features_i, label_i)

Aggregation (learn what's discriminative):
  positive_pattern = Aggregate([h for h if label = True])
  negative_pattern = Aggregate([h for h if label = False])
  induced_concept = Contrast(positive_pattern, negative_pattern)

Task-level integration (optional):
  task_desc = "All cards have the same color"  # MANUAL (existing)
  task_emb = Encode(task_desc)
  final_repr = Combine(induced_concept, task_emb)

Primitive prediction:
  primitive_probs = Predict(final_repr)
```

**Why This is Best:**
1. **Automated hand features** → Zero annotation cost for 4,500 hands
2. **Existing task descriptions** → Already have 45 manual descriptions
3. **Curriculum learning** → Can train hand-only first, add task-level later
4. **Ablation-friendly** → Can compare hand-only vs. hybrid

### 3.4 Automated Hand Feature Generation

```python
def generate_hand_description(hand: List[Card]) -> str:
    features = []

    # Basic counts
    features.append(f"{len(hand)} cards")

    # Suit distribution
    suit_counts = Counter(c.suit for c in hand)
    for suit, count in suit_counts.items():
        features.append(f"{count} {suit}")

    # Color distribution
    red_count = sum(1 for c in hand if c.suit in ['hearts', 'diamonds'])
    if red_count == len(hand):
        features.append("all red")
    elif red_count == 0:
        features.append("all black")
    else:
        features.append(f"{red_count} red, {len(hand)-red_count} black")

    # Rank features
    if len(set(c.rank for c in hand)) == 1:
        features.append("all same rank")
    if is_sorted([c.rank for c in hand]):
        features.append("sorted by rank")
    if has_pair(hand):
        features.append("has pair")

    return ", ".join(features)
```

---

# Part IV: The Program-Language Translation Function

## 4.1 Formal Definitions

### The Spaces

```
P = {p : well-typed programs in DSL}
L = {ℓ : natural language description strings}
D = {(h₁, b₁), ..., (hₙ, bₙ) : input-output examples}
```

### The Core Functions

**Describe: P → L**
```
Maps program to its canonical description.
Example: Describe(λh. all_same_suit h) = "All cards have the same suit"
```

**Meaning: L → 2^P**
```
Maps description to SET of consistent programs (many-to-many).
Example: Meaning("has a pair") = {has_pair_ranks, >= max_count_rank 2, ...}
```

**Parse: L × D → P**
```
Given description + examples, find the program.
Parse(ℓ, D) = argmax_p P(p | ℓ, D)
            = argmax_p P(D|p) · P(p|ℓ) · P(p|Γ)
```

## 4.2 Why Translation is Hard

1. **Many-to-many:** One program has many descriptions; one description fits many programs
2. **Compositionality gap:** Linguistic structure ≠ program structure
3. **Granularity mismatch:** "has a pair" vs. detailed implementation
4. **Novel concepts:** Learned abstractions have no standard name

## 4.3 The Describe Function: Three Approaches

### Template-Based
```python
def describe_template(program):
    for template in TEMPLATES:
        if matches(template.pattern, program):
            return template.render(program)
    return None
```

### Neural Seq2Seq
```python
def describe_neural(program):
    return seq2seq_model.generate(tokenize(program))
```

### LLM-Based (AutoDoc)
```python
def describe_llm(program, examples):
    return llm.complete(f"Describe this program: {program}\nExamples: {examples}")
```

### Recommended: Hybrid
```python
def describe_hybrid(program, examples=None):
    # Try template first (fast, precise)
    template_desc = describe_template(program)
    if template_desc:
        return template_desc

    # Try neural (learned patterns)
    neural_desc = describe_neural(program)
    if validate(neural_desc, program):
        return neural_desc

    # Fall back to LLM
    return describe_llm(program, examples)
```

## 4.4 The Parse Function: Full Synthesis Loop

Parse is what the synthesis system computes:

```python
def parse(description, examples, grammar, recognition_net):
    # Encode description → embedding
    desc_emb = encode_description(description)

    # Encode examples → embedding
    examples_emb = encode_examples(examples)

    # Combine → task representation
    task_repr = combine(desc_emb, examples_emb)

    # Recognition network → primitive weights
    primitive_scores = recognition_net(task_repr)

    # Reweight grammar
    biased_grammar = grammar.reweight(primitive_scores)

    # Enumerate until solution found
    for program in enumerate(biased_grammar):
        if all(program(h) == b for h, b in examples):
            return program

    return None
```

## 4.5 Example Translations

| Program | Description |
|---------|-------------|
| `λh. all_same_suit h` | All cards have the same suit |
| `λh. all_same_color h` | All cards are the same color |
| `λh. has_pair_ranks h` | The hand contains at least one pair |
| `λh. eq (n_unique_suits h) 2` | Exactly two different suits appear |
| `λh. is_sorted h` | Cards are sorted by rank |
| `λh. and (has_pair h) (is_sorted h)` | Has a pair and is sorted |

## 4.6 Compositional Translation

For compositional programs:

```
Program: λh. and (uniform_suit (first_half h)) (is_sorted (second_half h))

Parse tree:
                and
               /   \
      uniform_suit  is_sorted
           |            |
      first_half   second_half
           |            |
           h            h

Compositional description:
  describe(first_half h) = "the first half of the hand"
  describe(uniform_suit X) = X + " has uniform suit"
  describe(and A B) = A + " and " + B

Result: "The first half of the hand has uniform suit
         and the second half of the hand is sorted"
```

---

# Summary: Key Insights

## How Language Helps Search

1. **Disambiguation:** Language resolves ambiguity when examples underdetermine the program
2. **Prior sharpening:** Language concentrates probability on semantically appropriate programs
3. **Compositional hints:** Phrases like "first half" directly suggest primitives
4. **Transfer:** Similar descriptions enable generalization across rules

## Description Generation

- **Templates** for known patterns (precise but rigid)
- **Neural models** for learned variations (flexible but needs data)
- **LLMs** for novel cases (powerful but expensive)
- **Hybrid** approach recommended: templates → neural → LLM fallback

## Hand vs. Task Level

- **Task-level** provides direct semantic shortcut (less cognitively realistic)
- **Hand-level** mirrors human feature noticing (more realistic but expensive)
- **Hybrid recommended:** Automated hand features + existing task descriptions

## Translation Function

- **Bidirectional:** Programs ↔ Language
- **Many-to-many:** Not a simple function
- **Parse = full synthesis:** Language enters as prior, not deterministic parse
- **Compositionality** preserved through recursive description generation

---

*Generated as part of /research-ultrathink deep dive*
*December 2024*
