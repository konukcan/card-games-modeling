## Integration with Ellis et al.'s DreamCoder

This document explains how our card game modeling system relates to the original DreamCoder implementation and provides guidance for full integration.

## Overview

**DreamCoder** (Ellis et al., 2023) is a program synthesis system that learns through:
1. **Recognition**: Neural network guides program search
2. **Abstraction**: Library learning through compression
3. **Wake-Sleep**: Iterative refinement

Our implementation adapts this framework to the card game domain.

## Architecture Comparison

### Ellis et al.'s DreamCoder

```
Domain-Specific Language (DSL)
    ↓
Program Enumeration (Type-directed search)
    ↓                             ↑
Recognition Network          (guides search)
    ↓
Solved Programs → Library Learning (Compression)
    ↓
Wake: Train on new tasks
Sleep: Generate synthetic tasks → Retrain recognition
```

### Our Card Game Adaptation

```
Card Game DSL (5-level compositional grammar)
    ↓
Enumeration (Best-first with neural scores)
    ↓                             ↑
Recognition Network          (predicts primitives)
    ↓
Solved Rules → Extract shared subprograms
    ↓
Fit to human behavioral data
```

## Key Differences

| Aspect | Original DreamCoder | Our Adaptation |
|--------|---------------------|----------------|
| **Domain** | Lists, graphics, text, regex | Card game predicates |
| **DSL** | Minimal primitives + learned library | 5-level pre-defined grammar |
| **Search** | Type-directed + sketch-and-fill | Best-first enumeration |
| **Recognition** | Predicts full program sketch | Predicts primitive usage |
| **Library** | Induced from scratch | Based on compositional analysis |
| **Validation** | Held-out tasks | Human behavioral data |

## Implementation Mapping

### 1. DSL Definition

**DreamCoder (Haskell)**:
```haskell
data Type = TInt | TList Type | TArrow Type Type

primitives :: [Primitive]
primitives = [
  Primitive "map" (TArrow (TArrow t1 t2) (TArrow (TList t1) (TList t2))),
  Primitive "filter" (TArrow (TArrow t1 TBool) (TArrow (TList t1) (TList t1))),
  ...
]
```

**Our Implementation (Python)**:
```python
# src/rules/primitives.py

# Level 0: Atomic primitives
def get_suit(card: Card) -> Suit: ...
def get_rank(card: Card) -> Rank: ...

# Level 1: Combinators
def map_property(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], List[Any]]: ...
def filter_cards(pred: Callable[[Card], bool]) -> Callable[[Hand], Hand]: ...

# Level 2-4: Higher-order abstractions
def halves_equal(prop_fn: Callable[[Hand], Any]) -> Callable[[Hand], bool]: ...
```

**Mapping**: Our grammar explicitly defines all levels. To integrate with DreamCoder, export as typed lambda terms.

### 2. Recognition Network

**DreamCoder (PyTorch)**:
```python
class RecognitionModel(nn.Module):
    def __init__(self, tasks, primitives):
        self.encoder = TaskEncoder(...)
        self.decoder = nn.Linear(hidden_dim, len(primitives))

    def forward(self, task_examples):
        embedding = self.encoder(task_examples)
        return torch.sigmoid(self.decoder(embedding))
```

**Our Implementation**:
```python
# See dreamcoder_modeling/dreamcoder_demo.py (already implemented!)

class RecognitionNetwork(nn.Module):
    def __init__(self, example_feature_dim=104, num_primitives=30):
        self.example_encoder = ExampleEncoder(104, 64)
        self.set_aggregator = SetAggregator(64)
        self.primitive_predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_primitives)
        )
```

**Status**: ✅ Already implemented in `dreamcoder_modeling/dreamcoder_demo.py` (94.75% accuracy)

**Integration**: Replace our feature extractor with Ellis's task encoder for compatibility.

### 3. Program Enumeration

**DreamCoder (OCaml/Haskell)**:
```ocaml
let enumerate grammar request =
  let rec search depth =
    if depth = 0 then base_cases
    else
      let sub_programs = search (depth - 1) in
      combine_with_grammar grammar sub_programs
  in
  search max_depth |> sort_by_likelihood
```

**Our Implementation** (to be added):
```python
# src/dreamcoder/enumeration.py

def enumerate_programs(dsl: DSL, max_depth: int, neural_scores: Dict):
    """
    Enumerate programs up to max_depth, prioritized by:
      priority = log P(program | DSL) + log P_neural(primitives | task)
    """
    frontier = PriorityQueue()
    frontier.put((0, EmptyProgram))

    while not frontier.empty():
        score, program = frontier.get()
        if is_complete(program):
            yield program
        else:
            for expansion in dsl.expand(program):
                new_score = score + neural_scores.get(expansion, 0)
                frontier.put((new_score, expansion))
```

**Integration**: Use DreamCoder's type-directed search with our domain-specific DSL.

### 4. Library Learning

**DreamCoder**:
```python
def compress(programs):
    """
    Find frequently-used subprograms and abstract them.
    Uses anti-unification + MDL principle.
    """
    fragments = extract_fragments(programs)
    abstractions = []
    for fragment in fragments:
        if usage_count(fragment) >= 3:
            abstractions.append(abstract(fragment))
    return abstractions
```

**Our Approach**:
```python
# src/dreamcoder/compression.py

def learn_library(solved_tasks):
    """
    Extract shared subprograms from solved rules.
    Based on primitives_used analysis from catalogue.
    """
    # Already have compositional analysis!
    # Halves: used in 17 rules
    # hasAP: used in 9 rules
    # seqPalindrome: used in 5 rules
    ...
```

**Integration**: Our grammar already encodes the "learned library" that DreamCoder would induce. Can validate by running DreamCoder's compression on our 45 rules and comparing.

## Full Integration Steps

### Option A: Standalone (Current Approach)

1. ✅ Define card game DSL (DONE)
2. ✅ Implement recognition network (DONE in `dreamcoder_modeling/`)
3. ⏳ Implement enumeration with neural guidance
4. ⏳ Extract library from solved rules
5. ⏳ Fit to human data

**Pros**: Clean separation, domain-specific optimizations
**Cons**: Not leveraging DreamCoder's full infrastructure

### Option B: Full Integration with Ellis's Codebase

1. Clone DreamCoder repository:
   ```bash
   git clone https://github.com/ellisk42/ec.git
   cd ec
   ```

2. Add card game domain:
   ```bash
   mkdir domains/cards
   cp our_primitives.ml domains/cards/
   ```

3. Define DSL in DreamCoder format:
   ```ocaml
   (* domains/cards/primitives.ml *)
   let card_primitives = [
     primitive "getSuit" (tcard @> tsuit);
     primitive "getRank" (tcard @> trank);
     primitive "halves" (thand @> tpair thand thand);
     ...
   ]
   ```

4. Implement task generator:
   ```python
   # dreamcoder/domains/cards/tasks.py
   def make_card_tasks():
       return [Task(f"rule_{i}", examples) for i in range(45)]
   ```

5. Run DreamCoder:
   ```bash
   python bin/cards.py
   ```

6. Compare induced library with our analysis

**Pros**: Leverages full DreamCoder infrastructure
**Cons**: Requires significant OCaml/Haskell work

### Option C: Hybrid (Recommended)

1. Keep our Python implementation for modeling
2. Export our DSL to DreamCoder format
3. Run DreamCoder as validation/comparison
4. Use DreamCoder's library learning to refine our grammar

**Steps**:
```python
# tools/export_to_dreamcoder.py

def export_dsl_to_dreamcoder_format():
    """
    Convert our Python primitives to DreamCoder's typed lambda calculus.
    """
    dsl_export = []
    for primitive in LEVEL_0_PRIMITIVES:
        dsl_export.append({
            'name': primitive.__name__,
            'type': infer_type(primitive),
            'implementation': serialize_python_fn(primitive)
        })
    save_json(dsl_export, 'dreamcoder_dsl.json')
```

## Type System Translation

### Our Types → DreamCoder Types

| Our Type | DreamCoder Type |
|----------|-----------------|
| `Card` | `tcard` (base type) |
| `Hand` | `tlist tcard` |
| `Suit` | `tsuit` (enum) |
| `Rank` | `trank` (enum) |
| `Card → Suit` | `tarrow tcard tsuit` |
| `Hand → bool` | `tarrow (tlist tcard) tbool` |

### Example Translation

**Our primitive**:
```python
def get_suit(card: Card) -> Suit:
    return card.suit
```

**DreamCoder primitive**:
```ocaml
primitive "getSuit" (tcard @> tsuit)
  ~documentation:"Extract suit from card"
  (fun card -> card.suit)
```

## Recognition Network Integration

Our recognition network (from `dreamcoder_modeling/dreamcoder_demo.py`) is **already compatible** with DreamCoder's architecture. To integrate:

1. Replace our feature extractor with DreamCoder's:
   ```python
   # Instead of our 104-dim features
   features = dreamcoder.featureExtractor.featuresOfTask(task)
   ```

2. Use DreamCoder's training loop:
   ```python
   from dreamcoder.recognition import RecognitionModel

   model = RecognitionModel(tasks, our_primitives)
   model.train(num_epochs=50)
   ```

3. Our network predicts primitive usage → DreamCoder predicts program sketches. This is a design choice, not a limitation.

## Validation Plan

To validate that our analysis matches DreamCoder's induced library:

1. **Run DreamCoder on our 45 rules** (fresh, no prior library)
2. **Compare induced abstractions** with our Level 2-4 primitives
3. **Hypothesis**: DreamCoder will discover `halves`, `hasAP`, `seqPalindrome`, etc.
4. **Metric**: Compression ratio and library overlap

Expected result:
```
Our grammar:        halves (17 uses), hasAP (9 uses), seqPalindrome (5 uses)
DreamCoder induced: halves (17 uses), hasAP (9 uses), seqPalindrome (5 uses)
                    ↑ Should match!
```

## References

### Original DreamCoder

- **Paper**: Ellis et al. (2023), *Philosophical Transactions of the Royal Society A*
- **Code**: https://github.com/ellisk42/ec
- **Docs**: https://ellisk42.github.io/ec/

### Our Implementation

- **Primitives**: `src/rules/primitives.py` (Python)
- **Catalogue**: `src/rules/catalogue.py` (45 core rules)
- **Grammar Analysis**: `compositional_grammar_analysis/compositional_rule_grammar.tex`
- **Recognition Network**: `dreamcoder_modeling/dreamcoder_demo.py` (working!)

## Next Steps

1. **Short-term** (this week):
   - Complete enumeration module
   - Test on 10 demo rules
   - Validate against human data (if available)

2. **Medium-term** (next month):
   - Full integration with Ellis's codebase
   - Run DreamCoder on all 45 rules
   - Compare induced library with our grammar

3. **Long-term** (next quarter):
   - Fit model to behavioral data
   - Predict transfer learning patterns
   - Design curriculum based on model

## Contact

For questions about:
- **Our implementation**: [Your contact]
- **DreamCoder original**: Kevin Ellis (ellisk@mit.edu)
- **Behavioral experiment**: See main card-games repository
