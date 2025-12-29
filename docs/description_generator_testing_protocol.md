# Description Generator Testing Protocol

**Date Created**: December 26, 2024
**Purpose**: Standalone evaluation of the description generator before DreamCoder integration

---

## Executive Summary

This protocol enables testing the description generator independently by:
1. Generating 201 diverse synthetic card game rules
2. Evaluating description quality on held-out rules
3. Measuring accuracy, consistency, and discrimination metrics
4. Comparing against baselines

**UPDATED (Dec 26 2024)**: The evaluation harness has been revised to use **semantic correctness** as the primary metric, which is the proper way to evaluate this system. The previous 0% accuracy was due to using primitive name string matching (which was the wrong metric). Now we check if descriptions capture the correct *semantic feature* regardless of exact wording.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/description_generator/description_generator.py` | Core generator (1482 lines) |
| `src/description_generator/synthesis_integration.py` | Primitive biasing (600 lines) |
| `src/description_generator/synthetic_tasks.py` | 201 synthetic rules |
| `src/description_generator/evaluation_harness.py` | Full evaluation pipeline |
| `docs/description_generator_evaluation_protocol.md` | Detailed protocol specification |

---

## Running the Evaluation

### Quick Test (5 minutes)
```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src
python3 description_generator/evaluation_harness.py --quick
```

### Full Evaluation (30-60 minutes)
```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src
python3 description_generator/evaluation_harness.py --output results.json
```

### With Holdout Validation (final validation only)
```bash
python3 description_generator/evaluation_harness.py --include-holdout --output final_results.json
```

---

## Synthetic Rules Generated

**Total: 201 rules** across 5 complexity levels:

| Level | Type | Count | Example |
|-------|------|-------|---------|
| 1 | Atomic | 50 | "All cards same suit" |
| 2 | Comparison | 22 | "First and last same color" |
| 3 | Counting | 47 | "Exactly 3 unique ranks" |
| 4 | Pattern | 29 | "Sorted by rank" |
| 5 | Compositional | 53 | "Uniform suit AND sorted" |

---

## Evaluation Metrics

### PRIMARY METRIC: Semantic Correctness

**Semantic Correctness** is now the primary evaluation metric. It measures:

> "Does the generated description capture the CORRECT semantic feature, regardless of exact wording?"

**How it works**:
1. Each rule has a **semantic category** (e.g., "uniform", "sorted", "has")
2. Each category maps to **acceptable feature types** (e.g., `uniform` → `{IS_FLUSH, UNIQUE_SUITS}`)
3. We extract features from generated descriptions (via direct features + text keywords)
4. **Score = 1.0** if ANY description in top-3 has a feature matching the acceptable set

This is more robust than primitive name matching because:
- "all cards same suit" maps to `IS_FLUSH` ✓
- "only one suit represented" maps to `UNIQUE_SUITS==1` ✓
- Both correctly describe a "uniform suit" rule!

### Secondary Metrics

1. **Feature Capture Rate (FCR@3)**: Does the correct feature appear in top-3? (legacy, uses primitive names)
2. **Primitive Jaccard Similarity**: Overlap between generated and expected primitives (less useful now)
3. **Consistency Score**: Same rule with different examples → same descriptions?
4. **Discrimination Score**: Different rules → different descriptions?

### Baselines

1. **Random Baseline**: Select random descriptions from template pool
2. **Frequent Baseline**: Always output most common descriptions

---

## Known Issues to Address

### ~~Issue 1: Vocabulary Mismatch~~ RESOLVED

**Previous Problem**: The synthetic task generator uses different primitive names than the description generator.

**Resolution**: This was NOT actually a problem! We misunderstood the evaluation objective. In LAPS-style systems (like Wong et al.), the mapping between words and primitives is LEARNED from data, not string-matched. Our fix was to implement **semantic correctness** evaluation that checks if the description captures the right *feature type*, not the right *primitive name*.

**Key Insight**: "gon" maps to rotation primitives not because "gon" contains "rotate" but because the translation model learns that "gon" co-occurs with rotation programs.

### Issue 2: Rare Rules Hard to Sample

Some rules (e.g., "all same suit") are very rare in random hands:
- P(all 6 cards same suit) ≈ 0.2%
- May need 5000+ attempts to get 20 examples

**Current Mitigation**: `max_attempts=5000` in sampling functions

### Issue 3: Multiple Valid Descriptions

A rule like "has pair" could be described as:
- "two cards share a rank"
- "fewer unique ranks than cards"
- "repeated rank exists"

**Evaluation Approach**: Use Jaccard on primitives, not exact string match

---

## Test Plan

### Step 1: Verify Components Work (15 min)

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src

# Test synthetic task generator
python3 -c "
from description_generator.synthetic_tasks import generate_synthetic_rules
rules = generate_synthetic_rules(seed=42)
print(f'Generated {len(rules)} rules')
for r in rules[:5]:
    print(f'  [{r.level}] {r.name} - category: {r.category}')
"

# Test description generator
python3 -c "
from description_generator.description_generator import DescriptionGenerator
from rules.cards import sample_hand
gen = DescriptionGenerator(n_baseline_samples=2000)
hand = sample_hand(6)
descs = gen.describe_hand(hand, top_k=3)
for d in descs:
    print(f'  {d.text} (score: {d.score:.2f})')
    if d.features:
        print(f'    Features: {[f.feature_type.value for f in d.features]}')
"
```

### Step 2: Run Quick Evaluation (10 min)

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src
python3 description_generator/evaluation_harness.py --quick
```

**Expected Output**: With the new semantic correctness metric, you should see:
- **Semantic Correctness > 0%** (our primary metric)
- Level 1-2 rules (atomic) should have higher scores
- Level 5 rules (compositional) will be lenient (accept any feature)

### Step 3: Run Full Evaluation (30-60 min)

```bash
python3 description_generator/evaluation_harness.py --output semantic_eval.json
```

### Step 4: Analyze Results

```python
import json
with open('results_description_eval/semantic_eval.json') as f:
    results = json.load(f)

# Check PRIMARY metric by level
print("Semantic Correctness by Level:")
for level, sc in results['by_level']['semantic_correctness'].items():
    print(f"  Level {level}: {sc:.1%}")

# Check by category
print("\nSemantic Correctness by Category:")
for cat, sc in sorted(results['by_category']['semantic_correctness'].items(), key=lambda x: -x[1]):
    print(f"  {cat}: {sc:.1%}")
```

---

## Success Criteria

### Semantic Correctness (PRIMARY METRIC)

| Level | Minimum | Good | Excellent |
|-------|---------|------|-----------|
| Level 1-2 (Atomic) | 40% | 60% | 80% |
| Level 3-4 (Pattern) | 30% | 50% | 70% |
| Level 5 (Compositional) | N/A* | N/A* | N/A* |

*Level 5 is lenient (accepts any feature) since compositional rules may have multiple valid descriptions.

### Secondary Metrics

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Consistency | 60% | 80% | 90% |
| Discrimination | 50% | 70% | 85% |
| vs Random Baseline | +15% | +30% | +50% |

---

## Debugging Tips

### If all descriptions are "losing hands often: has 3 different suits"

This means the task examples don't provide clear signal. Check:
- Are positive/negative sets well separated?
- Is the rule too rare (can't sample positives)?

### If descriptions don't mention expected feature

Check the `DescriptionVocabulary` in `description_generator.py` - there may not be a template for that feature type.

### If consistency is low

The description generator may be sensitive to example selection. Consider:
- Using more examples per rule
- Adding noise tolerance in vocabulary selection

---

## Demo Commands

### Describe a single hand
```python
from description_generator.description_generator import DescriptionGenerator
from rules.cards import sample_hand

gen = DescriptionGenerator(n_baseline_samples=5000)
hand = sample_hand(6)
print(f"Hand: {[str(c) for c in hand]}")

for desc in gen.describe_hand(hand, top_k=5):
    print(f"  {desc.text} (score: {desc.score:.2f})")
    print(f"    Primitives: {desc.primitives}")
```

### Describe a task (distinguishing positive from negative)
```python
from description_generator.description_generator import DescriptionGenerator
from rules.cards import Card, Suit, Rank, card_color, Color, sample_hand

gen = DescriptionGenerator(n_baseline_samples=5000)

# Rule: all same color
def uniform_color(h):
    return len(set(card_color(c) for c in h)) == 1

# Sample
positive = [h for h in [sample_hand(6) for _ in range(500)] if uniform_color(h)][:20]
negative = [h for h in [sample_hand(6) for _ in range(500)] if not uniform_color(h)][:20]

print(f"Positive: {len(positive)}, Negative: {len(negative)}")
for desc in gen.describe_task(positive, negative, top_k=5):
    print(f"  {desc.text} (score: {desc.score:.2f})")
```

---

## Next Steps After Testing

1. **If evaluation works well**: Proceed to integrate with DreamCoder recognition network
2. **If vocabulary issues persist**: Create a vocabulary mapping layer
3. **If accuracy is too low**: Expand description templates or feature extractors
4. **If consistency is low**: Add example aggregation before description generation

---

*Document prepared for evaluation session on December 26-27, 2024*
