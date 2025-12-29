# Description Generator Evaluation Protocol

## Document Overview

This document specifies a comprehensive evaluation protocol for the card game description generator. The evaluation tests whether generated descriptions are:

1. **Accurate** - Do descriptions capture the true distinguishing feature?
2. **Informative** - Do descriptions convey useful, non-trivial information?
3. **Consistent** - Same rule yields same descriptions across example sets?
4. **Useful for Synthesis** - Do descriptions map to primitives that help program synthesis?

---

## 1. Task Generation Strategy

### 1.1 Primitive Coverage Matrix

Generate synthetic tasks that systematically cover the primitive space:

| Category | Primitives | Example Rules | Count |
|----------|-----------|---------------|-------|
| **Suit** | `get_suit`, `count_suit`, `has_suit`, `all_same_suit`, `n_unique_suits` | "All hearts", "Has 2+ spades", "Exactly 2 suits" | 15 |
| **Color** | `get_color`, `count_color`, `has_color`, `all_same_color`, `n_unique_colors` | "All red", "3 black cards", "Uniform color" | 12 |
| **Rank** | `rank_val`, `sum_ranks`, `max_rank`, `min_rank`, `n_unique_ranks` | "Sum > 40", "Max rank is face card", "Has pair" | 15 |
| **Position** | `head`, `last`, `first_half`, `second_half`, `at` | "First card red", "Ends match", "Left half > right" | 18 |
| **Pattern** | `adjacent_pairs`, `zip_with`, `reverse`, `all`, `any` | "Sorted", "Palindrome", "Adjacent same color" | 20 |
| **Compositional** | Combinations of above | "Halves copy colors", "Left flush AND right sorted" | 20 |

**Total: 100 synthetic tasks**

### 1.2 Synthetic Rule Generator

```python
def generate_synthetic_rules() -> List[Dict]:
    """Generate 100 synthetic rules with ground truth metadata."""
    rules = []

    # === ATOMIC SUIT RULES (15) ===
    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
        # "All cards are {suit}"
        rules.append({
            'id': f'all_{suit.name.lower()}',
            'predicate': lambda h, s=suit: all(c.suit == s for c in h),
            'ground_truth_description': f'all cards are {suit.name.lower()}',
            'primitives': ['all_same_suit', 'eq', suit.name],
            'category': 'suit_uniform',
            'difficulty': 'easy'
        })
        # "Has at least one {suit}"
        rules.append({
            'id': f'has_{suit.name.lower()}',
            'predicate': lambda h, s=suit: any(c.suit == s for c in h),
            'ground_truth_description': f'contains at least one {suit.name.lower()}',
            'primitives': ['has_suit', suit.name],
            'category': 'suit_existence',
            'difficulty': 'easy'
        })

    # "Exactly N unique suits" for N in 1,2,3
    for n in [1, 2, 3]:
        rules.append({
            'id': f'exactly_{n}_suits',
            'predicate': lambda h, n=n: len(set(c.suit for c in h)) == n,
            'ground_truth_description': f'exactly {n} different suit(s) appear',
            'primitives': ['n_unique_suits', 'eq', str(n)],
            'category': 'suit_cardinality',
            'difficulty': 'easy'
        })

    # === ATOMIC COLOR RULES (12) ===
    for color in [Color.RED, Color.BLACK]:
        rules.append({
            'id': f'all_{color.name.lower()}',
            'predicate': lambda h, c=color: all(card_color(card) == c for card in h),
            'ground_truth_description': f'all cards are {color.name.lower()}',
            'primitives': ['all_same_color', 'eq', color.name],
            'category': 'color_uniform',
            'difficulty': 'easy'
        })
        for count in [1, 2, 3]:
            rules.append({
                'id': f'exactly_{count}_{color.name.lower()}',
                'predicate': lambda h, c=color, n=count: sum(1 for card in h if card_color(card) == c) == n,
                'ground_truth_description': f'exactly {count} {color.name.lower()} card(s)',
                'primitives': ['count_color', 'eq', color.name, str(count)],
                'category': 'color_count',
                'difficulty': 'easy'
            })
    # Uniform color (either all red or all black)
    rules.append({
        'id': 'uniform_color',
        'predicate': lambda h: len(set(card_color(c) for c in h)) == 1,
        'ground_truth_description': 'all cards are the same color',
        'primitives': ['all_same_color'],
        'category': 'color_uniform',
        'difficulty': 'easy'
    })

    # === RANK RULES (15) ===
    # Sum thresholds
    for threshold in [30, 35, 40]:
        rules.append({
            'id': f'sum_gt_{threshold}',
            'predicate': lambda h, t=threshold: sum(RANK_VALUES[c.rank] for c in h) > t,
            'ground_truth_description': f'sum of ranks exceeds {threshold}',
            'primitives': ['sum_ranks', 'gt', str(threshold)],
            'category': 'rank_aggregate',
            'difficulty': 'medium'
        })

    # Max rank thresholds
    for threshold in [10, 11, 12]:
        rules.append({
            'id': f'max_rank_ge_{threshold}',
            'predicate': lambda h, t=threshold: max(RANK_VALUES[c.rank] for c in h) >= t,
            'ground_truth_description': f'highest rank is at least {threshold}',
            'primitives': ['max_rank', 'ge', str(threshold)],
            'category': 'rank_aggregate',
            'difficulty': 'medium'
        })

    # Pair detection
    rules.append({
        'id': 'has_pair',
        'predicate': lambda h: len(h) > len(set(c.rank for c in h)),
        'ground_truth_description': 'contains a pair (same rank)',
        'primitives': ['n_unique_ranks', 'lt', 'length'],
        'category': 'rank_pattern',
        'difficulty': 'medium'
    })

    # No pairs (all unique ranks)
    rules.append({
        'id': 'no_pair',
        'predicate': lambda h: len(h) == len(set(c.rank for c in h)),
        'ground_truth_description': 'no repeated ranks',
        'primitives': ['n_unique_ranks', 'eq', 'length'],
        'category': 'rank_pattern',
        'difficulty': 'medium'
    })

    # === POSITIONAL RULES (18) ===
    # First card properties
    for suit in Suit:
        rules.append({
            'id': f'first_is_{suit.name.lower()}',
            'predicate': lambda h, s=suit: h[0].suit == s if h else False,
            'ground_truth_description': f'first card is a {suit.name.lower()}',
            'primitives': ['head', 'get_suit', 'eq', suit.name],
            'category': 'positional_first',
            'difficulty': 'easy'
        })

    # Last card properties
    for color in Color:
        rules.append({
            'id': f'last_is_{color.name.lower()}',
            'predicate': lambda h, c=color: card_color(h[-1]) == c if h else False,
            'ground_truth_description': f'last card is {color.name.lower()}',
            'primitives': ['last', 'get_color', 'eq', color.name],
            'category': 'positional_last',
            'difficulty': 'easy'
        })

    # Ends match
    rules.append({
        'id': 'ends_same_suit',
        'predicate': lambda h: h[0].suit == h[-1].suit if len(h) >= 2 else False,
        'ground_truth_description': 'first and last cards share the same suit',
        'primitives': ['head', 'last', 'get_suit', 'eq'],
        'category': 'positional_terminals',
        'difficulty': 'medium'
    })
    rules.append({
        'id': 'ends_same_color',
        'predicate': lambda h: card_color(h[0]) == card_color(h[-1]) if len(h) >= 2 else False,
        'ground_truth_description': 'first and last cards are the same color',
        'primitives': ['head', 'last', 'get_color', 'eq'],
        'category': 'positional_terminals',
        'difficulty': 'medium'
    })

    # === PATTERN RULES (20) ===
    # Sorted
    rules.append({
        'id': 'sorted_ascending',
        'predicate': lambda h: all(RANK_VALUES[h[i].rank] <= RANK_VALUES[h[i+1].rank] for i in range(len(h)-1)),
        'ground_truth_description': 'ranks are in non-decreasing order',
        'primitives': ['adjacent_pairs', 'all', 'le', 'rank_val'],
        'category': 'pattern_order',
        'difficulty': 'hard'
    })

    # Palindrome patterns
    rules.append({
        'id': 'suit_palindrome',
        'predicate': lambda h: [c.suit for c in h] == [c.suit for c in reversed(h)],
        'ground_truth_description': 'suit sequence reads the same forward and backward',
        'primitives': ['map', 'get_suit', 'reverse', 'eq'],
        'category': 'pattern_symmetry',
        'difficulty': 'hard'
    })
    rules.append({
        'id': 'color_palindrome',
        'predicate': lambda h: [card_color(c) for c in h] == [card_color(c) for c in reversed(h)],
        'ground_truth_description': 'color sequence reads the same forward and backward',
        'primitives': ['map', 'get_color', 'reverse', 'eq'],
        'category': 'pattern_symmetry',
        'difficulty': 'hard'
    })

    # Adjacent constraints
    rules.append({
        'id': 'adjacent_same_color',
        'predicate': lambda h: all(card_color(h[i]) == card_color(h[i+1]) for i in range(len(h)-1)),
        'ground_truth_description': 'every adjacent pair shares the same color',
        'primitives': ['adjacent_pairs', 'all', 'get_color', 'eq'],
        'category': 'pattern_adjacent',
        'difficulty': 'hard'
    })
    rules.append({
        'id': 'adjacent_different_color',
        'predicate': lambda h: all(card_color(h[i]) != card_color(h[i+1]) for i in range(len(h)-1)),
        'ground_truth_description': 'every adjacent pair has different colors (alternating)',
        'primitives': ['adjacent_pairs', 'all', 'get_color', 'not', 'eq'],
        'category': 'pattern_adjacent',
        'difficulty': 'hard'
    })

    # === COMPOSITIONAL RULES (20) ===
    # Halves copy
    rules.append({
        'id': 'halves_copy_suits',
        'predicate': lambda h: [c.suit for c in h[:len(h)//2]] == [c.suit for c in h[len(h)//2:]],
        'ground_truth_description': 'left half suits match right half suits exactly',
        'primitives': ['first_half', 'second_half', 'map', 'get_suit', 'eq'],
        'category': 'compositional_halves',
        'difficulty': 'hard'
    })
    rules.append({
        'id': 'halves_copy_colors',
        'predicate': lambda h: [card_color(c) for c in h[:len(h)//2]] == [card_color(c) for c in h[len(h)//2:]],
        'ground_truth_description': 'left half colors match right half colors exactly',
        'primitives': ['first_half', 'second_half', 'map', 'get_color', 'eq'],
        'category': 'compositional_halves',
        'difficulty': 'hard'
    })

    # Halves property equality
    rules.append({
        'id': 'halves_both_uniform_color',
        'predicate': lambda h: (
            len(set(card_color(c) for c in h[:len(h)//2])) == 1 and
            len(set(card_color(c) for c in h[len(h)//2:])) == 1
        ),
        'ground_truth_description': 'both halves are uniform in color',
        'primitives': ['first_half', 'second_half', 'all_same_color', 'and'],
        'category': 'compositional_halves',
        'difficulty': 'hard'
    })

    # AND compositions
    rules.append({
        'id': 'first_red_and_last_black',
        'predicate': lambda h: card_color(h[0]) == Color.RED and card_color(h[-1]) == Color.BLACK if len(h) >= 2 else False,
        'ground_truth_description': 'first card is red AND last card is black',
        'primitives': ['head', 'last', 'get_color', 'eq', 'RED', 'BLACK', 'and'],
        'category': 'compositional_and',
        'difficulty': 'medium'
    })
    rules.append({
        'id': 'has_heart_and_has_spade',
        'predicate': lambda h: any(c.suit == Suit.HEARTS for c in h) and any(c.suit == Suit.SPADES for c in h),
        'ground_truth_description': 'contains at least one heart AND at least one spade',
        'primitives': ['has_suit', 'HEARTS', 'SPADES', 'and'],
        'category': 'compositional_and',
        'difficulty': 'medium'
    })

    # OR compositions
    rules.append({
        'id': 'all_red_or_all_black',
        'predicate': lambda h: len(set(card_color(c) for c in h)) == 1,
        'ground_truth_description': 'all cards are the same color (all red OR all black)',
        'primitives': ['all_same_color'],
        'category': 'compositional_or',
        'difficulty': 'easy'
    })

    return rules
```

### 1.3 Difficulty Levels

| Level | Description | Expected Hit Rate | Example |
|-------|-------------|-------------------|---------|
| **Easy** | Single primitive | 90%+ | "has_suit HEARTS" |
| **Medium** | 2-3 primitives composed | 70-90% | "ends_same_suit" |
| **Hard** | 4+ primitives, hierarchical | 40-70% | "halves_copy_colors" |

---

## 2. Train/Test Split Strategy

### 2.1 Dataset Partition

| Set | Purpose | Size | Rules |
|-----|---------|------|-------|
| **Baseline Calibration** | Build baseline statistics | 50% of hands | Random hands, no rules |
| **Development** | Tune thresholds | 20 rules (20%) | Mixed difficulty |
| **In-Distribution Test** | Evaluate on seen categories | 40 rules (40%) | Categories in dev set |
| **Out-of-Distribution Test** | Evaluate generalization | 20 rules (20%) | New compositions |
| **Holdout** | Final validation | 20 rules (20%) | Completely held out |

### 2.2 Category-Based Splitting

Ensure each category has representation in train AND test:

```
CATEGORY DISTRIBUTION:
                          Dev    InDist  OutDist  Holdout
suit_uniform              2      4       0        2
suit_existence            2      4       0        2
suit_cardinality          1      2       0        0
color_uniform             2      3       0        1
color_count               2      4       0        2
rank_aggregate            2      3       1        0
rank_pattern              1      2       0        0
positional_first          2      3       0        1
positional_last           1      2       0        1
positional_terminals      1      2       0        0
pattern_order             0      1       1        0
pattern_symmetry          1      1       1        1
pattern_adjacent          1      1       1        1
compositional_halves      1      2       3        1
compositional_and         1      2       2        1
compositional_or          0      1       1        0
```

### 2.3 Out-of-Distribution Design

The OOD test set should include:

1. **Novel Compositions**: Combine primitives in ways not seen during development
   - Example: "first half sorted AND second half has pair"

2. **Longer Chains**: More deeply nested compositions
   - Example: "palindrome AND ends same suit AND has exactly 2 colors"

3. **Rare Feature Combinations**: Statistically unusual configurations
   - Example: "exactly 4 hearts AND sorted" (very rare joint occurrence)

---

## 3. Evaluation Metrics

### 3.1 Accuracy Metrics

#### 3.1.1 Feature Capture Rate (FCR)

Does the top-K description mention the ground truth feature?

```python
def feature_capture_rate(descriptions: List[Description], ground_truth: str, k: int = 3) -> float:
    """
    Compute whether ground truth feature appears in top-k descriptions.

    Args:
        descriptions: List of generated descriptions, sorted by score
        ground_truth: The ground truth description string
        k: Number of top descriptions to check

    Returns:
        1.0 if captured, 0.0 otherwise
    """
    top_k = [d.text.lower() for d in descriptions[:k]]
    ground_truth_lower = ground_truth.lower()

    # Check for substring match or semantic similarity
    for desc in top_k:
        if semantic_overlap(desc, ground_truth_lower) > 0.5:
            return 1.0
    return 0.0
```

**Target**: FCR@3 >= 0.80 for Easy, >= 0.60 for Medium, >= 0.40 for Hard

#### 3.1.2 Primitive Jaccard Similarity (PJS)

How well do the description's primitives overlap with ground truth?

```python
def primitive_jaccard(desc_primitives: List[str], truth_primitives: List[str]) -> float:
    """
    Compute Jaccard similarity between primitive sets.

    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    desc_set = set(desc_primitives)
    truth_set = set(truth_primitives)

    intersection = len(desc_set & truth_set)
    union = len(desc_set | truth_set)

    return intersection / union if union > 0 else 0.0
```

**Target**: Mean PJS >= 0.50 across all difficulty levels

### 3.2 Consistency Metrics

#### 3.2.1 Same-Rule Consistency (SRC)

Given the same rule with different example sets, do we get the same descriptions?

```python
def same_rule_consistency(rule, n_trials: int = 10, k: int = 3) -> float:
    """
    Measure consistency of descriptions across different example sets.

    For each trial:
    1. Sample new positive/negative examples for the rule
    2. Generate top-k descriptions
    3. Compare to reference descriptions

    Returns: Mean pairwise description overlap
    """
    all_desc_sets = []

    for _ in range(n_trials):
        pos, neg = sample_examples(rule, n_pos=20, n_neg=20)
        descs = generator.describe_task(pos, neg, top_k=k)
        desc_texts = frozenset(d.text.lower() for d in descs)
        all_desc_sets.append(desc_texts)

    # Compute pairwise Jaccard similarities
    similarities = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            sim = len(all_desc_sets[i] & all_desc_sets[j]) / len(all_desc_sets[i] | all_desc_sets[j])
            similarities.append(sim)

    return np.mean(similarities)
```

**Target**: SRC >= 0.70 for Easy rules, >= 0.50 for Hard rules

#### 3.2.2 Description Stability (DS)

Does adding more examples change the description ranking?

```python
def description_stability(rule, example_sizes: List[int] = [10, 20, 50, 100]) -> float:
    """
    Measure how stable descriptions are as we add more examples.

    Compute Spearman rank correlation between description rankings
    at successive example sizes.
    """
    rankings = []

    for size in example_sizes:
        pos, neg = sample_examples(rule, n_pos=size, n_neg=size)
        descs = generator.describe_task(pos, neg, top_k=10)
        ranking = {d.text: i for i, d in enumerate(descs)}
        rankings.append(ranking)

    # Compute correlation between successive rankings
    correlations = []
    for i in range(len(rankings) - 1):
        r1, r2 = rankings[i], rankings[i+1]
        common = set(r1.keys()) & set(r2.keys())
        if len(common) >= 3:
            ranks1 = [r1[k] for k in common]
            ranks2 = [r2[k] for k in common]
            corr, _ = spearmanr(ranks1, ranks2)
            correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0
```

**Target**: DS >= 0.80 (rank correlation from 10 to 100 examples)

### 3.3 Discrimination Metrics

#### 3.3.1 Between-Rule Discrimination (BRD)

Different rules should yield different descriptions.

```python
def between_rule_discrimination(rules: List[Rule], k: int = 5) -> float:
    """
    Measure whether different rules get different descriptions.

    For each pair of different rules:
    1. Generate descriptions for each
    2. Compute description set overlap

    Returns: 1 - mean overlap (higher = better discrimination)
    """
    rule_descs = {}

    for rule in rules:
        pos, neg = sample_examples(rule, n_pos=20, n_neg=20)
        descs = generator.describe_task(pos, neg, top_k=k)
        rule_descs[rule.id] = frozenset(d.text.lower() for d in descs)

    # Compute pairwise overlaps between different rules
    overlaps = []
    rule_ids = list(rule_descs.keys())

    for i in range(len(rule_ids)):
        for j in range(i+1, len(rule_ids)):
            set_i = rule_descs[rule_ids[i]]
            set_j = rule_descs[rule_ids[j]]
            if set_i and set_j:
                overlap = len(set_i & set_j) / min(len(set_i), len(set_j))
                overlaps.append(overlap)

    return 1.0 - np.mean(overlaps)
```

**Target**: BRD >= 0.60 (40% or less description overlap between different rules)

#### 3.3.2 Category Confusion Matrix

Can we identify the rule category from descriptions alone?

```python
def build_category_confusion_matrix(rules: List[Rule], generator) -> np.ndarray:
    """
    Build confusion matrix: true category vs predicted category.

    Prediction: Assign rule to category whose prototype descriptions
    have highest overlap with generated descriptions.
    """
    # Build category prototypes from dev set
    category_prototypes = {}
    for rule in dev_rules:
        descs = generator.describe_task(...)
        category = rule['category']
        if category not in category_prototypes:
            category_prototypes[category] = Counter()
        for d in descs:
            category_prototypes[category][d.text.lower()] += 1

    # Predict category for test rules
    confusion = defaultdict(lambda: defaultdict(int))
    for rule in test_rules:
        descs = generator.describe_task(...)
        desc_set = set(d.text.lower() for d in descs)

        # Find best matching category
        best_category = None
        best_score = -1
        for cat, prototype in category_prototypes.items():
            score = sum(prototype.get(d, 0) for d in desc_set)
            if score > best_score:
                best_score = score
                best_category = cat

        true_category = rule['category']
        confusion[true_category][best_category] += 1

    return confusion
```

**Target**: Category prediction accuracy >= 0.50 (above chance for 15+ categories)

### 3.4 Information Metrics

#### 3.4.1 Surprise Score Distribution

Are descriptions actually informative (not just generic)?

```python
def surprise_score_analysis(rules: List[Rule], generator) -> Dict[str, float]:
    """
    Analyze the distribution of surprise scores in generated descriptions.

    Good descriptions should have:
    - Mean surprise > 2.0 bits (not just baseline features)
    - Variance in surprise (some features more surprising than others)
    - Top descriptions have higher surprise than random features
    """
    all_surprises = []
    top_k_surprises = []

    for rule in rules:
        pos, neg = sample_examples(rule, n_pos=20, n_neg=20)
        descs = generator.describe_task(pos, neg, top_k=10)

        for i, d in enumerate(descs):
            all_surprises.append(d.score)
            if i < 3:
                top_k_surprises.append(d.score)

    return {
        'mean_surprise': np.mean(all_surprises),
        'std_surprise': np.std(all_surprises),
        'mean_top3_surprise': np.mean(top_k_surprises),
        'min_surprise': np.min(all_surprises),
        'max_surprise': np.max(all_surprises)
    }
```

**Target**: Mean surprise >= 2.0 bits, top-3 mean >= 3.0 bits

#### 3.4.2 Description Entropy

Are we generating diverse descriptions or repeating the same ones?

```python
def description_entropy(rules: List[Rule], generator) -> float:
    """
    Compute entropy of description distribution across rules.

    Higher entropy = more diverse descriptions
    Lower entropy = descriptions are repetitive
    """
    desc_counts = Counter()

    for rule in rules:
        pos, neg = sample_examples(rule, n_pos=20, n_neg=20)
        descs = generator.describe_task(pos, neg, top_k=5)
        for d in descs:
            desc_counts[d.text.lower()] += 1

    # Compute entropy
    total = sum(desc_counts.values())
    probs = [c / total for c in desc_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    # Normalize by maximum entropy (log2 of number of unique descriptions)
    max_entropy = np.log2(len(desc_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy
```

**Target**: Normalized entropy >= 0.70 (descriptions are diverse, not repetitive)

---

## 4. Ground Truth Acquisition

### 4.1 Programmatic Ground Truth

Use rule metadata from `catalogue.py` to derive ground truth:

```python
def extract_ground_truth_from_rule(rule: Rule) -> Dict[str, Any]:
    """
    Extract ground truth description and primitives from rule metadata.

    The catalogue already contains:
    - rule.description: Human-readable description
    - rule.primitives_used: List of primitive names
    - rule.composition: Compositional structure
    """
    return {
        'description': rule.description,
        'primitives': rule.primitives_used,
        'composition': str(rule.composition),
        'family': rule.family,
        'level': rule.level
    }
```

### 4.2 Round-Trip Validation

Can we use descriptions to guide synthesis and verify correctness?

```python
def round_trip_validation(rule, generator, synthesizer) -> Dict[str, Any]:
    """
    Round-trip validation:
    1. Generate descriptions from examples
    2. Use descriptions to bias primitive probabilities
    3. Run synthesis with biased grammar
    4. Verify synthesized program against held-out examples

    Returns:
        success: Whether synthesis found correct program
        speedup: Ratio of programs enumerated with vs without descriptions
        program: The synthesized program (if successful)
    """
    # Step 1: Generate descriptions
    pos_train, neg_train = sample_examples(rule, n_pos=30, n_neg=30)
    pos_test, neg_test = sample_examples(rule, n_pos=20, n_neg=20)

    descriptions = generator.describe_task(pos_train, neg_train, top_k=5)

    # Step 2: Extract primitive weights from descriptions
    primitive_weights = defaultdict(lambda: 1.0)
    for desc in descriptions:
        for prim in desc.primitives:
            primitive_weights[prim] += desc.score

    # Step 3: Run synthesis with biased grammar
    biased_grammar = build_biased_grammar(primitive_weights)

    # Baseline synthesis (unbiased)
    baseline_result = synthesizer.enumerate(
        grammar=uniform_grammar,
        examples=(pos_train, neg_train),
        timeout=60
    )

    # Guided synthesis (biased by descriptions)
    guided_result = synthesizer.enumerate(
        grammar=biased_grammar,
        examples=(pos_train, neg_train),
        timeout=60
    )

    # Step 4: Verify on held-out examples
    if guided_result.program:
        holdout_success = all(
            guided_result.program.eval(h) == True for h in pos_test
        ) and all(
            guided_result.program.eval(h) == False for h in neg_test
        )
    else:
        holdout_success = False

    return {
        'success': holdout_success,
        'baseline_programs': baseline_result.programs_enumerated,
        'guided_programs': guided_result.programs_enumerated,
        'speedup': baseline_result.programs_enumerated / guided_result.programs_enumerated if guided_result.programs_enumerated > 0 else 0,
        'program': str(guided_result.program) if guided_result.program else None
    }
```

**Target**:
- Synthesis success rate >= 80% with descriptions
- Mean speedup >= 2x compared to unguided synthesis

### 4.3 Human Validation (Optional Future Work)

If human validation is desired later:

```python
def generate_human_evaluation_tasks(rules: List[Rule], n_tasks: int = 50) -> List[Dict]:
    """
    Generate tasks for human evaluation via crowdsourcing.

    Task types:
    1. Description Accuracy: "Does this description correctly distinguish winning hands?"
    2. Description Clarity: "How clear is this description?" (1-5 scale)
    3. Description Preference: "Which description better captures the rule?" (A/B)
    """
    tasks = []

    for rule in random.sample(rules, n_tasks):
        pos, neg = sample_examples(rule, n_pos=5, n_neg=5)
        descs = generator.describe_task(pos, neg, top_k=3)

        tasks.append({
            'type': 'accuracy',
            'positive_hands': [hand_to_string(h) for h in pos],
            'negative_hands': [hand_to_string(h) for h in neg],
            'description': descs[0].text,
            'question': 'Does this description correctly capture what distinguishes winning from losing hands?',
            'options': ['Yes', 'Partially', 'No']
        })

    return tasks
```

---

## 5. Experimental Procedure

### 5.1 Setup (Day Before)

| Time | Task | Details |
|------|------|---------|
| T-24h | Generate synthetic rules | Run `generate_synthetic_rules()` to create 100 rules |
| T-23h | Create data splits | Partition into Dev/InDist/OOD/Holdout |
| T-22h | Sample baseline hands | Generate 10,000 random hands for baseline statistics |
| T-21h | Build baseline model | Run `SurpriseScorer` with baseline samples |
| T-20h | Verify infrastructure | Test all metric functions with 5 example rules |

### 5.2 Calibration Phase (Morning)

| Step | Action | Duration | Output |
|------|--------|----------|--------|
| 1 | Initialize generator with baseline | 5 min | `generator` object |
| 2 | Run on Dev set (20 rules) | 30 min | Description files |
| 3 | Compute FCR, PJS on Dev | 10 min | Calibration metrics |
| 4 | Adjust thresholds if needed | 15 min | Updated config |
| 5 | Re-run Dev with adjusted params | 30 min | Final Dev results |

### 5.3 Evaluation Phase (Afternoon)

| Step | Action | Duration | Output |
|------|--------|----------|--------|
| 6 | Run on In-Distribution test (40 rules) | 60 min | InDist descriptions |
| 7 | Run on OOD test (20 rules) | 30 min | OOD descriptions |
| 8 | Run on Holdout (20 rules) | 30 min | Holdout descriptions |
| 9 | Compute all metrics | 45 min | Full metrics report |
| 10 | Generate visualizations | 30 min | Plots and tables |

### 5.4 Round-Trip Validation Phase (Evening)

| Step | Action | Duration | Output |
|------|--------|----------|--------|
| 11 | Select 30 rules for synthesis test | 10 min | Synthesis test set |
| 12 | Run guided synthesis | 120 min | Synthesis results |
| 13 | Compute speedup metrics | 20 min | Speedup report |
| 14 | Verify on holdout examples | 30 min | Verification results |

### 5.5 Report Generation (End of Day)

| Step | Action | Duration | Output |
|------|--------|----------|--------|
| 15 | Compile all metrics | 30 min | `evaluation_results.json` |
| 16 | Generate HTML report | 15 min | `evaluation_report.html` |
| 17 | Create summary figures | 30 min | `figures/` directory |
| 18 | Write analysis notes | 30 min | `evaluation_notes.md` |

---

## 6. Expected Results and Success Criteria

### 6.1 Primary Success Criteria

| Metric | Threshold | Priority |
|--------|-----------|----------|
| FCR@3 (Easy) | >= 0.80 | High |
| FCR@3 (Medium) | >= 0.60 | High |
| FCR@3 (Hard) | >= 0.40 | Medium |
| Mean PJS | >= 0.50 | High |
| SRC (Easy) | >= 0.70 | Medium |
| BRD | >= 0.60 | Medium |
| Synthesis Speedup | >= 2x | High |
| Holdout Verification | >= 80% | High |

### 6.2 Secondary Metrics to Monitor

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Mean Surprise | 2.0-5.0 bits | Too low = generic; too high = noisy |
| Normalized Entropy | 0.60-0.90 | Balance diversity and consistency |
| Description Stability | >= 0.80 | Descriptions shouldn't change with more examples |
| OOD FCR Drop | <= 20% | Generalization penalty |

### 6.3 Failure Modes to Watch For

1. **Generic Descriptions**: If most descriptions are "all cards are the same color" regardless of rule
   - Diagnostic: Low surprise scores, high BRD violation
   - Fix: Increase surprise weight, add more specific templates

2. **Unstable Descriptions**: Different examples of same rule yield different descriptions
   - Diagnostic: Low SRC, high variance in DS
   - Fix: Increase example count, use more robust feature aggregation

3. **Category Leakage**: Descriptions from one category appear in unrelated rules
   - Diagnostic: Low BRD, confusing confusion matrix
   - Fix: Add more discriminative features, improve template specificity

4. **Primitive Mismatch**: High FCR but low PJS
   - Diagnostic: Descriptions are semantically correct but don't map to right primitives
   - Fix: Improve template-to-primitive mapping

---

## 7. Implementation Checklist

### 7.1 Pre-Experiment

- [ ] Synthetic rule generator implemented and tested
- [ ] Data splits created and saved to disk
- [ ] Baseline statistics computed and cached
- [ ] All metric functions unit tested
- [ ] Logging infrastructure set up
- [ ] Results directory structure created

### 7.2 During Experiment

- [ ] Dev set calibration complete
- [ ] Threshold adjustments documented
- [ ] InDist evaluation complete
- [ ] OOD evaluation complete
- [ ] Holdout evaluation complete
- [ ] Round-trip synthesis tests complete

### 7.3 Post-Experiment

- [ ] All metrics computed
- [ ] Results saved to JSON
- [ ] HTML report generated
- [ ] Figures created
- [ ] Analysis notes written
- [ ] Git commit with results

---

## 8. Appendix: Metric Computation Code

See `/Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src/description_generator/evaluation_metrics.py` (to be created) for full implementation of all metrics defined in this protocol.
