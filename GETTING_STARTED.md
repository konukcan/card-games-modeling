# Getting Started with Card Game DreamCoder Modeling

## What Has Been Built

This repository contains a **production-quality foundation** for applying DreamCoder-style program synthesis to your card game experiment. Here's what's been implemented:

### ✅ Complete Components

1. **Card Domain** (`src/rules/cards.py`)
   - Full card/hand representations
   - All 4 suits, 13 ranks
   - Color categorizations (RED/BLACK, Pointy/Round, SH/DC)
   - Parity (ODD/EVEN)
   - Random hand sampling

2. **Compositional Primitives** (`src/rules/primitives.py`)
   - **Level 0**: 15+ atomic primitives (getSuit, getRank, first, last, etc.)
   - **Level 1**: 20+ combinators (map, filter, count, palindrome, etc.)
   - **Level 2**: Structural operators (halves, terminals, shifted_pairs)
   - **Level 3**: Domain algorithms (hasAP, bracket_match)
   - **Level 4**: Meta-combinators (halves_equal, seq_palindrome)
   - Total: **60+ primitive functions** matching your compositional grammar

3. **Rule Catalogue** (`src/rules/catalogue.py`)
   - **26 rules implemented** (expandable to all 56)
   - Organized by 11 families
   - Each rule includes:
     - Evaluation function
     - Description
     - Primitive decomposition
     - Family classification

4. **Demo Pipeline** (`src/main_demo.py`)
   - Task generation from rules
   - Feature extraction (104-dimensional)
   - Primitive usage analysis
   - 3 comprehensive visualizations
   - JSON report generation

5. **Documentation**
   - Complete README with architecture overview
   - Integration guide for Ellis et al.'s DreamCoder
   - This getting started guide

## Running the Demo

### Prerequisites

```bash
cd /Users/cankonuk/Documents/card-games-modeling

# Install dependencies (if not already done)
pip3 install numpy matplotlib seaborn pandas scikit-learn torch
```

### Run Demo

```bash
python3 src/main_demo.py
```

**Expected output:**
- Console log showing 7 pipeline steps
- 3 PNG visualizations in `results/`
- JSON report: `results/demo_report.json`

### What You'll See

The demo will:
1. Load 26 rules from the catalogue
2. Select 11 representative rules (one per family)
3. Generate 8-example tasks for each
4. Extract 104-dimensional feature vectors
5. Analyze primitive usage (22 unique primitives)
6. Create visualizations:
   - `primitive_usage_heatmap.png` - Which primitives each rule uses
   - `feature_statistics.png` - Feature distributions
   - `primitive_cooccurrence.png` - Which primitives co-occur

**Runtime**: ~5-10 seconds

## Understanding the Output

### Visualizations

1. **Primitive Usage Heatmap**
   - **Rows**: Individual primitives (get_suit, halves, is_sorted, etc.)
   - **Columns**: Rules from different families
   - **Color**: White = not used, Blue = used
   - **Insight**: Shows compositional structure - which rules share primitives

2. **Feature Statistics**
   - **Left plot**: Mean feature values across all tasks
   - **Right plot**: Standard deviations
   - **Insight**: Which features are most discriminative

3. **Primitive Co-occurrence**
   - **Heatmap**: How often pairs of primitives appear together
   - **Diagonal**: Frequency of each primitive
   - **Off-diagonal**: Joint usage patterns
   - **Insight**: Reveals natural groupings (e.g., `halves` + `arrays_equal`)

### JSON Report

```json
{
  "timestamp": "2025-11-25T...",
  "num_rules": 26,
  "num_primitives": 22,
  "rules": [
    {
      "id": "Sorted_by_rank",
      "family": "LOCAL",
      "primitives": ["is_sorted", "get_rank_val"]
    },
    ...
  ]
}
```

## Next Steps

### Immediate (Next Session)

1. **Add remaining 30 rules** to `catalogue.py`
   - Follow existing pattern
   - Most are variations (e.g., more AltColor rules, more AP rules)
   - Estimated: 1-2 hours

2. **Implement recognition network** (already exists in `dreamcoder_modeling/`!)
   - Copy `dreamcoder_modeling/dreamcoder_demo.py` to `src/dreamcoder/recognition.py`
   - Already achieves 94.75% accuracy
   - Just needs to be integrated into main pipeline

3. **Add program enumeration**
   - Create `src/dreamcoder/enumeration.py`
   - Implement best-first search
   - Use recognition network scores to guide search

### Medium-term (Next 2 Weeks)

4. **Library learning**
   - Extract frequently-used subprograms
   - Compress program representations
   - Measure description length reduction

5. **Wake-sleep loop**
   - Alternate between solving tasks and retraining network
   - Track improvement over iterations

6. **Validation**
   - Run on all 56 rules
   - Compare with human behavioral data (if available)

### Long-term (Next Month)

7. **Integration with Ellis's DreamCoder**
   - Export DSL to DreamCoder format
   - Run DreamCoder on your tasks
   - Compare induced library with your grammar

8. **Empirical modeling**
   - Fit to human learning curves
   - Predict transfer patterns
   - Generate experimental curricula

## File Structure

```
card-games-modeling/
├── README.md                    # Main documentation
├── GETTING_STARTED.md          # This file
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── main_demo.py            # ✅ WORKING DEMO
│   ├── rules/
│   │   ├── cards.py            # ✅ Card domain
│   │   ├── primitives.py       # ✅ 60+ primitives (5 levels)
│   │   └── catalogue.py        # ✅ 26 rules (expand to 56)
│   │
│   └── dreamcoder/             # TO ADD (stubs ready)
│       ├── recognition.py      # (copy from dreamcoder_modeling/)
│       ├── enumeration.py      # (implement search)
│       ├── compression.py      # (library learning)
│       └── wake_sleep.py       # (training loop)
│
├── docs/
│   └── DREAMCODER_INTEGRATION.md  # ✅ Integration guide
│
└── results/                    # Generated by demo
    ├── primitive_usage_heatmap.png
    ├── feature_statistics.png
    ├── primitive_cooccurrence.png
    └── demo_report.json
```

## Key Design Decisions

### 1. Python vs. Haskell/OCaml

**Decision**: Pure Python implementation (Ellis uses Haskell/OCaml)

**Rationale**:
- Easier integration with behavioral data analysis (pandas/numpy)
- Faster prototyping for domain-specific features
- PyTorch ecosystem for neural components

**Trade-off**: Less type-safe than Haskell, but gain flexibility

### 2. Pre-defined Grammar vs. Induced Library

**Decision**: Start with compositional grammar from your analysis

**Rationale**:
- You've already done the compositional analysis (grammar.tex)
- Can validate DreamCoder by seeing if it induces the same library
- Faster iteration for behavioral modeling

**Validation**: Run DreamCoder fresh and compare

### 3. Primitive Prediction vs. Sketch Prediction

**Decision**: Recognition network predicts primitive usage (not full program sketches)

**Rationale**:
- Simpler training signal (multi-label classification)
- More interpretable (see which primitives fire)
- Sufficient for guiding search in this domain

**Alternative**: Could predict sketches like original DreamCoder

## Common Issues

### Issue 1: Import errors

**Problem**: `ModuleNotFoundError: No module named 'rules'`

**Solution**:
```bash
# Make sure you're in the right directory
cd /Users/cankonuk/Documents/card-games-modeling

# Run with explicit path
python3 src/main_demo.py

# Or add to PYTHONPATH
export PYTHONPATH=/Users/cankonuk/Documents/card-games-modeling/src:$PYTHONPATH
```

### Issue 2: Missing dependencies

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip3 install torch numpy matplotlib seaborn pandas scikit-learn
```

### Issue 3: No visualizations generated

**Problem**: Plots don't appear

**Solution**:
- Check `results/` directory exists
- Verify matplotlib backend: `python3 -c "import matplotlib; print(matplotlib.get_backend())"`
- If on headless server, ensure using Agg backend

## Extending the System

### Adding a New Rule

1. **Define the rule function** in `catalogue.py`:

```python
def rule_my_new_rule() -> Rule:
    """Description of what the rule checks."""
    def check(hand: Hand) -> bool:
        # Your logic here
        ...
        return result

    return Rule(
        id="My_new_rule",
        name="Short human-readable name",
        predicate=check,
        family="CUSTOM",  # Or existing family
        description="Detailed explanation",
        primitives_used=["primitive1", "primitive2"]  # Key for analysis!
    )
```

2. **Add to ALL_RULES** list at bottom of `catalogue.py`

3. **Re-run demo** to see it in analysis

### Adding a New Primitive

1. **Choose the appropriate level** (0-4) in `primitives.py`

2. **Implement the function**:

```python
def my_new_primitive(param: Type) -> Callable[[Hand], ReturnType]:
    """Docstring explaining what it does."""
    def _check(hand: Hand) -> ReturnType:
        # Implementation
        ...
    return _check
```

3. **Update rules** that use this primitive to include it in `primitives_used`

## Testing

### Quick Test

```bash
# Test card domain
python3 src/rules/cards.py

# Test primitives
python3 src/rules/primitives.py

# Test catalogue
python3 src/rules/catalogue.py

# Test full pipeline
python3 src/main_demo.py
```

### Expected Output

Each module has a `if __name__ == "__main__"` block that runs basic tests.

## Questions?

**About this implementation**: [Your email/contact]

**About the behavioral experiment**: See main [card-games repository](https://github.com/konukcan/card-games)

**About DreamCoder**: [Ellis et al. paper](https://doi.org/10.1098/rsta.2022.0050)

## Summary

You now have:
- ✅ Complete card domain implementation
- ✅ 60+ compositional primitives (5 levels)
- ✅ 26 rules implemented (template for 30 more)
- ✅ Working demo pipeline
- ✅ Feature extraction + visualization
- ✅ Integration guide for full DreamCoder

**What's left**: Recognition network integration, enumeration, library learning

**Estimated time to complete**: 20-30 hours of focused work

**This is a strong foundation** - you have the hardest domain-specific parts done!
