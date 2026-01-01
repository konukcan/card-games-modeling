# Feature Status Reference

This document tracks the implementation status of all major features in the DreamCoder card game modeling system.

**Last Updated**: January 2025

---

## Legend

| Status | Meaning |
|--------|---------|
| WORKING | Fully implemented, tested, and in production use |
| EXPERIMENTAL | Implemented but under active development/tuning |
| PARTIAL | Core functionality works, some features incomplete |
| PLANNED | Designed but not yet implemented |
| DEPRECATED | No longer maintained, use alternative |

---

## Core DreamCoder Features

### Program Synthesis

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Top-down enumeration | WORKING | `enumeration.py` | Priority queue based |
| Memoized enumeration | WORKING | `enumeration.py:enumerate_memoized()` | **1000x+ speedup** - always use this |
| Best-first search | WORKING | `enumeration.py` | Cost-ordered by -log_prob |
| Type-guided synthesis | WORKING | `type_system.py`, `enumeration.py` | Hindley-Milner inference |
| Depth limiting | WORKING | `enumeration.py` | Configurable max_depth |
| Budget limiting | WORKING | `enumeration.py` | max_programs parameter |
| Timeout handling | WORKING | `enumeration.py` | Graceful timeout |
| Parallel enumeration | WORKING | `enumeration_worker.py` | PyPy workers for speed |

### Recognition Model (ContrastiveRecognitionModel - PRIMARY)

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Factored card embeddings | WORKING | `contrastive_recognition.py` | suit + rank + position embeddings |
| Contrastive task encoding | WORKING | `contrastive_recognition.py` | τ = mean(pos) - mean(neg) |
| Sigmoid output | WORKING | `contrastive_recognition.py` | Independent per-primitive probabilities |
| Softmax output | WORKING | `contrastive_recognition.py` | Use `output_mode='softmax'` for search ranking |
| Count head | WORKING | `contrastive_recognition.py` | Predicts number of primitives |
| Bigram head | WORKING | `contrastive_recognition.py` | Predicts primitive co-occurrence |
| Structural similarity loss | WORKING | `contrastive_recognition.py` | Tasks with similar primitives cluster |
| Training loop | WORKING | `contrastive_recognition.py` | Adam optimizer with multiple losses |
| Embedding variants | EXPERIMENTAL | `recognition_variants.py` | EnhancedCardEncoder, BigramHead, etc. |

#### Archived Recognition Models (in `archived/legacy_recognition/`)
| Feature | Why Archived |
|---------|--------------|
| BiGRU encoder | Sequential processing suboptimal for cards; limited transfer results |
| Set Transformer | Embedding collapse problem - all tasks had cosine similarity ≈ 1.0 |

### Library Learning

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Fragment extraction | WORKING | `compression.py` | From solved frontiers |
| MDL scoring | WORKING | `compression.py` | Minimum description length |
| Grammar update | WORKING | `grammar.py` | Adds learned primitives |
| Abstraction naming | WORKING | `compression.py` | Auto-generated names |
| Refactoring check | PARTIAL | `compression.py` | Basic implementation |

### Wake-Sleep Loop

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Wake phase (enumeration) | WORKING | `wake_sleep.py` | Guided by recognition |
| Sleep phase (recognition training) | WORKING | `wake_sleep.py` | On solved tasks |
| Dream phase (synthetic tasks) | PARTIAL | `wake_sleep.py` | Basic implementation |
| Compression phase | WORKING | `wake_sleep.py` | Library learning |
| Contrastive wake-sleep | WORKING | `contrastive_wake_sleep.py` | Uses ContrastiveRecognitionModel |
| Multi-iteration training | WORKING | Various runners | Configurable iterations |
| Curriculum learning | PARTIAL | Various runners | Phase-based progression |

---

## Card Game Domain Features

### Rule Representation

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| 45 core rules | WORKING | `rules/catalogue.py` | Main experimental set |
| 44 pretraining rules | WORKING | `rules/pretraining_rules.py` | Alternative rule set |
| Rule difficulty classification | WORKING | `docs/rule_difficulty_classification.md` | Easy/Medium/Hard/Very Hard |
| Hand generation | WORKING | `rules/catalogue.py` | Positive/negative examples |
| Holdout verification | WORKING | `rules/catalogue.py` | Prevents spurious solutions |

### Primitives (60 total in 5 levels)

| Level | Feature | Status | Count |
|-------|---------|--------|-------|
| Level 0 | Constants (TRUE, FALSE, 0-13, suits) | WORKING | 18 |
| Level 1 | Basic operations (eq, lt, add, etc.) | WORKING | 12 |
| Level 2 | Card accessors (get_rank, get_suit, get_color) | WORKING | 8 |
| Level 3 | List operations (map, filter, all, any, etc.) | WORKING | 14 |
| Level 4 | Aggregate operations (count, sum, unique, etc.) | WORKING | 8 |

---

## Experiment Infrastructure

### Reporting

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| HTML report generation | WORKING | `generate_systematic_report.py` | Comprehensive analysis |
| Training curves | WORKING | `visualization.py` | Matplotlib plots |
| Per-iteration metrics | WORKING | Result JSON files | Loss, accuracy, solve rate |
| Rule family analysis | WORKING | Report generation | Groups by difficulty |
| Performance benchmarks | WORKING | `benchmark_enumeration.py` | Programs/second metrics |

### Logging & Checkpoints

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Per-iteration logging | WORKING | All runners | Detailed metrics |
| Model checkpoints | WORKING | `results_*/models/` | PyTorch state dicts |
| Result JSON export | WORKING | All runners | Full experiment data |
| Log organization | WORKING | `src/logs/` | By experiment type |

---

## Experimental Features (Active Development)

### Contrastive Recognition (December 2024 - Present)

| Feature | Status | Notes |
|---------|--------|-------|
| Contrastive loss | WORKING | InfoNCE-style |
| Positive pair sampling | WORKING | Same-rule tasks |
| Negative pair sampling | EXPERIMENTAL | Random vs. hard negatives |
| Temperature scaling | EXPERIMENTAL | Tuning ongoing |
| Softmax normalization | EXPERIMENTAL | vs. sigmoid baseline |
| Wake-sleep integration | EXPERIMENTAL | `contrastive_wake_sleep.py` |

### Set Transformer Recognition

| Feature | Status | Notes |
|---------|--------|-------|
| Self-attention encoder | WORKING | Permutation equivariant |
| Pooling by attention | WORKING | Aggregates card features |
| Multi-head architecture | WORKING | Configurable heads |
| Wake-sleep integration | EXPERIMENTAL | `run_overnight_set_transformer.py` |

### Interpretability Tools

| Feature | Status | Notes |
|---------|--------|-------|
| Feature importance | WORKING | `interpretability.py` |
| Attention visualization | WORKING | `interpretability.py` |
| Embedding analysis | PARTIAL | PCA/clustering |
| Prediction explanation | PARTIAL | Per-task analysis |

---

## Deprecated / Removed Features

| Feature | Reason | Alternative |
|---------|--------|-------------|
| Cython enumeration | Pickle serialization issues | PyPy workers |
| Cost-banding enumeration | 6x slower than memoized | `enumerate_memoized()` |
| Legacy GRU (unidirectional) | Lower accuracy | BiGRU with attention |
| Old primitive set | Inconsistent naming | `lean_primitives.py` |
| Shell script launchers | Fragile, hard to maintain | Direct Python execution |

---

## Planned Features (Not Yet Implemented)

| Feature | Priority | Complexity | Notes |
|---------|----------|------------|-------|
| Transfer analysis | HIGH | MEDIUM | Measure cross-domain generalization |
| Interactive exploration | MEDIUM | LOW | Web UI for results browsing |
| Hierarchical abstractions | MEDIUM | HIGH | Multi-level library learning |
| Natural language guidance | LOW | HIGH | LAPS integration (paused) |
| Distributed training | LOW | HIGH | Multi-machine parallelism |

---

## Performance Characteristics

### Enumeration Speed (with memoization)

| Depth | Programs/Second | Budget (50K) Time |
|-------|-----------------|-------------------|
| 6 | ~300,000 | < 1 second |
| 7 | ~70,000 | < 1 second |
| 8 | ~200,000 | < 1 second |
| 10+ | ~50,000 | 1-2 seconds |

### Training Speed

| Model | Time per Epoch | Notes |
|-------|----------------|-------|
| BiGRU Recognition | ~2-5 seconds | On 45 tasks |
| Set Transformer | ~5-10 seconds | More parameters |
| Contrastive | ~3-6 seconds | Additional loss terms |

### Overnight Run Characteristics

| Phase | Typical Duration | Tasks Solved |
|-------|------------------|--------------|
| Phase 1 (Easy) | 30-60 min | 6-8 / 8 |
| Phase 2 (Medium) | 1-2 hours | 10-12 / 12 |
| Phase 3 (Hard) | 2-4 hours | 8-12 / 18 |
| Phase 4 (Very Hard) | 2-4 hours | 2-5 / 7 |
| **Total** | **6-10 hours** | **26-37 / 45** |
