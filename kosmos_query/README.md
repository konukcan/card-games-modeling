# Kosmos Query Package: Recognition Model for Card Game Rule Synthesis

## Purpose

This folder contains curated materials for submitting a query to Kosmos AI about designing a neural recognition model for a DreamCoder-style program synthesis system.

## The Problem in Brief

We have a working program synthesis system that learns compositional rules for a card game domain. The system has three phases:

1. **Enumeration**: Best-first search through program space (works well)
2. **Recognition**: Neural model predicts useful primitives to guide search (BROKEN)
3. **Compression**: Extract reusable abstractions (works well)

**The recognition model fails to discriminate between different tasks.** All tasks get nearly identical embeddings and primitive predictions, providing no speedup to enumeration.

## Files in This Package

| File | Description |
|------|-------------|
| `01_problem_statement.md` | Detailed problem description, constraints, and research questions |
| `02_dsl_primitives.py` | Complete DSL with ~60 primitives, type signatures, and examples |
| `03_rules_catalogue.md` | Description of 56 rules organized by family with examples |
| `04_sample_dataset.json` | Training data for 8 representative rules with 10 examples each |
| `05_failed_approaches.md` | What we've tried and why it didn't work |
| `06_cards.py` | Card domain implementation (Suit, Rank, Card, Hand types) |
| `07_current_architecture.md` | Current neural architecture with code snippets |
| `README.md` | This file |

## Key Technical Details

### Task Structure
- **Input**: Ordered list of 6 playing cards (suit + rank + position)
- **Output**: Boolean classification (True/False)
- **Training signal**: ~20 labeled examples per task

### The Core Challenge
Different rules attend to different feature subspaces:
- "Halves have same suits" → position + suit (ignore rank)
- "Is sorted" → position + rank (ignore suit)
- "At least 3 hearts" → suit only (ignore position)

The model must learn **task-specific feature relevance** from labeled examples alone.

### Why Standard Approaches Fail
- In original DreamCoder (list tasks), the OUTPUT is informative (e.g., `[1,2,3] → [6]`)
- In classification, the OUTPUT is just True/False (2 bits)
- Standard architectures can't learn *why* some hands are True and others False

## Deliverables Requested

1. **Analysis** of why standard recognition models fail for classification tasks
2. **Proposed architecture** with specific recommendations for:
   - Card encoding scheme
   - Example set aggregation method
   - Positive/negative example contrast mechanism
   - Output head structure
3. **Reference implementations** or closely related architectures
4. **Recommended experiments** to validate the architecture
5. **Discussion** of cognitive modeling implications

## Constraints

- Must be neuro-symbolic (neural recognition + symbolic enumeration)
- Should exhibit human-like sample efficiency
- Must interface with existing DreamCoder infrastructure
- Output: log-probabilities over ~60 grammar primitives

## Usage

Upload this entire folder to Kosmos AI along with the main query text.

---

*Package created: December 2024*
*Project: DreamCoder Card Game Modeling*
