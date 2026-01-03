# Legacy Compression Module Backup

This directory contains the original monolithic `compression.py` file before it was refactored into the `compression/` package.

## Backup Details

- **Original location**: `src/dreamcoder_core/compression.py`
- **Backup date**: 2026-01-03
- **Original size**: 4,099 lines
- **MD5 checksum**: `ccbc236e93d3a7ad5264ef49218da4c0`

## Why This Was Refactored

The original file contained 4,000+ lines with 13+ distinct functional sections:
- Data structures
- Abstraction quality filtering
- Anti-unification
- Common subtree finding
- Arity-aware abstraction
- Program rewriting
- Semantic verification
- MDL scoring
- Multiple compression algorithms
- Recognition-guided compression
- 1,000+ lines of inline tests

## How to Restore

If the new modular structure causes issues:

```bash
# 1. Remove the new package
rm -rf src/dreamcoder_core/compression/

# 2. Restore the original file
cp archived/legacy_compression/compression_backup_*.py src/dreamcoder_core/compression.py

# 3. Verify MD5
md5 src/dreamcoder_core/compression.py
# Should match: ccbc236e93d3a7ad5264ef49218da4c0
```

## New Structure

The refactored code lives in `src/dreamcoder_core/compression/`:
```
compression/
├── __init__.py           # Re-exports all public symbols
├── data_structures.py    # SubtreeOccurrence, CompressionResult, CompressionState
├── quality_filters.py    # Abstraction quality checks
├── anti_unification.py   # LGG finding
├── subtree_finding.py    # Common subtree detection
├── arity_search.py       # Arity-aware factorization
├── rewriting.py          # Program rewriting with inventions
├── mdl_scoring.py        # MDL computation
├── compress.py           # Main compression functions
├── recognition_guided.py # DreamDecompiler integration
└── helpers.py            # Small utility functions
```

All existing imports should continue to work:
```python
from dreamcoder_core.compression import compress_frontiers  # Still works!
```
