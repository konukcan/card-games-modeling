# Archived Diagnostic Scripts

This directory contains diagnostic and debugging scripts created during active development of the recognition model (December 28-31, 2024). They are preserved for potential future debugging but are not part of the main experimental workflow.

## Categories

### Diagnose Scripts (15 files)
These scripts analyze specific components of the recognition model:

| Script | Purpose | Component Tested |
|--------|---------|------------------|
| `diagnose_baseline.py` | Baseline model behavior | NeuralRecognitionModel |
| `diagnose_contrastive_encoder.py` | Task encoding quality | ContrastiveRecognitionModel |
| `diagnose_task_encoder.py` | Multihead encoder analysis | TaskEncoder variants |
| `diagnose_encoder_layers.py` | Layer-by-layer analysis | Encoder architecture |
| `diagnose_embedding_head.py` | Embedding predictions | PrimitiveEmbedding head |
| `diagnose_embedding_head_v2.py` | Improved embedding analysis | PrimitiveEmbedding head |
| `diagnose_bias.py` | Bias evaluation | Model biases |
| `diagnose_neural_bias.py` | Neural network bias | Network weights |
| `diagnose_encodings.py` | Encoding quality | Card/Hand encodings |
| `diagnose_encodings_v2.py` | Improved encoding analysis | Card/Hand encodings |
| `diagnose_predictions.py` | Prediction analysis | Primitive predictions |
| `diagnose_training_data.py` | Training data inspection | Data pipeline |
| `diagnose_deep.py` | Deep model debugging | Deep architectures |

### Check Scripts (2 files)
Post-training verification scripts:

| Script | Purpose |
|--------|---------|
| `check_embedding_after_training.py` | Verify embeddings after training |
| `check_trained_predictions.py` | Verify predictions with trained weights |

### Test Scripts (3 files)
Integration tests for specific fixes:

| Script | Purpose | Date |
|--------|---------|------|
| `test_contextual_grammar_integration.py` | Grammar integration | Dec 31 |
| `test_contrastive_fix.py` | Contrastive bug fix | Dec 29 |
| `test_dreaming_integration.py` | Dreaming pipeline | Dec 28 |
| `test_embedding_fixes.py` | Embedding bug fixes | Dec 29 |
| `test_fixes_with_trained_weights.py` | Weight loading fixes | Dec 29 |

## When to Use These

These scripts may be useful when:
- Debugging similar issues in the recognition model
- Understanding how specific components were tested
- Reviving old debugging approaches for new problems

## Associated Modules

These scripts primarily test:
- `dreamcoder_core/contrastive_recognition.py`
- `dreamcoder_core/recognition_variants.py`
- `dreamcoder_core/neural_recognition.py`

## Note

These are **debugging tools**, not production experiments. For active experiments, see the parent `experiments/` directory.
