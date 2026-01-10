# Deprecated Experiment Scripts (Archived)

These scripts were archived in January 2025 because they reference deprecated recognition models (`NeuralRecognitionModel` or `SetTransformerRecognitionModel`) that have been superseded by `ContrastiveRecognitionModel`.

**Current Primary Model**: `src/dreamcoder_core/contrastive_recognition.py`

---

## Archived Files

### 1. run_softmax_ablation.py

**Purpose**: Ablation study comparing NeuralRecognitionModel vs ContrastiveRecognitionModel with different output modes (sigmoid vs softmax).

**Why Archived**:
- Compares deprecated NeuralRecognitionModel against ContrastiveRecognitionModel
- The comparison is now moot since NeuralRecognitionModel is deprecated
- The key finding (softmax > sigmoid for search guidance) is now documented and applied

### 2. generate_prediction_comparison.py

**Purpose**: Generates PDF reports comparing primitive predictions between neural and contrastive models.

**Why Archived**:
- Relies on NeuralRecognitionModel which no longer exists
- Was used for one-time comparison; results are documented in `docs/`

### 3. run_overnight_set_transformer.py

**Purpose**: Overnight DreamCoder run using SetTransformerRecognitionModel.

**Why Archived**:
- SetTransformerRecognitionModel was deprecated due to embedding collapse issue
- See `archived/legacy_recognition/README.md` for detailed explanation

### 4. run_experimental_rules.py

**Purpose**: Run experiments on custom rule sets using SetTransformerRecognitionModel.

**Why Archived**:
- Same as above - uses deprecated SetTransformerRecognitionModel

---

## If You Need to Resurrect These Scripts

If you need to use these scripts again:

1. The deprecated models are preserved in `archived/legacy_recognition/`
2. You would need to add that directory to your Python path
3. Or update the scripts to use `ContrastiveRecognitionModel` instead

**Recommendation**: Update to use `ContrastiveRecognitionModel` rather than resurrecting the deprecated models.

---

*Archived: January 2025*
