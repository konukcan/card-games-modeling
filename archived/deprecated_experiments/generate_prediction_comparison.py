#!/usr/bin/env python3
"""
Generate Detailed Prediction Comparison PDF

Creates a comprehensive table comparing all three recognition models:
- Neural (softmax/CE)
- Contrastive Sigmoid (BCE)
- Contrastive Softmax (CE)

For each rule in both pretraining and catalogue sets, shows:
- Top-5 predicted primitives with probabilities
- Whether the rule was solved
- The actual primitives used in the solution (if solved)
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules

from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import create_all_rules as get_catalogue_rules


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("results/warmstart_experiment")

MODEL_CONFIGS = {
    'neural': {
        'name': 'Neural (GRU + Softmax)',
        'model_path': RESULTS_DIR / "neural_BOTH_20251225_145356" / "pretrained_recognition.pt",
        'results_path': RESULTS_DIR / "neural_BOTH_20251225_145356" / "results_WARM.json",
        'model_class': NeuralRecognitionModel,
        'color': colors.lightblue
    },
    'contrastive_sigmoid': {
        'name': 'Contrastive (Sigmoid)',
        'model_path': RESULTS_DIR / "contrastive_BOTH_20251225_145818" / "pretrained_recognition.pt",
        'results_path': RESULTS_DIR / "contrastive_BOTH_20251225_145818" / "results_WARM.json",
        'model_class': ContrastiveRecognitionModel,
        'color': colors.lightyellow
    },
    'contrastive_softmax': {
        'name': 'Contrastive (Softmax)',
        'model_path': RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "pretrained_recognition.pt",
        'results_path': RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "results_WARM.json",
        'model_class': ContrastiveRecognitionModel,
        'color': colors.lightgreen
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RulePrediction:
    """Prediction data for a single rule."""
    rule_name: str
    solved: bool
    top_primitives: List[Tuple[str, float]]  # (name, probability)
    solution_primitives: Optional[List[str]] = None
    programs_enumerated: Optional[int] = None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_key: str, grammar) -> Optional[object]:
    """Load a saved model."""
    config = MODEL_CONFIGS[model_key]
    model_path = config['model_path']

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        return None

    try:
        if model_key == 'neural':
            model = NeuralRecognitionModel(grammar)
        elif model_key == 'contrastive_sigmoid':
            # Use original hidden dimensions from the saved model
            model = ContrastiveRecognitionModel(
                grammar,
                card_hidden=128,  # Original dimensions
                card_out=64,
                pred_hidden=128,
                output_mode='sigmoid'
            )
        else:  # contrastive_softmax
            model = ContrastiveRecognitionModel(
                grammar,
                card_hidden=64,
                card_out=32,
                pred_hidden=64,
                output_mode='softmax'
            )

        model.load(str(model_path))
        print(f"Loaded {config['name']} from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        return None


def load_results(model_key: str) -> Dict:
    """Load results JSON for a model."""
    config = MODEL_CONFIGS[model_key]
    results_path = config['results_path']

    if not results_path.exists():
        print(f"Warning: Results not found at {results_path}")
        return {}

    with open(results_path) as f:
        return json.load(f)


# ============================================================================
# PREDICTION EXTRACTION
# ============================================================================

def get_predictions(model, task, top_k: int = 5) -> List[Tuple[str, float]]:
    """Get top-k primitive predictions for a task."""
    try:
        # Handle different model interfaces
        if isinstance(model, NeuralRecognitionModel):
            probs = model.predict_primitive_probs(task)
            prim_names = [str(p.program) for p in model.grammar.productions]
        else:  # ContrastiveRecognitionModel
            probs = model.predict_primitives(task)
            prim_names = model.primitive_names

        # Get top-k
        values, indices = torch.topk(probs, min(top_k, len(prim_names)))
        return [(prim_names[i], float(v)) for v, i in zip(values.cpu(), indices.cpu())]
    except Exception as e:
        print(f"Error getting predictions for {task.name}: {e}")
        return []


def extract_all_predictions(
    models: Dict[str, object],
    results: Dict[str, Dict],
    tasks: List,
    task_set: str  # 'pretraining' or 'catalogue'
) -> Dict[str, Dict[str, RulePrediction]]:
    """Extract predictions for all tasks from all models."""
    predictions = {key: {} for key in models.keys()}

    for task in tasks:
        for model_key, model in models.items():
            if model is None:
                continue

            # Get predictions
            top_preds = get_predictions(model, task)

            # Get solve status from results
            model_results = results.get(model_key, {})

            if task_set == 'pretraining':
                # For pretraining, check if task is in solved_tasks list
                pretraining = model_results.get('pretraining', {})
                solved_tasks = pretraining.get('solved_tasks', [])
                solved = task.name in solved_tasks
                solution_prims = None  # Not available for pretraining
                programs = None
            else:
                # For catalogue/main, use task_metrics
                main = model_results.get('main_training', {})
                task_metrics = main.get('task_metrics', {})
                task_result = task_metrics.get(task.name, {})
                solved = task_result.get('solved', False)
                solution_prims = task_result.get('primitives_used', None)
                programs = task_result.get('programs_enumerated', None)

            predictions[model_key][task.name] = RulePrediction(
                rule_name=task.name,
                solved=solved,
                top_primitives=top_preds,
                solution_primitives=solution_prims,
                programs_enumerated=programs
            )

    return predictions


# ============================================================================
# PDF GENERATION
# ============================================================================

def format_predictions(preds: List[Tuple[str, float]], max_len: int = 25) -> str:
    """Format predictions for display."""
    if not preds:
        return "N/A"

    lines = []
    for name, prob in preds[:5]:
        # Truncate long names
        if len(name) > max_len:
            name = name[:max_len-2] + ".."
        lines.append(f"{name}: {prob:.3f}")

    return "\n".join(lines)


def create_pdf(
    pretraining_preds: Dict[str, Dict[str, RulePrediction]],
    catalogue_preds: Dict[str, Dict[str, RulePrediction]],
    output_path: str
):
    """Generate the comparison PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        rightMargin=0.5*cm,
        leftMargin=0.5*cm,
        topMargin=0.5*cm,
        bottomMargin=0.5*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=12,
        alignment=TA_LEFT
    )
    cell_style = ParagraphStyle(
        'Cell',
        parent=styles['Normal'],
        fontSize=6,
        leading=7
    )

    elements = []

    # Title
    elements.append(Paragraph("Recognition Model Prediction Comparison", title_style))
    elements.append(Spacer(1, 0.3*cm))

    # Summary
    summary = """
    <b>Models Compared:</b><br/>
    • Neural (GRU + Softmax/CE): Bidirectional GRU encoder with attention, softmax output<br/>
    • Contrastive Sigmoid (BCE): Factored embeddings, τ = pos - neg encoding, sigmoid output<br/>
    • Contrastive Softmax (CE): Same as above but with softmax output<br/>
    <br/>
    <b>Legend:</b> ✓ = Solved, ✗ = Not solved. Top-5 predictions shown with probabilities.
    """
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 0.5*cm))

    # Helper to create table for a rule set
    def create_rule_table(predictions: Dict, title: str):
        elements.append(Paragraph(title, subtitle_style))
        elements.append(Spacer(1, 0.2*cm))

        # Get all rule names
        rule_names = sorted(list(next(iter(predictions.values())).keys()))

        # Header
        header = ['Rule', 'Neural\n(Softmax)', 'Contr. Sigmoid\n(BCE)', 'Contr. Softmax\n(CE)']

        data = [header]

        for rule_name in rule_names:
            row = [Paragraph(f"<b>{rule_name}</b>", cell_style)]

            for model_key in ['neural', 'contrastive_sigmoid', 'contrastive_softmax']:
                pred = predictions[model_key].get(rule_name)
                if pred is None:
                    row.append(Paragraph("N/A", cell_style))
                    continue

                status = "✓" if pred.solved else "✗"
                preds_str = format_predictions(pred.top_primitives)

                cell_text = f"<b>{status}</b>"
                if pred.programs_enumerated:
                    cell_text += f" ({pred.programs_enumerated:,} progs)"
                cell_text += f"<br/><font size=5>{preds_str.replace(chr(10), '<br/>')}</font>"

                if pred.solved and pred.solution_primitives:
                    sol_str = ", ".join(pred.solution_primitives[:5])
                    cell_text += f"<br/><font size=5 color='green'>Sol: {sol_str}</font>"

                row.append(Paragraph(cell_text, cell_style))

            data.append(row)

        # Create table
        col_widths = [3.5*cm, 7*cm, 7*cm, 7*cm]
        table = Table(data, colWidths=col_widths, repeatRows=1)

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ])

        # Color solved cells
        for i, rule_name in enumerate(rule_names, start=1):
            for j, model_key in enumerate(['neural', 'contrastive_sigmoid', 'contrastive_softmax'], start=1):
                pred = predictions[model_key].get(rule_name)
                if pred and pred.solved:
                    style.add('BACKGROUND', (j, i), (j, i), colors.Color(0.8, 1.0, 0.8))

        table.setStyle(style)
        elements.append(table)
        elements.append(PageBreak())

    # Create tables
    create_rule_table(pretraining_preds, "Pretraining Rules (44 rules)")
    create_rule_table(catalogue_preds, "Catalogue Rules (45 rules)")

    # Summary statistics
    elements.append(Paragraph("Summary Statistics", subtitle_style))
    elements.append(Spacer(1, 0.2*cm))

    summary_data = [
        ['Metric', 'Neural', 'Contr. Sigmoid', 'Contr. Softmax'],
    ]

    for task_set, preds in [('Pretraining', pretraining_preds), ('Catalogue', catalogue_preds)]:
        row = [f'{task_set} Solved']
        for model_key in ['neural', 'contrastive_sigmoid', 'contrastive_softmax']:
            if model_key in preds and preds[model_key]:
                solved = sum(1 for p in preds[model_key].values() if p.solved)
                total = len(preds[model_key])
                if total > 0:
                    row.append(f"{solved}/{total} ({100*solved/total:.1f}%)")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        summary_data.append(row)

    summary_table = Table(summary_data, colWidths=[4*cm, 5*cm, 5*cm, 5*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(summary_table)

    # Build PDF
    doc.build(elements)
    print(f"\nPDF saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("Generating Recognition Model Prediction Comparison")
    print("="*60)

    # Build grammar
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar.productions)} primitives")

    # Create tasks
    pretraining_rules = get_all_pretraining_rules()
    catalogue_rules = get_catalogue_rules()

    pretrain_tasks = create_tasks_from_rules(
        pretraining_rules, n_examples=100, n_holdout=20, hand_size=6
    )
    catalogue_tasks = create_tasks_from_rules(
        catalogue_rules, n_examples=100, n_holdout=20, hand_size=6
    )

    print(f"Pretraining tasks: {len(pretrain_tasks)}")
    print(f"Catalogue tasks: {len(catalogue_tasks)}")

    # Load models
    print("\nLoading models...")
    models = {}
    for key in MODEL_CONFIGS.keys():
        models[key] = load_model(key, grammar)

    # Load results
    print("\nLoading results...")
    results = {}
    for key in MODEL_CONFIGS.keys():
        results[key] = load_results(key)

    # Extract predictions
    print("\nExtracting predictions...")
    pretraining_preds = extract_all_predictions(models, results, pretrain_tasks, 'pretraining')

    # For catalogue, we need to check both pretraining and main results
    catalogue_preds = extract_all_predictions(models, results, catalogue_tasks, 'catalogue')

    # Generate PDF
    output_path = "results/recognition_model_comparison.pdf"
    print(f"\nGenerating PDF...")
    create_pdf(pretraining_preds, catalogue_preds, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
