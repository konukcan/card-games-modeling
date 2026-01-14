#!/usr/bin/env python3
"""
Evaluate Primitive Prediction Quality.

This module provides comprehensive metrics for evaluating how well a recognition model's
primitive predictions match the primitives actually needed to solve tasks.

Key insight: A good recognition model should rank primitives from actual solutions
higher than primitives not used in solutions.

Metrics:
- Recall@k: Fraction of solution primitives in top-k predictions
- Precision@k: Fraction of top-k predictions that are solution primitives
- MRR (Mean Reciprocal Rank): Average of 1/rank for solution primitives
- Log-likelihood Boost: How much the model upweights solution primitives vs uniform
- Enumeration Speedup: Actual programs/second improvement with guided search
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.program import Program
from rules.catalogue import create_all_rules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a single task's primitive predictions."""
    task_name: str
    solution_primitives: Set[str]
    n_solution_primitives: int

    # Recall@k metrics
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_15: float = 0.0

    # Precision@k metrics
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0

    # Ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    mean_rank: float = 0.0  # Mean rank of solution primitives

    # Probability metrics
    log_likelihood_boost: float = 0.0  # Log(P_model/P_uniform) for solution prims
    mean_solution_prob: float = 0.0
    mean_nonsolution_prob: float = 0.0

    # Top predictions
    top_5_predictions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['solution_primitives'] = list(d['solution_primitives'])
        return d


@dataclass
class ModelEvaluation:
    """Aggregate evaluation results for a model."""
    model_name: str
    config: Dict[str, Any]
    n_tasks_evaluated: int
    n_tasks_solved: int

    # Aggregate metrics (mean across tasks)
    mean_recall_at_5: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_recall_at_15: float = 0.0
    mean_precision_at_5: float = 0.0
    mean_mrr: float = 0.0
    mean_log_likelihood_boost: float = 0.0

    # Enumeration metrics
    uniform_solve_rate: float = 0.0
    guided_solve_rate: float = 0.0
    mean_speedup: float = 0.0

    # Per-task results
    task_metrics: List[PredictionMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict already converts nested dataclasses, but we need to handle sets
        for m in d['task_metrics']:
            if isinstance(m.get('solution_primitives'), set):
                m['solution_primitives'] = list(m['solution_primitives'])
        return d


def extract_primitives_from_program(program: Program) -> Set[str]:
    """Extract all primitive names used in a program."""
    primitives = set()

    def traverse(p):
        ptype = type(p).__name__
        if ptype == 'Primitive' and hasattr(p, 'name'):
            primitives.add(p.name)
        # Application nodes use 'f' and 'x' attributes
        if hasattr(p, 'f'):
            traverse(p.f)
        if hasattr(p, 'x'):
            traverse(p.x)
        # Abstraction uses 'body'
        if hasattr(p, 'body'):
            traverse(p.body)

    traverse(program)
    return primitives


def solve_task_with_timing(
    task,
    grammar: Grammar,
    timeout: float = 3.0,
    max_programs: int = 50000
) -> Tuple[Optional[Program], float, int]:
    """
    Solve a task and return (solution, time_taken, programs_enumerated).
    """
    enumerator = TopDownEnumerator(grammar, max_depth=15, max_programs=max_programs)
    start_time = time.time()
    programs_tried = 0

    for program, log_prob in enumerator.enumerate(task.request_type, timeout_seconds=timeout):
        programs_tried += 1
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break

        try:
            fn = program.evaluate([])
            correct = 0
            for inp, expected in task.examples:
                result = fn(inp)
                if result == expected:
                    correct += 1

            if correct == len(task.examples):
                return program, elapsed, programs_tried
        except Exception:
            continue

    return None, time.time() - start_time, programs_tried


def compute_prediction_metrics(
    predictions: np.ndarray,
    solution_primitives: Set[str],
    primitive_names: List[str]
) -> PredictionMetrics:
    """Compute comprehensive metrics for a single prediction."""
    n_primitives = len(primitive_names)
    n_solution = len(solution_primitives)

    # Get ranking by descending probability
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_names = [primitive_names[i] for i in sorted_indices]

    # Find ranks of solution primitives (1-indexed)
    solution_ranks = []
    for i, name in enumerate(sorted_names):
        if name in solution_primitives:
            solution_ranks.append(i + 1)

    # Recall@k
    def recall_at_k(k):
        if n_solution == 0:
            return 1.0
        top_k = set(sorted_names[:k])
        return len(solution_primitives & top_k) / n_solution

    # Precision@k
    def precision_at_k(k):
        top_k = set(sorted_names[:k])
        hits = len(solution_primitives & top_k)
        return hits / k

    # MRR
    if solution_ranks:
        mrr = np.mean([1.0 / r for r in solution_ranks])
        mean_rank = np.mean(solution_ranks)
    else:
        mrr = 0.0
        mean_rank = n_primitives

    # Log-likelihood boost
    # Compare model's probability for solution prims vs uniform (1/n_primitives)
    uniform_prob = 1.0 / n_primitives
    solution_indices = [primitive_names.index(p) for p in solution_primitives if p in primitive_names]
    if solution_indices:
        solution_probs = predictions[solution_indices]
        # Avoid log(0)
        solution_probs = np.clip(solution_probs, 1e-10, 1.0)
        log_boost = np.mean(np.log(solution_probs) - np.log(uniform_prob))
        mean_sol_prob = np.mean(solution_probs)
    else:
        log_boost = 0.0
        mean_sol_prob = 0.0

    # Non-solution primitive probabilities
    nonsolution_indices = [i for i in range(n_primitives) if primitive_names[i] not in solution_primitives]
    mean_nonsol_prob = np.mean(predictions[nonsolution_indices]) if nonsolution_indices else 0.0

    return PredictionMetrics(
        task_name="",  # Will be set by caller
        solution_primitives=solution_primitives,
        n_solution_primitives=n_solution,
        recall_at_5=recall_at_k(5),
        recall_at_10=recall_at_k(10),
        recall_at_15=recall_at_k(15),
        precision_at_5=precision_at_k(5),
        precision_at_10=precision_at_k(10),
        mrr=mrr,
        mean_rank=mean_rank,
        log_likelihood_boost=log_boost,
        mean_solution_prob=mean_sol_prob,
        mean_nonsolution_prob=mean_nonsol_prob,
        top_5_predictions=sorted_names[:5]
    )


def evaluate_model(
    model: ContrastiveRecognitionModel,
    tasks: List,
    grammar: Grammar,
    model_name: str,
    model_config: Dict[str, Any],
    solve_timeout: float = 3.0,
    n_tasks: int = 30
) -> ModelEvaluation:
    """
    Evaluate a model's primitive predictions against ground truth solutions.
    """
    logger.info(f"\nEvaluating: {model_name}")

    eval_tasks = tasks[:n_tasks]
    task_metrics = []

    # First, solve tasks with uniform grammar to get ground truth
    logger.info("  Solving tasks with uniform grammar...")
    solutions = {}
    for task in eval_tasks:
        program, time_taken, n_programs = solve_task_with_timing(
            task, grammar, timeout=solve_timeout
        )
        if program is not None:
            solutions[task.name] = {
                'program': program,
                'primitives': extract_primitives_from_program(program),
                'time': time_taken,
                'n_programs': n_programs
            }

    logger.info(f"  Solved {len(solutions)}/{len(eval_tasks)} tasks")

    # Evaluate predictions for solved tasks
    model.eval()
    with torch.no_grad():
        for task in eval_tasks:
            if task.name not in solutions:
                continue

            sol_info = solutions[task.name]
            pred = model.predict_primitives(task).cpu().numpy()

            metrics = compute_prediction_metrics(
                pred, sol_info['primitives'], model.primitive_names
            )
            metrics.task_name = task.name
            task_metrics.append(metrics)

    # Compute aggregate metrics
    if task_metrics:
        mean_recall_5 = np.mean([m.recall_at_5 for m in task_metrics])
        mean_recall_10 = np.mean([m.recall_at_10 for m in task_metrics])
        mean_recall_15 = np.mean([m.recall_at_15 for m in task_metrics])
        mean_precision_5 = np.mean([m.precision_at_5 for m in task_metrics])
        mean_mrr = np.mean([m.mrr for m in task_metrics])
        mean_log_boost = np.mean([m.log_likelihood_boost for m in task_metrics])
    else:
        mean_recall_5 = mean_recall_10 = mean_recall_15 = 0.0
        mean_precision_5 = mean_mrr = mean_log_boost = 0.0

    # Now test guided enumeration with this model
    logger.info("  Testing guided enumeration...")
    guided_solved = 0
    speedups = []

    for task in eval_tasks:
        if task.name not in solutions:
            continue

        # Get guided grammar
        guided_grammar = model.predict_grammar_weights(task)

        # Solve with guided grammar
        program, guided_time, guided_programs = solve_task_with_timing(
            task, guided_grammar, timeout=solve_timeout
        )

        if program is not None:
            guided_solved += 1
            uniform_time = solutions[task.name]['time']
            if guided_time > 0:
                speedup = uniform_time / guided_time
                speedups.append(speedup)

    mean_speedup = np.mean(speedups) if speedups else 1.0

    eval_result = ModelEvaluation(
        model_name=model_name,
        config=model_config,
        n_tasks_evaluated=len(eval_tasks),
        n_tasks_solved=len(solutions),
        mean_recall_at_5=mean_recall_5,
        mean_recall_at_10=mean_recall_10,
        mean_recall_at_15=mean_recall_15,
        mean_precision_at_5=mean_precision_5,
        mean_mrr=mean_mrr,
        mean_log_likelihood_boost=mean_log_boost,
        uniform_solve_rate=len(solutions) / len(eval_tasks),
        guided_solve_rate=guided_solved / len(solutions) if solutions else 0.0,
        mean_speedup=mean_speedup,
        task_metrics=task_metrics
    )

    # Log summary
    logger.info(f"  Results:")
    logger.info(f"    Recall@5: {mean_recall_5:.3f}")
    logger.info(f"    Recall@10: {mean_recall_10:.3f}")
    logger.info(f"    MRR: {mean_mrr:.3f}")
    logger.info(f"    Log-likelihood boost: {mean_log_boost:.3f}")
    logger.info(f"    Mean speedup: {mean_speedup:.2f}x")

    return eval_result


def create_model_variants() -> List[Tuple[str, Dict[str, Any]]]:
    """Define model variants to test."""
    return [
        ("baseline_no_norm", {
            "normalize_embeddings": False,
            "encoding_mode": "standard",
        }),
        ("standard_layernorm_scale10", {
            "normalize_embeddings": True,
            "embedding_scale": 10.0,
            "encoding_mode": "standard",
        }),
        ("standard_layernorm_scale20", {
            "normalize_embeddings": True,
            "embedding_scale": 20.0,
            "encoding_mode": "standard",
        }),
        ("standard_layernorm_scale50", {
            "normalize_embeddings": True,
            "embedding_scale": 50.0,
            "encoding_mode": "standard",
        }),
        ("random_contrast_scale20", {
            "normalize_embeddings": True,
            "embedding_scale": 20.0,
            "encoding_mode": "random_contrast",
            "n_random_hands": 10,
            "lambda_random": 0.5,
        }),
        ("triple_contrast_scale20", {
            "normalize_embeddings": True,
            "embedding_scale": 20.0,
            "encoding_mode": "triple_contrast",
            "n_random_hands": 10,
        }),
    ]


def pretrain_model(
    model: ContrastiveRecognitionModel,
    tasks: List,
    grammar: Grammar,
    n_tasks: int = 20,
    epochs: int = 30
) -> float:
    """Pre-train model on solved tasks."""
    training_tasks = tasks[:n_tasks]

    # Solve tasks
    frontiers = {}
    for task in training_tasks:
        program, _, _ = solve_task_with_timing(task, grammar, timeout=2.0)
        if program is not None:
            frontiers[task.name] = type('Frontier', (), {
                'solved': True,
                'entries': [type('Entry', (), {
                    'program': program,
                    'log_likelihood': 0.0
                })()]
            })()

    # Train
    if frontiers:
        loss = model.train_on_frontiers(
            tasks=training_tasks,
            frontiers=frontiers,
            epochs=epochs,
            batch_size=8
        )
        return loss
    return 0.0


def main():
    print("=" * 70)
    print("PRIMITIVE PREDICTION QUALITY EVALUATION")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/prediction_quality_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load grammar and tasks
    logger.info("\nLoading grammar and tasks...")
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    logger.info(f"  Grammar: {len(grammar.productions)} primitives")
    logger.info(f"  Tasks: {len(tasks)}")

    # Get model variants
    variants = create_model_variants()
    all_results = []

    for name, config in variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training and evaluating: {name}")
        logger.info(f"{'='*60}")

        # Create model
        model_kwargs = {
            'grammar': grammar,
            'card_hidden': 128,
            'card_out': 64,
            'pred_hidden': 128,
        }
        model_kwargs.update(config)

        try:
            model = ContrastiveRecognitionModel(**model_kwargs)

            # Pre-train
            logger.info("  Pre-training...")
            loss = pretrain_model(model, tasks, grammar, n_tasks=25, epochs=30)
            logger.info(f"  Final training loss: {loss:.4f}")

            # Evaluate
            eval_result = evaluate_model(
                model=model,
                tasks=tasks,
                grammar=grammar,
                model_name=name,
                model_config=config,
                solve_timeout=3.0,
                n_tasks=35
            )
            all_results.append(eval_result)

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\n{'Model':<30} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'LogBoost':<10} {'Speedup':<8}")
    print("-" * 90)

    for r in all_results:
        print(f"{r.model_name:<30} {r.mean_recall_at_5:<8.3f} {r.mean_recall_at_10:<8.3f} "
              f"{r.mean_mrr:<8.3f} {r.mean_log_likelihood_boost:<10.3f} {r.mean_speedup:<8.2f}x")

    # Save results
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Find best model
    if all_results:
        best_by_recall = max(all_results, key=lambda x: x.mean_recall_at_5)
        best_by_mrr = max(all_results, key=lambda x: x.mean_mrr)
        best_by_speedup = max(all_results, key=lambda x: x.mean_speedup)

        print("\n" + "=" * 70)
        print("BEST MODELS:")
        print(f"  By Recall@5: {best_by_recall.model_name} ({best_by_recall.mean_recall_at_5:.3f})")
        print(f"  By MRR: {best_by_mrr.model_name} ({best_by_mrr.mean_mrr:.3f})")
        print(f"  By Speedup: {best_by_speedup.model_name} ({best_by_speedup.mean_speedup:.2f}x)")
        print("=" * 70)


if __name__ == "__main__":
    main()
