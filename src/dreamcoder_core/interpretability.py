#!/usr/bin/env python3
"""
Mechanistic Interpretability Tools for Neural Recognition Model

This module provides tools to understand what the neural recognition model
has learned and how it makes predictions:

1. Gradient-based Attribution (LRP-like):
   - Which input features drive predictions for each primitive?
   - Which examples are most informative for a prediction?

2. Embedding Visualization (UMAP/t-SNE):
   - How are tasks clustered in embedding space?
   - Which rules does the model "see" as similar?

3. Probing Classifiers:
   - What features are linearly decodable from embeddings?
   - Can we predict rule family, difficulty, etc. from embeddings?

4. Primitive Correlation Analysis:
   - Which primitives co-activate?
   - Are there learned "primitive modules"?
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# GRADIENT-BASED ATTRIBUTION
# ============================================================================

@dataclass
class AttributionResult:
    """Result of attribution analysis for a single task-primitive pair."""
    task_name: str
    primitive_name: str
    primitive_log_prob: float

    # Per-example attributions
    example_scores: List[float]  # How important is each example?
    most_important_example_idx: int

    # Per-feature attributions (for most important example)
    card_scores: List[float]  # How important is each card position?
    feature_scores: Dict[str, float]  # Aggregated feature importances

    # Overall confidence
    total_attribution: float


class GradientAttribution:
    """
    Gradient-based attribution for understanding model predictions.

    Uses integrated gradients / input x gradient to compute feature importance.
    This is similar to LRP (Layer-wise Relevance Propagation) in spirit.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def attribute_primitive(
        self,
        task,
        primitive_name: str,
        n_steps: int = 20
    ) -> AttributionResult:
        """
        Compute attribution for why the model predicts a specific primitive.

        Uses integrated gradients: attribute importance by integrating
        gradients along a path from baseline to input.

        Args:
            task: Task to analyze
            primitive_name: Which primitive to explain
            n_steps: Number of integration steps

        Returns:
            AttributionResult with feature importances
        """
        self.model.eval()

        # Check if primitive exists
        if primitive_name not in self.model.primitive_to_idx:
            raise ValueError(f"Unknown primitive: {primitive_name}")

        prim_idx = self.model.primitive_to_idx[primitive_name]

        # Encode examples individually to get per-example gradients
        example_scores = []
        all_card_scores = []

        for ex_idx, (inp, out) in enumerate(task.examples[:self.model.max_examples]):
            # Create input tensor
            input_features = torch.zeros(
                1, self.model.max_cards, 24,
                requires_grad=True,
                device=self.model.device
            )

            # Fill in actual card features
            from dreamcoder_core.neural_recognition import encode_hand, encode_output
            hand_features = encode_hand(inp, self.model.max_cards)
            input_features.data[0] = hand_features.to(self.model.device)

            output_features = encode_output(out).unsqueeze(0).to(self.model.device)

            # Baseline: zero features
            baseline = torch.zeros_like(input_features)

            # Integrated gradients
            total_grad = torch.zeros_like(input_features)

            for step in range(n_steps):
                alpha = step / n_steps
                interpolated = baseline + alpha * (input_features - baseline)
                interpolated = interpolated.clone().detach().requires_grad_(True)

                # Forward pass
                example_enc = self.model.example_encoder(interpolated, output_features)
                task_enc = self.model.task_encoder(example_enc.unsqueeze(0))
                log_probs = self.model.primitive_predictor(task_enc)

                # Get gradient for target primitive
                target_prob = log_probs[0, prim_idx]

                # Compute gradient with respect to input
                grad = torch.autograd.grad(target_prob, interpolated, retain_graph=False)[0]
                total_grad += grad.detach()

            # Compute attribution
            attribution = (input_features - baseline).detach() * total_grad / n_steps

            # Sum across feature dimension for per-card score
            card_scores = attribution.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
            all_card_scores.append(card_scores)

            # Example importance is sum of all card importances
            example_score = float(card_scores.sum())
            example_scores.append(example_score)

        # Find most important example
        most_important_idx = int(np.argmax(example_scores))

        # Get feature breakdown for most important example
        feature_scores = self._aggregate_feature_scores(
            task.examples[most_important_idx][0],
            all_card_scores[most_important_idx]
        )

        # Get primitive log prob
        with torch.no_grad():
            log_probs = self.model.predict_primitive_probs(task)
            prim_log_prob = float(log_probs[prim_idx].cpu())

        return AttributionResult(
            task_name=task.name,
            primitive_name=primitive_name,
            primitive_log_prob=prim_log_prob,
            example_scores=example_scores,
            most_important_example_idx=most_important_idx,
            card_scores=all_card_scores[most_important_idx].tolist(),
            feature_scores=feature_scores,
            total_attribution=sum(example_scores)
        )

    def _aggregate_feature_scores(
        self,
        hand,
        card_scores: np.ndarray
    ) -> Dict[str, float]:
        """Aggregate per-card scores into semantic feature scores."""
        from dreamcoder_core.neural_recognition import extract_card_features
        from rules.cards import card_color

        feature_scores = defaultdict(float)

        for i, (card, score) in enumerate(zip(hand, card_scores)):
            cf = extract_card_features(card)

            # Attribute to semantic features
            feature_scores[f'suit_{card.suit.name}'] += score
            feature_scores[f'color_{card_color(card).name}'] += score
            feature_scores[f'rank_{card.rank.name}'] += score
            feature_scores[f'position_{i}'] += score

            if cf.is_face:
                feature_scores['is_face_card'] += score
            if cf.is_ace:
                feature_scores['is_ace'] += score
            if cf.rank_value % 2 == 0:
                feature_scores['even_rank'] += score
            else:
                feature_scores['odd_rank'] += score

        return dict(feature_scores)


# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

@dataclass
class EmbeddingAnalysis:
    """Result of embedding space analysis."""
    task_names: List[str]
    task_families: List[str]
    embeddings_2d: np.ndarray  # (n_tasks, 2) for visualization
    cluster_labels: Optional[np.ndarray]  # Discovered clusters
    silhouette_score: float  # Cluster quality

    # Family-based analysis
    family_centroids: Dict[str, np.ndarray]
    family_similarities: Dict[Tuple[str, str], float]


class EmbeddingVisualizer:
    """
    Visualize and analyze task embeddings from the recognition model.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_embeddings(self, tasks: List) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute embeddings for a list of tasks.

        Returns:
            embeddings: (n_tasks, hidden_dim) numpy array
            task_names: List of task names
            task_families: List of task families
        """
        embeddings = []
        task_names = []
        task_families = []

        self.model.eval()
        with torch.no_grad():
            for task in tasks:
                emb = self.model.get_task_embedding(task)
                embeddings.append(emb.cpu().numpy())
                task_names.append(task.name)
                task_families.append(getattr(task, 'family', 'unknown'))

        return np.stack(embeddings), task_names, task_families

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'umap',
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.

        Args:
            embeddings: (n_tasks, hidden_dim)
            method: 'umap', 'tsne', or 'pca'
            n_components: Output dimensions (usually 2)

        Returns:
            Reduced embeddings: (n_tasks, n_components)
        """
        if method == 'pca':
            # Simple PCA without sklearn dependency
            mean = embeddings.mean(axis=0)
            centered = embeddings - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            top_components = eigenvectors[:, idx[:n_components]]
            return centered @ top_components

        elif method == 'tsne':
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
                return tsne.fit_transform(embeddings)
            except ImportError:
                print("sklearn not available, falling back to PCA")
                return self.reduce_dimensions(embeddings, 'pca', n_components)

        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                return reducer.fit_transform(embeddings)
            except ImportError:
                print("UMAP not available, falling back to t-SNE")
                return self.reduce_dimensions(embeddings, 'tsne', n_components)

        else:
            raise ValueError(f"Unknown method: {method}")

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Cluster embeddings to discover task groupings.

        Returns:
            cluster_labels: (n_tasks,) cluster assignments
            silhouette_score: Cluster quality metric
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            if n_clusters is None:
                # Try to find optimal number of clusters
                best_score = -1
                best_labels = None
                best_k = 2

                for k in range(2, min(10, len(embeddings) // 2)):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_k = k

                return best_labels, best_score

            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                return labels, score

        except ImportError:
            # Fallback: no clustering
            return np.zeros(len(embeddings), dtype=int), 0.0

    def analyze(
        self,
        tasks: List,
        reduction_method: str = 'pca'
    ) -> EmbeddingAnalysis:
        """
        Full embedding analysis.

        Returns:
            EmbeddingAnalysis with visualizable results
        """
        # Compute embeddings
        embeddings, task_names, task_families = self.compute_embeddings(tasks)

        # Reduce for visualization
        embeddings_2d = self.reduce_dimensions(embeddings, reduction_method)

        # Cluster
        cluster_labels, silhouette = self.cluster_embeddings(embeddings)

        # Compute family centroids
        family_centroids = {}
        unique_families = set(task_families)

        for family in unique_families:
            family_mask = [f == family for f in task_families]
            family_embeddings = embeddings[family_mask]
            if len(family_embeddings) > 0:
                family_centroids[family] = family_embeddings.mean(axis=0)

        # Compute family similarities
        family_similarities = {}
        families_list = list(unique_families)

        for i, f1 in enumerate(families_list):
            for f2 in families_list[i+1:]:
                if f1 in family_centroids and f2 in family_centroids:
                    sim = np.dot(family_centroids[f1], family_centroids[f2]) / (
                        np.linalg.norm(family_centroids[f1]) *
                        np.linalg.norm(family_centroids[f2]) + 1e-8
                    )
                    family_similarities[(f1, f2)] = float(sim)

        return EmbeddingAnalysis(
            task_names=task_names,
            task_families=task_families,
            embeddings_2d=embeddings_2d,
            cluster_labels=cluster_labels,
            silhouette_score=silhouette,
            family_centroids=family_centroids,
            family_similarities=family_similarities
        )


# ============================================================================
# PROBING CLASSIFIERS
# ============================================================================

@dataclass
class ProbingResult:
    """Result of probing classifier analysis."""
    property_name: str
    accuracy: float
    baseline_accuracy: float  # Majority class baseline
    improvement: float  # accuracy - baseline

    # Per-class breakdown
    class_accuracies: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]


class ProbingClassifier:
    """
    Train simple classifiers on embeddings to probe what information is encoded.

    The idea: if a linear classifier can predict property X from embeddings,
    then the model has learned to encode X.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def probe(
        self,
        tasks: List,
        property_fn: Callable,
        property_name: str,
        test_fraction: float = 0.2
    ) -> ProbingResult:
        """
        Probe if embeddings encode a specific property.

        Args:
            tasks: List of tasks
            property_fn: Function task -> property value
            property_name: Name of property being probed
            test_fraction: Fraction of data for testing

        Returns:
            ProbingResult with accuracy metrics
        """
        # Get embeddings and labels
        self.model.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for task in tasks:
                emb = self.model.get_task_embedding(task)
                embeddings.append(emb.cpu().numpy())
                labels.append(property_fn(task))

        X = np.stack(embeddings)
        y = np.array(labels)

        # Encode labels
        unique_labels = list(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y_encoded = np.array([label_to_idx[l] for l in labels])

        # Split
        n_test = max(1, int(len(X) * test_fraction))
        indices = np.random.permutation(len(X))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Train linear classifier
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Baseline: majority class
            from collections import Counter
            majority_class = Counter(y_train).most_common(1)[0][0]
            baseline_accuracy = (y_test == majority_class).mean()

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Per-class accuracy
            class_accuracies = {}
            for i, label in enumerate(unique_labels):
                mask = y_test == i
                if mask.sum() > 0:
                    class_accuracies[str(label)] = float((y_pred[mask] == i).mean())

        except ImportError:
            # Fallback: simple nearest centroid
            accuracy, baseline_accuracy = self._simple_probe(X_train, y_train, X_test, y_test)
            cm = None
            class_accuracies = {}

        return ProbingResult(
            property_name=property_name,
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            improvement=accuracy - baseline_accuracy,
            class_accuracies=class_accuracies,
            confusion_matrix=cm
        )

    def _simple_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """Simple nearest centroid classifier as fallback."""
        # Compute centroids
        unique_classes = np.unique(y_train)
        centroids = {}
        for c in unique_classes:
            centroids[c] = X_train[y_train == c].mean(axis=0)

        # Predict
        correct = 0
        for i, x in enumerate(X_test):
            best_class = min(
                centroids.keys(),
                key=lambda c: np.linalg.norm(x - centroids[c])
            )
            if best_class == y_test[i]:
                correct += 1

        accuracy = correct / len(X_test)

        # Baseline
        from collections import Counter
        majority = Counter(y_train).most_common(1)[0][0]
        baseline = (y_test == majority).mean()

        return accuracy, baseline

    def probe_standard_properties(self, tasks: List) -> Dict[str, ProbingResult]:
        """
        Probe for standard task properties.

        Returns:
            Dict mapping property name to ProbingResult
        """
        results = {}

        # Family
        if any(hasattr(t, 'family') for t in tasks):
            results['family'] = self.probe(
                tasks,
                lambda t: getattr(t, 'family', 'unknown'),
                'family'
            )

        # Difficulty level
        if any(hasattr(t, 'difficulty_level') for t in tasks):
            results['difficulty'] = self.probe(
                tasks,
                lambda t: getattr(t, 'difficulty_level', 0),
                'difficulty'
            )

        # Example balance
        def get_balance(task):
            n_pos = sum(1 for _, out in task.examples if out == True)
            return 'balanced' if 0.3 < n_pos / len(task.examples) < 0.7 else 'imbalanced'

        results['balance'] = self.probe(tasks, get_balance, 'balance')

        return results


# ============================================================================
# PRIMITIVE CORRELATION ANALYSIS
# ============================================================================

@dataclass
class PrimitiveCorrelations:
    """Analysis of primitive co-activation patterns."""
    primitive_names: List[str]
    correlation_matrix: np.ndarray  # (n_prims, n_prims) correlations
    clusters: Dict[int, List[str]]  # Discovered primitive clusters
    top_correlations: List[Tuple[str, str, float]]  # Top positive correlations
    top_anticorrelations: List[Tuple[str, str, float]]  # Top negative correlations


class PrimitiveAnalyzer:
    """
    Analyze relationships between primitive predictions.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_correlations(self, tasks: List) -> PrimitiveCorrelations:
        """
        Compute correlation matrix between primitive predictions.

        Returns:
            PrimitiveCorrelations with detailed analysis
        """
        self.model.eval()

        # Collect predictions for all tasks
        all_predictions = []

        with torch.no_grad():
            for task in tasks:
                log_probs = self.model.predict_primitive_probs(task)
                all_predictions.append(log_probs.cpu().numpy())

        predictions = np.stack(all_predictions)  # (n_tasks, n_prims)

        # Compute correlation matrix
        # Normalize each primitive's predictions
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0) + 1e-8
        normalized = (predictions - mean) / std

        corr_matrix = np.corrcoef(normalized.T)

        # Find clusters of correlated primitives
        try:
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='precomputed',
                linkage='average'
            )

            # Convert correlation to distance
            distance = 1 - np.abs(corr_matrix)
            np.fill_diagonal(distance, 0)

            labels = clustering.fit_predict(distance)

            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(self.model.primitive_names[i])
            clusters = dict(clusters)

        except ImportError:
            clusters = {0: self.model.primitive_names}

        # Find top correlations
        n_prims = len(self.model.primitive_names)
        correlations = []

        for i in range(n_prims):
            for j in range(i + 1, n_prims):
                correlations.append((
                    self.model.primitive_names[i],
                    self.model.primitive_names[j],
                    corr_matrix[i, j]
                ))

        correlations.sort(key=lambda x: -x[2])
        top_positive = correlations[:10]

        correlations.sort(key=lambda x: x[2])
        top_negative = correlations[:10]

        return PrimitiveCorrelations(
            primitive_names=self.model.primitive_names,
            correlation_matrix=corr_matrix,
            clusters=clusters,
            top_correlations=top_positive,
            top_anticorrelations=top_negative
        )


# ============================================================================
# COMPREHENSIVE INTERPRETABILITY REPORT
# ============================================================================

@dataclass
class InterpretabilityReport:
    """Comprehensive interpretability analysis."""
    model_info: Dict
    embedding_analysis: Optional[EmbeddingAnalysis]
    probing_results: Dict[str, ProbingResult]
    primitive_correlations: Optional[PrimitiveCorrelations]
    sample_attributions: List[AttributionResult]


class InterpretabilityAnalyzer:
    """
    Run comprehensive interpretability analysis on a trained recognition model.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_attr = GradientAttribution(model)
        self.embedding_viz = EmbeddingVisualizer(model)
        self.prober = ProbingClassifier(model)
        self.prim_analyzer = PrimitiveAnalyzer(model)

    def full_analysis(
        self,
        tasks: List,
        n_sample_attributions: int = 5
    ) -> InterpretabilityReport:
        """
        Run full interpretability analysis.

        Args:
            tasks: List of tasks to analyze
            n_sample_attributions: Number of tasks to do detailed attribution for

        Returns:
            InterpretabilityReport with all analyses
        """
        # Model info
        model_info = {
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'hidden_dim': self.model.hidden_dim,
            'num_primitives': self.model.num_primitives,
            'training_epochs': len(self.model.epoch_history)
        }

        # Embedding analysis
        print("  Running embedding analysis...")
        try:
            embedding_analysis = self.embedding_viz.analyze(tasks, 'pca')
        except Exception as e:
            print(f"  Warning: Embedding analysis failed: {e}")
            embedding_analysis = None

        # Probing
        print("  Running probing classifiers...")
        try:
            probing_results = self.prober.probe_standard_properties(tasks)
        except Exception as e:
            print(f"  Warning: Probing failed: {e}")
            probing_results = {}

        # Primitive correlations
        print("  Analyzing primitive correlations...")
        try:
            primitive_correlations = self.prim_analyzer.compute_correlations(tasks)
        except Exception as e:
            print(f"  Warning: Correlation analysis failed: {e}")
            primitive_correlations = None

        # Sample attributions
        print("  Computing sample attributions...")
        sample_attributions = []
        sample_tasks = tasks[:n_sample_attributions]

        for task in sample_tasks:
            try:
                # Get top predicted primitive
                top_preds = self.model.get_top_predictions(task, n=1)
                if top_preds:
                    prim_name = top_preds[0][0]
                    attr = self.gradient_attr.attribute_primitive(task, prim_name)
                    sample_attributions.append(attr)
            except Exception as e:
                print(f"  Warning: Attribution failed for {task.name}: {e}")

        return InterpretabilityReport(
            model_info=model_info,
            embedding_analysis=embedding_analysis,
            probing_results=probing_results,
            primitive_correlations=primitive_correlations,
            sample_attributions=sample_attributions
        )

    def generate_html_report(
        self,
        report: InterpretabilityReport,
        output_path: str
    ):
        """Generate an HTML report from the analysis."""
        html = []
        html.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Neural Recognition Model - Interpretability Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .metric { display: inline-block; padding: 10px 20px; margin: 5px; background: #ecf0f1; border-radius: 4px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }
        .metric-label { font-size: 12px; color: #7f8c8d; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        .chart-container { height: 400px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Neural Recognition Model - Interpretability Report</h1>
""")

        # Model info
        html.append('<div class="card">')
        html.append('<h2>Model Overview</h2>')
        html.append('<div class="metrics">')
        for key, value in report.model_info.items():
            html.append(f'<div class="metric"><div class="metric-value">{value:,}</div><div class="metric-label">{key.replace("_", " ").title()}</div></div>')
        html.append('</div></div>')

        # Probing results
        if report.probing_results:
            html.append('<div class="card">')
            html.append('<h2>Probing Classifier Results</h2>')
            html.append('<p>Can we decode task properties from embeddings?</p>')
            html.append('<table><tr><th>Property</th><th>Accuracy</th><th>Baseline</th><th>Improvement</th></tr>')
            for prop_name, result in report.probing_results.items():
                improvement_color = 'green' if result.improvement > 0 else 'red'
                html.append(f'<tr><td>{prop_name}</td><td>{result.accuracy:.1%}</td><td>{result.baseline_accuracy:.1%}</td><td style="color:{improvement_color}">{result.improvement:+.1%}</td></tr>')
            html.append('</table></div>')

        # Primitive correlations
        if report.primitive_correlations:
            html.append('<div class="card">')
            html.append('<h2>Primitive Co-activation Patterns</h2>')
            html.append('<h3>Top Correlated Primitive Pairs</h3>')
            html.append('<table><tr><th>Primitive 1</th><th>Primitive 2</th><th>Correlation</th></tr>')
            for p1, p2, corr in report.primitive_correlations.top_correlations[:5]:
                html.append(f'<tr><td>{p1}</td><td>{p2}</td><td>{corr:.3f}</td></tr>')
            html.append('</table>')

            html.append('<h3>Discovered Primitive Clusters</h3>')
            for cluster_id, primitives in report.primitive_correlations.clusters.items():
                html.append(f'<p><strong>Cluster {cluster_id}:</strong> {", ".join(primitives[:10])}{"..." if len(primitives) > 10 else ""}</p>')
            html.append('</div>')

        # Embedding visualization
        if report.embedding_analysis:
            html.append('<div class="card">')
            html.append('<h2>Task Embedding Space</h2>')
            html.append(f'<p>Clustering quality (silhouette score): {report.embedding_analysis.silhouette_score:.3f}</p>')
            html.append('<div class="chart-container"><canvas id="embeddingChart"></canvas></div>')

            # Add embedding data as JavaScript
            points_data = []
            for i, (x, y) in enumerate(report.embedding_analysis.embeddings_2d):
                points_data.append({
                    'x': float(x),
                    'y': float(y),
                    'name': report.embedding_analysis.task_names[i],
                    'family': report.embedding_analysis.task_families[i]
                })

            html.append(f'''
<script>
const embeddingData = {json.dumps(points_data)};
const families = [...new Set(embeddingData.map(p => p.family))];
const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];

new Chart(document.getElementById('embeddingChart'), {{
    type: 'scatter',
    data: {{
        datasets: families.map((family, i) => ({{
            label: family,
            data: embeddingData.filter(p => p.family === family).map(p => ({{x: p.x, y: p.y}})),
            backgroundColor: colors[i % colors.length],
        }}))
    }},
    options: {{
        plugins: {{
            tooltip: {{
                callbacks: {{
                    label: (ctx) => embeddingData[ctx.dataIndex]?.name || ''
                }}
            }}
        }}
    }}
}});
</script>
''')
            html.append('</div>')

        # Sample attributions
        if report.sample_attributions:
            html.append('<div class="card">')
            html.append('<h2>Sample Feature Attributions</h2>')
            for attr in report.sample_attributions:
                html.append(f'<h3>{attr.task_name}</h3>')
                html.append(f'<p>Top primitive: <strong>{attr.primitive_name}</strong> (log-prob: {attr.primitive_log_prob:.3f})</p>')
                html.append(f'<p>Most important example: #{attr.most_important_example_idx + 1}</p>')

                # Top features
                top_features = sorted(attr.feature_scores.items(), key=lambda x: -x[1])[:5]
                html.append('<p>Top features: ')
                html.append(', '.join(f'{k}: {v:.3f}' for k, v in top_features))
                html.append('</p>')
            html.append('</div>')

        html.append('</div></body></html>')

        with open(output_path, 'w') as f:
            f.write('\n'.join(html))


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INTERPRETABILITY TOOLS TEST")
    print("=" * 70)

    # Import dependencies
    from dreamcoder_core.card_primitives import build_card_grammar
    from dreamcoder_core.neural_recognition import NeuralRecognitionModel
    from rules.cards import sample_hand
    from dreamcoder_core.type_system import arrow, HAND, BOOL

    # Build grammar and model
    grammar = build_card_grammar()
    model = NeuralRecognitionModel(grammar=grammar, hidden_dim=64)

    # Create fake tasks
    class FakeTask:
        def __init__(self, name, family):
            self.name = name
            self.family = family
            self.difficulty_level = 1
            self.request_type = arrow(HAND, BOOL)
            self.examples = [(sample_hand(6), i % 2 == 0) for i in range(10)]

    tasks = [
        FakeTask("task_1", "palindrome"),
        FakeTask("task_2", "palindrome"),
        FakeTask("task_3", "counting"),
        FakeTask("task_4", "counting"),
        FakeTask("task_5", "comparison"),
    ]

    # Test embedding visualization
    print("\nTesting embedding visualization...")
    viz = EmbeddingVisualizer(model)
    embeddings, names, families = viz.compute_embeddings(tasks)
    print(f"  Embeddings shape: {embeddings.shape}")

    reduced = viz.reduce_dimensions(embeddings, 'pca')
    print(f"  Reduced shape: {reduced.shape}")

    # Test probing
    print("\nTesting probing classifier...")
    prober = ProbingClassifier(model)
    result = prober.probe(tasks, lambda t: t.family, 'family')
    print(f"  Family probing accuracy: {result.accuracy:.1%}")
    print(f"  Baseline: {result.baseline_accuracy:.1%}")
    print(f"  Improvement: {result.improvement:+.1%}")

    # Test primitive analyzer
    print("\nTesting primitive correlation analysis...")
    analyzer = PrimitiveAnalyzer(model)
    corr = analyzer.compute_correlations(tasks)
    print(f"  Top correlation: {corr.top_correlations[0]}")
    print(f"  Number of clusters: {len(corr.clusters)}")

    # Test gradient attribution
    print("\nTesting gradient attribution...")
    attr = GradientAttribution(model)
    result = attr.attribute_primitive(tasks[0], 'map')
    print(f"  Task: {result.task_name}")
    print(f"  Primitive: {result.primitive_name}")
    print(f"  Most important example: {result.most_important_example_idx}")
    print(f"  Top features: {sorted(result.feature_scores.items(), key=lambda x: -x[1])[:3]}")

    print("\n" + "=" * 70)
    print("INTERPRETABILITY TOOLS TEST COMPLETE")
    print("=" * 70)
