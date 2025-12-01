from pathlib import Path
from typing import Dict, Any

import json
import numpy as np


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}"
        )

    # Use all labels that appear in either y_true or y_pred
    classes = sorted(set(y_true).union(set(y_pred)))
    idx = {c: i for i, c in enumerate(classes)}

    # Confusion matrix: rows = true labels, cols = predicted labels
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for c in classes:
        i = idx[c]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    accuracy = float((y_true == y_pred).mean())

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def main() -> None:
    """
    Load true labels and predictions, evaluate both index types,
    and write metrics to JSON files.
    """
    project_root = Path(__file__).resolve().parents[1]
    emb_dir = project_root / "src" / "outputs" / "embeddings"
    results_dir = project_root / "src" / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth labels
    y_true_path = emb_dir / "test_labels.npy"
    if not y_true_path.exists():
        raise FileNotFoundError(f"Missing ground-truth labels file: {y_true_path}")
    y_true = np.load(y_true_path, allow_pickle=True)

    # Predictions from classification step
    flat_pred_path = results_dir / "flat_predictions.npy"
    ivf_pred_path = results_dir / "ivf_predictions.npy"
    if not flat_pred_path.exists() or not ivf_pred_path.exists():
        raise FileNotFoundError(
            "Prediction files not found. Run `python -m src.classify` first."
        )

    y_pred_flat = np.load(flat_pred_path, allow_pickle=True)
    y_pred_ivf = np.load(ivf_pred_path, allow_pickle=True)

    # Evaluate both index types
    metrics_flat = evaluate_predictions(y_true, y_pred_flat)
    metrics_ivf = evaluate_predictions(y_true, y_pred_ivf)

    # Save JSON reports
    with open(results_dir / "flat_results.json", "w") as f:
        json.dump(metrics_flat, f, indent=4)

    with open(results_dir / "ivf_results.json", "w") as f:
        json.dump(metrics_ivf, f, indent=4)

    # Print a brief summary
    print("FlatL2 Results:")
    print(f"Accuracy: {metrics_flat['accuracy']:.4f}")
    print()
    print("IVFFlat Results:")
    print(f"Accuracy: {metrics_ivf['accuracy']:.4f}")
    print()
    print("Results saved to:", results_dir)


if __name__ == "__main__":
    main()
