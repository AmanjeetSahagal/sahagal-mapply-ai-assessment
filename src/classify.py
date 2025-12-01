from pathlib import Path
from typing import List
import numpy as np
import faiss  # type: ignore
from collections import Counter

K = 5

def majority_vote(labels: List[str]) -> str:
    """
    Majority vote over the k nearest neighbor labels
    """
    count = Counter(labels)
    max_count = max(count.values())
    candidates = [lab for lab, c in count.items() if c == max_count]
    return sorted(candidates)[0]


def load_test_data(emb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test embeddings and labels.
    """
    emb_path = emb_dir / "test_embeddings.npy"
    labels_path = emb_dir / "test_labels.npy"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing test embeddings file: {emb_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing test labels file: {labels_path}")

    embeddings = np.load(emb_path)
    labels = np.load(labels_path, allow_pickle=True)

    return embeddings.astype("float32"), labels


def load_faiss_index(path: Path) -> faiss.Index:
    """
    Load a FAISS index from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index file: {path}")
    return faiss.read_index(str(path))


def run_knn(index: faiss.Index, test_vectors: np.ndarray, index_labels: np.ndarray) -> np.ndarray:
    """
    Perform k-NN classification on all test vectors using the given FAISS index.
    """
    distances, indices = index.search(test_vectors, K)

    preds = []
    for neigh_idx in indices:
        neighbor_labels = [index_labels[i] for i in neigh_idx]
        pred = majority_vote(neighbor_labels)
        preds.append(pred)

    return np.array(preds, dtype=object)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    emb_dir = project_root / "src" / "outputs" / "embeddings"
    index_dir = project_root / "src" / "outputs" / "indexes"

    results_dir = project_root / "src" / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_embeddings, test_labels = load_test_data(emb_dir)

    index_labels = np.load(emb_dir / "index_labels.npy", allow_pickle=True)

    flat_index_path = index_dir / "faiss_index_flatl2.index"
    ivf_index_path = index_dir / "faiss_index_ivfflat.index"

    flat_index = load_faiss_index(flat_index_path)
    ivf_index = load_faiss_index(ivf_index_path)

    # Run predictions
    y_pred_flat = run_knn(flat_index, test_embeddings, index_labels)
    y_pred_ivf = run_knn(ivf_index, test_embeddings, index_labels)

    # Save raw predictions for later evaluation
    np.save(results_dir / "flat_predictions.npy", y_pred_flat)
    np.save(results_dir / "ivf_predictions.npy", y_pred_ivf)

    print("Saved predictions to:", results_dir)


if __name__ == "__main__":
    main()
