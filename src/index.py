# TODO: implement index

import time
from pathlib import Path
from typing import Tuple

import faiss  # type: ignore
import numpy as np


def load_index_data(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load index embeddings and labels from disk.
    """
    emb_path = emb_dir / "index_embeddings.npy"
    labels_path = emb_dir / "index_labels.npy"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    embeddings = np.load(emb_path)
    labels = np.load(labels_path, allow_pickle=True)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

    if len(labels) != embeddings.shape[0]:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match number of embeddings ({embeddings.shape[0]})"
        )

    return embeddings.astype("float32"), labels


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build an exact FAISS index (IndexFlatL2) over the given embeddings.
    """
    n, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    assert index.ntotal == n
    return index


def build_ivf_index(embeddings: np.ndarray, nlist: int | None = None) -> faiss.IndexIVFFlat:
    """
    Build an approximate FAISS index (IndexIVFFlat with L2 metric).

    nlist controls the number of Voronoi cells (clusters). A common heuristic is:
        nlist ~ sqrt(N) or N / 100
    """
    n, d = embeddings.shape

    if nlist is None:
        # simple heuristic: at most 1000, at least 1, roughly N/100
        nlist = max(1, min(1000, n // 100))

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # IVF indices must be trained before adding vectors
    index.train(embeddings)
    index.add(embeddings)
    assert index.ntotal == n

    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Save a FAISS index to disk at the given path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def describe_index(path: Path, build_time_sec: float) -> None:
    """
    Print basic information about an index file on disk.
    """
    size_bytes = path.stat().st_size if path.exists() else 0
    print(
        f"{path.name}: build_time={build_time_sec:.3f}s, size={size_bytes / (1024 ** 2):.2f} MB"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    emb_dir = project_root / "src" / "outputs" / "embeddings"
    index_dir = project_root / "src" / "outputs" / "indexes"

    embeddings, labels = load_index_data(emb_dir)
    n, d = embeddings.shape
    print(f"Loaded embeddings: n={n}, d={d}")

    # 1. Build exact IndexFlatL2
    flat_path = index_dir / "faiss_index_flatl2.index"
    t0 = time.time()
    flat_index = build_flat_index(embeddings)
    t1 = time.time()
    save_index(flat_index, flat_path)
    describe_index(flat_path, t1 - t0)

    # 2. Build approximate IndexIVFFlat
    ivf_path = index_dir / "faiss_index_ivfflat.index"
    t0 = time.time()
    ivf_index = build_ivf_index(embeddings)
    t1 = time.time()
    save_index(ivf_index, ivf_path)
    describe_index(ivf_path, t1 - t0)


if __name__ == "__main__":
    main()
