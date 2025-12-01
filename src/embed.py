# TODO: implement embed

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_split_texts_and_labels(
    data_dir: Path, split: str
) -> Tuple[List[str], np.ndarray]:

    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = [
        "product_title",
        "about_product",
        "product_specification",
        "category",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")

    # Fill NaNs with empty strings
    for col in ["product_title", "about_product", "product_specification"]:
        df[col] = df[col].fillna("")

    # Concatenate fields into a single text representation per product
    texts = (
        df["product_title"].astype(str)
        + " [SEP] "
        + df["about_product"].astype(str)
        + " [SEP] "
        + df["product_specification"].astype(str)
    ).tolist()

    labels = df["category"].astype(str).to_numpy()

    return texts, labels


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into a 2D numpy array of embeddings.
    """
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # FAISS works with float32, so we cast here.
    return embeddings.astype("float32")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_dir = project_root / "src" / "outputs" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load texts and labels
    index_texts, index_labels = load_split_texts_and_labels(data_dir, "index")
    test_texts, test_labels = load_split_texts_and_labels(data_dir, "test")

    # Load embedding model and generate embeddings
    model = SentenceTransformer(MODEL_NAME)
    index_embeddings = embed_texts(model, index_texts)
    test_embeddings = embed_texts(model, test_texts)

    # Save embeddings and labels
    np.save(output_dir / "index_embeddings.npy", index_embeddings)
    np.save(output_dir / "test_embeddings.npy", test_embeddings)
    np.save(output_dir / "index_labels.npy", index_labels)
    np.save(output_dir / "test_labels.npy", test_labels)


if __name__ == "__main__":
    main()

