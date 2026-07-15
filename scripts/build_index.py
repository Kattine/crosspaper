"""
Build FAISS indexes from the paper embeddings.

One index is built from the base model and one from the fine-tuned model so
the app can compare results side by side.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATA_DIR = Path("data/processed")
BASE_MODEL_DIR = Path("models/base")
FINETUNED_MODEL_DIR = Path("models/fine_tuned")
BATCH_SIZE = 256


class IndexBuilder:
    """Builds FAISS indexes for both base and fine-tuned models.

    Encodes all paper texts using each model and stores the resulting
    vectors in FAISS IndexFlatIP indexes for cosine similarity search.
    """

    def __init__(
        self,
        data_dir=DATA_DIR,
        base_model_dir=BASE_MODEL_DIR,
        finetuned_model_dir=FINETUNED_MODEL_DIR,
    ):
        """Initialize the index builder.

        Args:
            data_dir: Directory containing papers.csv and for saving indexes.
            base_model_dir: Path to the base model checkpoint.
            finetuned_model_dir: Path to the fine-tuned model checkpoint.
        """
        self.data_dir = Path(data_dir)
        self.base_model_dir = Path(base_model_dir)
        self.finetuned_model_dir = Path(finetuned_model_dir)

    def load_papers(self):
        """Load paper metadata and texts from CSV.

        Returns:
            Tuple of (texts list, metadata DataFrame).
        """
        print("Loading paper corpus...")
        papers_df = pd.read_csv(self.data_dir / "papers.csv")
        papers_df = papers_df.dropna(subset=["text"])
        texts = papers_df["text"].tolist()
        print(f"  {len(texts)} papers loaded")
        return texts, papers_df

    def encode_and_build(self, model_path, texts, index_name):
        """Encode texts with a model and build a FAISS index.

        Args:
            model_path: Path to the sentence-transformer checkpoint.
            texts: List of text strings to encode.
            index_name: Name prefix for the saved index file.

        Returns:
            Tuple of (FAISS index, numpy array of embeddings).
        """
        print(f"\nEncoding with {index_name} model ({model_path})...")
        model = SentenceTransformer(str(model_path))

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        index_path = self.data_dir / f"{index_name}.index"
        embeddings_path = self.data_dir / f"{index_name}_embeddings.npy"

        faiss.write_index(index, str(index_path))
        np.save(str(embeddings_path), embeddings)

        print(f"  Index saved: {index_path} ({index.ntotal} vectors, {dimension}d)")
        return index, embeddings

    def build_all(self):
        """Build both base and fine-tuned indexes."""
        texts, papers_df = self.load_papers()

        metadata_path = self.data_dir / "paper_metadata.pkl"
        papers_df.to_pickle(str(metadata_path))
        print(f"  Metadata saved: {metadata_path}")

        self.encode_and_build(self.base_model_dir, texts, "base")

        self.encode_and_build(self.finetuned_model_dir, texts, "finetuned")

        print("\nAll indexes built successfully.")


def main():
    """Run the index building pipeline."""
    builder = IndexBuilder()
    builder.build_all()


if __name__ == "__main__":
    main()
