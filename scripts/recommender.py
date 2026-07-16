"""
Recommendation engine using FAISS retrieval and MMR reranking for diversity.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data/processed")
BASE_MODEL_DIR = Path("models/base")
FINETUNED_MODEL_DIR = Path("models/fine_tuned")

# Field penalty: subtracted from MMR score when paper's field is already selected
# Sized relative to cosine similarity range [-1, 1]; 0.15 is a noticeable but not overwhelming push
FIELD_REPEAT_PENALTY = 0.15


class CrossPaperRecommender:
    """FAISS retrieval with MMR reranking for diverse recommendations."""

    def __init__(self, data_dir=DATA_DIR, model_dir=FINETUNED_MODEL_DIR):
        """Initialize with data and model directories."""
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model = None
        self.index = None
        self.metadata = None
        self.embeddings = None

    def load(self, index_name="finetuned"):
        """Load model, index, and metadata."""
        model_path = (
            BASE_MODEL_DIR if index_name == "base" else FINETUNED_MODEL_DIR
        )
        print(f"Loading {index_name} model from {model_path}...")
        self.model = SentenceTransformer(str(model_path))

        index_path = self.data_dir / f"{index_name}.index"
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))

        embeddings_path = self.data_dir / f"{index_name}_embeddings.npy"
        self.embeddings = np.load(str(embeddings_path))

        metadata_path = self.data_dir / "paper_metadata.pkl"
        self.metadata = pd.read_pickle(str(metadata_path))

        print(f"  Ready: {self.index.ntotal} papers indexed")

    def retrieve(self, query, top_k=50):
        """Retrieve top-k candidates via FAISS cosine similarity search."""
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]

    def mmr_rerank(self, query, candidates_idx, candidates_scores, top_n=10, lambda_param=0.6):
        """Rerank candidates using MMR to balance relevance and diversity."""
        selected = []
        selected_indices = []
        remaining = list(range(len(candidates_idx)))

        for _ in range(min(top_n, len(candidates_idx))):
            best_score = -float("inf")
            best_idx = -1

            for i in remaining:
                paper_idx = candidates_idx[i]
                relevance = candidates_scores[i]

                # Diversity: max similarity to any already-selected paper
                if selected_indices:
                    candidate_emb = self.embeddings[paper_idx].reshape(1, -1)
                    selected_embs = self.embeddings[selected_indices]
                    similarities = np.dot(selected_embs, candidate_emb.T).flatten()
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                # Field diversity penalty
                paper_field = self.metadata.iloc[paper_idx]["field"]
                selected_fields = [
                    self.metadata.iloc[idx]["field"] for idx in selected_indices
                ]
                if paper_field in selected_fields:
                    mmr_score -= FIELD_REPEAT_PENALTY

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx == -1:
                break

            paper_idx = candidates_idx[best_idx]
            selected_indices.append(paper_idx)

            paper_row = self.metadata.iloc[paper_idx]
            selected.append({
                "title": paper_row["title"],
                "abstract": paper_row.get("abstract", "")[:300],
                "field": paper_row["field"],
                "year": int(paper_row.get("year", 0)),
                "cited_by_count": int(paper_row.get("cited_by_count", 0)),
                "relevance_score": float(candidates_scores[best_idx]),
                "mmr_score": float(best_score),
            })

            remaining.remove(best_idx)

        return selected

    def recommend(self, query, top_n=10, lambda_param=0.6):
        """Generate recommendations with MMR reranking and diversity metrics."""
        scores, indices = self.retrieve(query, top_k=top_n * 5)
        recommendations = self.mmr_rerank(
            query, indices, scores, top_n=top_n, lambda_param=lambda_param
        )

        diversity_metrics = self._compute_diversity(recommendations)

        return {
            "recommendations": recommendations,
            "diversity": diversity_metrics,
        }

    def _compute_diversity(self, recommendations):
        """Compute Shannon entropy and field distribution metrics."""
        if not recommendations:
            return {"entropy": 0.0, "distribution": {}, "cross_field_rate": 0.0}

        fields = [r["field"] for r in recommendations]
        unique, counts = np.unique(fields, return_counts=True)
        probs = counts / counts.sum()

        # Shannon entropy (higher = more diverse)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Distribution as percentages
        distribution = {
            disc: float(count / len(fields))
            for disc, count in zip(unique, counts)
        }

        # Cross-field rate (fraction of results NOT from the dominant field)
        dominant_fraction = max(probs)
        cross_rate = 1.0 - dominant_fraction

        return {
            "entropy": float(entropy),
            "distribution": distribution,
            "cross_field_rate": float(cross_rate),
            "num_fields": int(len(unique)),
        }


def main():
    """Quick smoke test for the recommender."""
    recommender = CrossPaperRecommender()
    recommender.load(index_name="finetuned")

    test_queries = [
        "attention mechanism in visual processing",
        "reinforcement learning for decision making",
        "gene expression regulation in neural development",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        result = recommender.recommend(query, top_n=5)

        for i, rec in enumerate(result["recommendations"], 1):
            print(f"  {i}. [{rec['field']}] {rec['title'][:80]}")
            print(f"     relevance={rec['relevance_score']:.3f}  mmr={rec['mmr_score']:.3f}")

        div = result["diversity"]
        print(f"  Diversity: entropy={div['entropy']:.2f}, "
              f"fields={div['num_fields']}, "
              f"cross_rate={div['cross_field_rate']:.0%}")


if __name__ == "__main__":
    main()