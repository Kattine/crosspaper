"""
Compare base and fine-tuned model recommendations. Computes cross-field hit rates,
diversity metrics, and generates before/after plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import CrossPaperRecommender


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")

# Benchmark queries spanning different fields
EVAL_QUERIES = [
    {
        "query": "attention mechanism in visual processing",
        "source_field": "computer_science",
        "expected_cross": ["neuroscience", "psychology"],
    },
    {
        "query": "neural network optimization and gradient descent",
        "source_field": "computer_science",
        "expected_cross": ["physics", "biochemistry_genetics"],
    },
    {
        "query": "memory consolidation during sleep",
        "source_field": "neuroscience",
        "expected_cross": ["psychology", "computer_science"],
    },
    {
        "query": "evolutionary algorithm for complex systems",
        "source_field": "computer_science",
        "expected_cross": ["biochemistry_genetics", "physics"],
    },
    {
        "query": "decision making under uncertainty",
        "source_field": "psychology",
        "expected_cross": ["computer_science", "neuroscience"],
    },
]


class CrossPaperEvaluator:
    """Evaluates recommendation quality across base and fine-tuned models."""

    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
        """Initialize the evaluator.

        Args:
            data_dir: Directory containing indexes and metadata.
            output_dir: Directory to save evaluation results and plots.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, index_name, queries, top_n=10):
        """Run evaluation queries through a single model.

        Args:
            index_name: 'base' or 'finetuned'.
            queries: List of evaluation query dictionaries.
            top_n: Number of recommendations per query.

        Returns:
            List of result dictionaries with metrics per query.
        """
        recommender = CrossPaperRecommender()
        recommender.load(index_name=index_name)

        results = []
        for q in queries:
            output = recommender.recommend(q["query"], top_n=top_n)
            recs = output["recommendations"]
            diversity = output["diversity"]

            rec_fields = [r["field"] for r in recs]
            cross_hits = sum(
                1 for d in rec_fields if d != q["source_field"]
            )
            cross_rate = cross_hits / len(rec_fields) if rec_fields else 0.0

            expected_hits = sum(
                1 for d in rec_fields if d in q["expected_cross"]
            )
            expected_rate = expected_hits / len(rec_fields) if rec_fields else 0.0

            results.append({
                "query": q["query"],
                "source_field": q["source_field"],
                "cross_field_rate": cross_rate,
                "expected_cross_rate": expected_rate,
                "diversity_entropy": diversity["entropy"],
                "num_fields": diversity["num_fields"],
                "distribution": diversity["distribution"],
                "recommendations": recs,
            })

        return results

    def compare(self):
        """Run full before/after comparison and save results.

        Returns:
            Dictionary with base and fine-tuned results plus summary stats.
        """
        print("Evaluating BASE model...")
        base_results = self.evaluate_model("base", EVAL_QUERIES)

        print("\nEvaluating FINE-TUNED model...")
        ft_results = self.evaluate_model("finetuned", EVAL_QUERIES)

        summary = {
            "base": {
                "avg_cross_rate": np.mean([r["cross_field_rate"] for r in base_results]),
                "avg_expected_rate": np.mean([r["expected_cross_rate"] for r in base_results]),
                "avg_entropy": np.mean([r["diversity_entropy"] for r in base_results]),
                "avg_num_fields": np.mean([r["num_fields"] for r in base_results]),
            },
            "finetuned": {
                "avg_cross_rate": np.mean([r["cross_field_rate"] for r in ft_results]),
                "avg_expected_rate": np.mean([r["expected_cross_rate"] for r in ft_results]),
                "avg_entropy": np.mean([r["diversity_entropy"] for r in ft_results]),
                "avg_num_fields": np.mean([r["num_fields"] for r in ft_results]),
            },
        }

        print("\n" + "=" * 60)
        print("  BEFORE/AFTER COMPARISON")
        print("=" * 60)
        for metric in ["avg_cross_rate", "avg_expected_rate", "avg_entropy", "avg_num_fields"]:
            base_val = summary["base"][metric]
            ft_val = summary["finetuned"][metric]
            delta = ft_val - base_val
            arrow = "+" if delta > 0 else ""
            print(f"  {metric:30s}  base={base_val:.3f}  ft={ft_val:.3f}  ({arrow}{delta:.3f})")

        full_results = {
            "summary": summary,
            "base_details": base_results,
            "finetuned_details": ft_results,
        }
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

        self._plot_comparison(base_results, ft_results)

        return full_results

    def _plot_comparison(self, base_results, ft_results):
        """Generate before/after comparison plots for key metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        queries_short = [r["query"][:30] + "..." for r in base_results]
        x = np.arange(len(queries_short))
        width = 0.35

        axes[0].bar(x - width/2, [r["cross_field_rate"] for r in base_results],
                     width, label="Before (base)", color="#94a3b8")
        axes[0].bar(x + width/2, [r["cross_field_rate"] for r in ft_results],
                     width, label="After (fine-tuned)", color="#6366f1")
        axes[0].set_ylabel("Cross-Field Rate")
        axes[0].set_title("Cross-Field Hit Rate")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(queries_short, rotation=45, ha="right", fontsize=7)
        axes[0].legend()

        axes[1].bar(x - width/2, [r["diversity_entropy"] for r in base_results],
                     width, label="Before (base)", color="#94a3b8")
        axes[1].bar(x + width/2, [r["diversity_entropy"] for r in ft_results],
                     width, label="After (fine-tuned)", color="#6366f1")
        axes[1].set_ylabel("Shannon Entropy")
        axes[1].set_title("Field Diversity (Entropy)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(queries_short, rotation=45, ha="right", fontsize=7)
        axes[1].legend()

        axes[2].bar(x - width/2, [r["num_fields"] for r in base_results],
                     width, label="Before (base)", color="#94a3b8")
        axes[2].bar(x + width/2, [r["num_fields"] for r in ft_results],
                     width, label="After (fine-tuned)", color="#6366f1")
        axes[2].set_ylabel("Field Count")
        axes[2].set_title("Number of Fields in Top-10")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(queries_short, rotation=45, ha="right", fontsize=7)
        axes[2].legend()

        plt.tight_layout()
        plot_path = self.output_dir / "before_after_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Comparison plot saved to {plot_path}")


def main():
    """Run the full evaluation pipeline."""
    evaluator = CrossPaperEvaluator()
    evaluator.compare()


if __name__ == "__main__":
    main()
