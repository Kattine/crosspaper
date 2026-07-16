"""Generate pitch figures for CrossPaper.

Creates margin distribution, lambda sweep, and field-bridge plots from
pipeline outputs.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recommender import CrossPaperRecommender


DATA_PROCESSED = Path("data/processed")
BASE_MODEL_DIR = Path("models/base")
FINETUNED_MODEL_DIR = Path("models/fine_tuned")
FIGURES_DIR = Path("data/outputs/figures")

# Color palette used in the slides
STOCK = "#E4E7EE"
INK = "#161A2C"
INK_SOFT = "#5B6178"
RULE = "#B9BFCE"
BASE_COLOR = "#9AA2B8"
FT_COLOR = "#4F46E5"

FIELD_COLORS = {
    "computer_science": "#4F46E5",
    "neuroscience": "#9333EA",
    "psychology": "#0D9488",
    "biochemistry_genetics": "#16A34A",
    "physics": "#D97706",
}

FIELD_LABELS = {
    "computer_science": "Computer Science",
    "neuroscience": "Neuroscience",
    "psychology": "Psychology",
    "biochemistry_genetics": "Biochem & Genetics",
    "physics": "Physics",
}

SWEEP_QUERIES = [
    "attention mechanism in visual processing",
    "neural network optimization and gradient descent",
    "memory consolidation during sleep",
    "evolutionary algorithm for complex systems",
    "decision making under uncertainty",
]

LAMBDA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def style_axes(ax):
    """Apply shared axis styling."""
    ax.set_facecolor(STOCK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=1)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


class FigureMaker:
    """Build pitch figures from pipeline artifacts."""

    def __init__(self, output_dir=FIGURES_DIR):
        """Initialize output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _margins(self, model_path, triplets):
        """Compute triplet margins: sim(anchor, positive) - sim(anchor, negative)."""
        model = SentenceTransformer(str(model_path))
        enc = lambda texts: model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )

        anchors = enc(triplets["anchor"].tolist())
        positives = enc(triplets["positive"].tolist())
        negatives = enc(triplets["negative"].tolist())

        sim_pos = np.sum(anchors * positives, axis=1)
        sim_neg = np.sum(anchors * negatives, axis=1)
        return sim_pos - sim_neg

    def margin_distribution(self):
        """Plot margin distributions for base and fine-tuned models."""
        print("Building margin_distribution.png...")
        triplets = pd.read_csv(DATA_PROCESSED / "val_triplets.csv")

        base_m = self._margins(BASE_MODEL_DIR, triplets)
        ft_m = self._margins(FINETUNED_MODEL_DIR, triplets)

        base_acc = float((base_m > 0).mean())
        ft_acc = float((ft_m > 0).mean())
        print(f"  base accuracy from margins:       {base_acc:.4f}")
        print(f"  fine-tuned accuracy from margins: {ft_acc:.4f}")

        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=STOCK)
        style_axes(ax)

        lo = min(base_m.min(), ft_m.min())
        hi = max(base_m.max(), ft_m.max())
        bins = np.linspace(lo, hi, 60)

        ax.hist(base_m, bins=bins, color=BASE_COLOR, alpha=0.85,
                label=f"Base  ·  {base_acc:.1%} correct", edgecolor="none")
        ax.hist(ft_m, bins=bins, color=FT_COLOR, alpha=0.7,
                label=f"Fine-tuned  ·  {ft_acc:.1%} correct", edgecolor="none")

        ax.axvline(0, color=INK, linewidth=1.5, linestyle="--", zorder=5)
        ax.text(0, ax.get_ylim()[1] * 0.97, "  decision boundary",
                color=INK, fontsize=9, va="top", ha="left")

        ax.set_xlabel("cosine margin:  sim(anchor, cross-field citation) − sim(anchor, same-field lookalike)",
                      color=INK_SOFT, fontsize=9.5, labelpad=9)
        ax.set_ylabel("triplets", color=INK_SOFT, fontsize=9.5)
        ax.set_title("Adversarial protocol · margin distribution",
                     color=INK, fontsize=13, loc="left", pad=14, fontweight="medium")

        leg = ax.legend(frameon=False, fontsize=10, loc="upper left",
                        bbox_to_anchor=(0.02, 0.88))
        for text in leg.get_texts():
            text.set_color(INK)

        fig.text(0.99, 0.02,
                 "mass right of the boundary = triplet accuracy",
                 ha="right", color=INK_SOFT, fontsize=8.5, style="italic")

        fig.tight_layout()
        out = self.output_dir / "margin_distribution.png"
        fig.savefig(out, dpi=200, facecolor=STOCK, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    def lambda_sweep(self):
        """Plot diversity metrics across different MMR lambda values."""
        print("Building lambda_sweep.png...")

        results = {}
        for name in ("base", "finetuned"):
            rec = CrossPaperRecommender()
            rec.load(index_name=name)
            entropies, cross_rates = [], []

            for lam in LAMBDA_GRID:
                e_run, c_run = [], []
                for query in SWEEP_QUERIES:
                    out = rec.recommend(query, top_n=10, lambda_param=lam)
                    e_run.append(out["diversity"]["entropy"])
                    c_run.append(out["diversity"]["cross_field_rate"])
                entropies.append(float(np.mean(e_run)))
                cross_rates.append(float(np.mean(c_run)))
                print(f"  {name:10s} lambda={lam:.1f}  "
                      f"entropy={entropies[-1]:.3f}  cross={cross_rates[-1]:.3f}")

            results[name] = {"entropy": entropies, "cross": cross_rates}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=STOCK)

        panels = [
            (axes[0], "entropy", "Field diversity (Shannon entropy)"),
            (axes[1], "cross", "Cross-field rate"),
        ]

        for ax, key, title in panels:
            style_axes(ax)
            ax.plot(LAMBDA_GRID, results["base"][key], "o-", color=BASE_COLOR,
                    linewidth=2, markersize=5, label="Base")
            ax.plot(LAMBDA_GRID, results["finetuned"][key], "o-", color=FT_COLOR,
                    linewidth=2, markersize=5, label="Fine-tuned")
            ax.axvline(0.6, color=RULE, linewidth=1, linestyle=":", zorder=0)
            ax.text(0.6, ax.get_ylim()[0], " deployed", color=INK_SOFT,
                    fontsize=8, va="bottom")
            ax.set_xlabel("lambda   (0 = diversity only,  1 = relevance only)",
                          color=INK_SOFT, fontsize=9)
            ax.set_title(title, color=INK, fontsize=11.5, loc="left",
                         pad=10, fontweight="medium")
            leg = ax.legend(frameon=False, fontsize=9.5)
            for text in leg.get_texts():
                text.set_color(INK)

        fig.suptitle("MMR trade-off · averaged over 5 benchmark queries",
                     color=INK, fontsize=13, x=0.02, ha="left", y=1.0,
                     fontweight="medium")
        fig.tight_layout()
        out = self.output_dir / "lambda_sweep.png"
        fig.savefig(out, dpi=200, facecolor=STOCK, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    def field_bridges(self):
        """Plot counts of cross-field citation pairs by field pair."""
        print("Building field_bridges.png...")
        pairs = pd.read_csv(DATA_PROCESSED / "train_positives.csv")

        counts = Counter()
        for _, row in pairs.iterrows():
            key = tuple(sorted([row["anchor_field"], row["positive_field"]]))
            counts[key] += 1

        ordered = sorted(counts.items(), key=lambda kv: kv[1])
        labels = [
            f"{FIELD_LABELS.get(a, a)}  ·  {FIELD_LABELS.get(b, b)}"
            for (a, b), _ in ordered
        ]
        values = [v for _, v in ordered]
        colors = [FIELD_COLORS.get(a, INK_SOFT) for (a, _), _ in ordered]

        fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=STOCK)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.5)

        y = np.arange(len(values))
        ax.barh(y, values, color=colors, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5, color=INK)

        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.012, i, f"{v:,}", va="center",
                    fontsize=9, color=INK_SOFT)

        ax.set_xlabel("cross-field citation pairs in training data",
                      color=INK_SOFT, fontsize=9.5, labelpad=8)
        ax.set_title("Which bridges the model could learn from",
                     color=INK, fontsize=13, loc="left", pad=14,
                     fontweight="medium")
        ax.set_xlim(0, max(values) * 1.1)

        fig.tight_layout()
        out = self.output_dir / "field_bridges.png"
        fig.savefig(out, dpi=200, facecolor=STOCK, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    def run_all(self):
        """Generate all figures."""
        self.margin_distribution()
        self.lambda_sweep()
        self.field_bridges()
        print(f"\nAll figures written to {self.output_dir}/")


def main():
    """Generate pitch deck figures."""
    parser = argparse.ArgumentParser(description="Generate CrossPaper pitch figures")
    parser.add_argument(
        "--only",
        choices=["margin", "sweep", "bridges"],
        help="Generate a single figure instead of all three",
    )
    args = parser.parse_args()

    maker = FigureMaker()
    if args.only == "margin":
        maker.margin_distribution()
    elif args.only == "sweep":
        maker.lambda_sweep()
    elif args.only == "bridges":
        maker.field_bridges()
    else:
        maker.run_all()


if __name__ == "__main__":
    main()
