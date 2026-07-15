"""
Evaluate models using two triplet protocols: adversarial negatives (hard nearest neighbors)
and standard negatives (random same-field papers). Tests if fine-tuning improved accuracy
on hard cases without regressing on standard ones. assisted by Claude (Anthropic, https://claude.ai)
"""

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator


DATA_DIR = Path("data/processed")
BASE_MODEL_DIR = Path("models/base")
FINETUNED_MODEL_DIR = Path("models/fine_tuned")
OUTPUT_DIR = Path("data/outputs")
BATCH_SIZE = 32

ACCURACY_KEY_SUFFIX = "_cosine_accuracy"
CHANCE_LEVEL = 0.5


class TripletProtocolEvaluator:
    """Runs triplet evaluation under two negative-sampling protocols."""

    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
        """Initialize the evaluator.

        Args:
            data_dir: Directory containing validation CSVs and metadata.
            output_dir: Directory to write the results table.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.id_to_title = {}

    def load_titles(self):
        """Build a paper ID to title lookup from the corpus metadata."""
        metadata = pd.read_pickle(str(self.data_dir / "paper_metadata.pkl"))
        self.id_to_title = dict(zip(metadata["id"], metadata["title"]))

    def _title_anchor(self, anchor_id, fallback):
        """Resolve an anchor to its title, matching the fine-tuning format.

        Args:
            anchor_id: OpenAlex ID of the anchor paper.
            fallback: Text to use if no title is available.

        Returns:
            The anchor title, or the fallback text.
        """
        title = self.id_to_title.get(anchor_id)
        if isinstance(title, str) and title.strip():
            return title
        return fallback

    def build_adversarial_evaluator(self):
        """Build evaluator for hard negatives mined from embeddings."""
        path = self.data_dir / "val_triplets.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping adversarial protocol")
            return None

        df = pd.read_csv(path)
        return TripletEvaluator(
            anchors=df["anchor"].tolist(),
            positives=df["positive"].tolist(),
            negatives=df["negative"].tolist(),
            name="adversarial",
            batch_size=BATCH_SIZE,
        )

    def build_standard_evaluator(self):
        """Build evaluator for random negatives (standard protocol)."""
        pos_path = self.data_dir / "val_positives.csv"
        neg_path = self.data_dir / "val_negatives.csv"

        if not pos_path.exists() or not neg_path.exists():
            print("  WARNING: val_positives.csv or val_negatives.csv not found, "
                  "skipping standard protocol")
            return None

        pos_df = pd.read_csv(pos_path)
        neg_df = pd.read_csv(neg_path)

        n = min(len(pos_df), len(neg_df))

        anchors = [
            self._title_anchor(row["anchor_id"], row["anchor"])
            for _, row in pos_df.iloc[:n].iterrows()
        ]

        return TripletEvaluator(
            anchors=anchors,
            positives=pos_df["positive"].iloc[:n].tolist(),
            negatives=neg_df["negative"].iloc[:n].tolist(),
            name="standard",
            batch_size=BATCH_SIZE,
        )

    def score(self, model_path, evaluator, protocol_name):
        """Score a model under a single protocol, return cosine accuracy."""
        model = SentenceTransformer(str(model_path))
        results = evaluator(model)
        key = f"{protocol_name}{ACCURACY_KEY_SUFFIX}"
        return float(results.get(key, results.get(f"eval_{key}", 0.0)))

    def run(self):
        """Run both protocols on both models and print a comparison table.

        Returns:
            DataFrame with one row per protocol and columns for base,
            fine-tuned, and the delta.
        """
        self.load_titles()

        protocols = []
        adversarial = self.build_adversarial_evaluator()
        if adversarial:
            protocols.append(("adversarial", adversarial))
        standard = self.build_standard_evaluator()
        if standard:
            protocols.append(("standard", standard))

        rows = []
        for name, evaluator in protocols:
            print(f"\nEvaluating protocol: {name}")
            base_acc = self.score(BASE_MODEL_DIR, evaluator, name)
            print(f"  base:       {base_acc:.4f}")
            ft_acc = self.score(FINETUNED_MODEL_DIR, evaluator, name)
            print(f"  fine-tuned: {ft_acc:.4f}")
            rows.append({
                "protocol": name,
                "base": base_acc,
                "fine_tuned": ft_acc,
                "delta": ft_acc - base_acc,
            })

        results_df = pd.DataFrame(rows)

        print("\n" + "=" * 64)
        print("  TRIPLET ACCURACY BY NEGATIVE-SAMPLING PROTOCOL")
        print(f"  (chance level = {CHANCE_LEVEL:.0%})")
        print("=" * 64)
        print(f"  {'protocol':<14}{'base':>10}{'fine-tuned':>14}{'delta':>12}")
        print("  " + "-" * 48)
        for _, row in results_df.iterrows():
            print(
                f"  {row['protocol']:<14}"
                f"{row['base']:>10.4f}"
                f"{row['fine_tuned']:>14.4f}"
                f"{row['delta']:>+12.4f}"
            )
        print("=" * 64)

        out_path = self.output_dir / "triplet_protocol_results.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

        return results_df


def main():
    """Run the dual-protocol triplet evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate triplet accuracy under adversarial and standard negatives"
    )
    parser.parse_args()

    evaluator = TripletProtocolEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
