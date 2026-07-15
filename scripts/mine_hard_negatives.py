"""
Mine hard negatives from the base model's embedding space.

Random same-field negatives turned out to be too easy — the base model already
got 95%+ triplet accuracy on them and barely changed during fine-tuning. The
fix is to use embedding-nearest-neighbors as negatives: same-field papers that
the model already finds similar to the anchor but that share no citation link.

Needs the base embeddings from build_index.py to run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

FIELD_NAMES = [
    "computer_science",
    "neuroscience",
    "psychology",
    "biochemistry_genetics",
    "physics",
]

NEIGHBOR_POOL = 60
SKIP_TOP = 3


class HardNegativeMiner:
    """Mines topically-confusable same-field negatives via embedding search.

    Uses the base model's embedding space to find papers that look similar
    to each anchor but are not connected by citation, producing negatives
    that are genuinely hard to distinguish from true cross-field matches.
    """

    def __init__(self, raw_dir=DATA_RAW, processed_dir=DATA_PROCESSED, anchor_mode="title"):
        """Initialize the miner.

        Args:
            raw_dir: Directory containing raw field JSON files.
            processed_dir: Directory with base embeddings and pair CSVs.
            anchor_mode: How to format the anchor text.
                'title': use the title only (~10-15 words). This matches the
                    short free-text queries the deployed app receives, so the
                    model trains on the same short-to-long distribution it is
                    asked to handle at inference time.
                'full': use title + abstract (~250 words). This mismatches
                    inference and was found to degrade query-based retrieval.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.anchor_mode = anchor_mode
        self.embeddings = None
        self.metadata = None
        self.id_to_row = {}
        self.refs_by_id = {}

    def _anchor_text(self, anchor_id, fallback_text):
        """Build the anchor text according to the configured anchor mode.

        Args:
            anchor_id: OpenAlex ID of the anchor paper.
            fallback_text: Full text to use if the title is unavailable.

        Returns:
            Anchor string to use in the training triplet.
        """
        if self.anchor_mode == "full":
            return fallback_text

        row = self.id_to_row.get(anchor_id)
        if row is None:
            return fallback_text

        title = self.metadata.iloc[row]["title"]
        if not isinstance(title, str) or not title.strip():
            return fallback_text
        return title

    def load_resources(self):
        """Load base embeddings, paper metadata, and citation references."""
        print("Loading base embeddings and metadata...")

        emb_path = self.processed_dir / "base_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"{emb_path} not found. Run build_index.py first."
            )
        self.embeddings = np.load(str(emb_path))

        self.metadata = pd.read_pickle(str(self.processed_dir / "paper_metadata.pkl"))
        self.metadata = self.metadata.reset_index(drop=True)

        self.id_to_row = {
            pid: row for row, pid in enumerate(self.metadata["id"].tolist())
        }

        print(f"  Embeddings: {self.embeddings.shape}")
        print(f"  Metadata:   {len(self.metadata)} papers")

        # Load citation references from raw data
        print("Loading citation references...")
        for field_name in FIELD_NAMES:
            filepath = self.raw_dir / f"{field_name}.json"
            if not filepath.exists():
                continue
            with open(filepath, "r") as f:
                papers = json.load(f)
            for paper in papers:
                pid = paper["id"]
                if pid not in self.refs_by_id:
                    self.refs_by_id[pid] = set(paper.get("referenced_works", []))

        print(f"  Citation lists: {len(self.refs_by_id)} papers")

    def _find_hard_negative(self, anchor_id, anchor_field):
        """Find a same-field near neighbor of the anchor with no citation link.

        Args:
            anchor_id: OpenAlex ID of the anchor paper.
            anchor_field: Field name of the anchor paper.

        Returns:
            Row index of the chosen hard negative, or None if none qualifies.
        """
        anchor_row = self.id_to_row.get(anchor_id)
        if anchor_row is None:
            return None

        anchor_emb = self.embeddings[anchor_row]

        # Cosine similarity against the whole corpus (embeddings are normalized)
        similarities = self.embeddings @ anchor_emb

        # Take the most similar candidates
        candidate_rows = np.argpartition(-similarities, NEIGHBOR_POOL)[:NEIGHBOR_POOL]
        candidate_rows = candidate_rows[np.argsort(-similarities[candidate_rows])]

        anchor_refs = self.refs_by_id.get(anchor_id, set())

        for rank, row in enumerate(candidate_rows):
            if rank < SKIP_TOP:
                continue
            if row == anchor_row:
                continue

            cand = self.metadata.iloc[row]
            cand_id = cand["id"]

            if cand["field"] != anchor_field:
                continue

            if cand_id in anchor_refs:
                continue
            if anchor_id in self.refs_by_id.get(cand_id, set()):
                continue

            return row

        return None

    def mine(self, pairs_df, split_name):
        """Mine one hard negative per positive pair.

        Args:
            pairs_df: DataFrame with anchor_id, anchor, positive, anchor_field.
            split_name: Name for logging ('train' or 'val').

        Returns:
            DataFrame with anchor, positive, negative columns (triplets).
        """
        print(f"\nMining hard negatives for {split_name} split...")

        triplets = []
        misses = 0

        for _, row in tqdm(
            pairs_df.iterrows(), total=len(pairs_df), desc=f"  {split_name}"
        ):
            neg_row = self._find_hard_negative(
                row["anchor_id"], row["anchor_field"]
            )
            if neg_row is None:
                misses += 1
                continue

            triplets.append({
                "anchor": self._anchor_text(row["anchor_id"], row["anchor"]),
                "positive": row["positive"],
                "negative": self.metadata.iloc[neg_row]["text"],
                "anchor_field": row["anchor_field"],
                "positive_field": row["positive_field"],
                "negative_field": self.metadata.iloc[neg_row]["field"],
            })

        print(f"  Mined {len(triplets)} triplets ({misses} anchors had no valid negative)")
        return pd.DataFrame(triplets)

    def run(self):
        """Run the full hard negative mining pipeline and save triplets."""
        self.load_resources()

        print(f"\nAnchor mode: {self.anchor_mode}")
        if self.anchor_mode == "title":
            print("  Anchors are titles only, matching short queries at inference.")
        else:
            print("  Anchors are title + abstract (mismatches short queries).")

        pos_train = pd.read_csv(self.processed_dir / "train_positives.csv")
        pos_val = pd.read_csv(self.processed_dir / "val_positives.csv")

        if "anchor_id" not in pos_train.columns:
            raise ValueError(
                "train_positives.csv has no anchor_id column. "
                "Re-run build_training_pairs.py with the updated script."
            )

        train_triplets = self.mine(pos_train, "train")
        val_triplets = self.mine(pos_val, "val")

        train_triplets.to_csv(
            self.processed_dir / "train_triplets.csv", index=False
        )
        val_triplets.to_csv(
            self.processed_dir / "val_triplets.csv", index=False
        )

        print(f"\nSaved to {self.processed_dir}/:")
        print(f"  train_triplets.csv  ({len(train_triplets)} triplets)")
        print(f"  val_triplets.csv    ({len(val_triplets)} triplets)")
        if len(train_triplets):
            sample = train_triplets.iloc[0]
            print(f"\n  Sample anchor:   {str(sample['anchor'])[:90]}")
            print(f"  Sample positive: {str(sample['positive'])[:90]}")
            print(f"  Sample negative: {str(sample['negative'])[:90]}")


def main():
    """Run the hard negative mining pipeline."""
    parser = argparse.ArgumentParser(
        description="Mine embedding-based hard negatives for cross-field training"
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["title", "full"],
        default="title",
        help="Anchor text format: 'title' matches short queries (default), "
             "'full' uses title + abstract",
    )
    args = parser.parse_args()

    miner = HardNegativeMiner(anchor_mode=args.anchor_mode)
    miner.run()


if __name__ == "__main__":
    main()
