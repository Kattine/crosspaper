"""
Build training pairs from the fetched OpenAlex data.

Positive pairs come from cross-field citation links, while hard negatives are
same-field papers that do not cite each other.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

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


class TrainingPairBuilder:
    """Builds contrastive training pairs from cross-field citations.

    Loads papers from all fields, identifies cross-field citation
    links, and constructs positive/negative pairs for contrastive learning.
    """

    def __init__(self, raw_dir=DATA_RAW, output_dir=DATA_PROCESSED):
        """Initialize the pair builder.

        Args:
            raw_dir: Directory containing raw field JSON files.
            output_dir: Directory to save constructed training pairs.
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.papers = {}          # id -> paper dict
        self.field_map = {}       # id -> source field name

    def load_papers(self):
        """Load all papers from raw JSON files and build lookup indexes."""
        print("Loading papers from all fields...")

        for field_name in FIELD_NAMES:
            filepath = self.raw_dir / f"{field_name}.json"
            if not filepath.exists():
                print(f"  WARNING: {filepath} not found, skipping")
                continue

            with open(filepath, "r") as f:
                papers = json.load(f)

            count = 0
            for paper in papers:
                paper_id = paper["id"]
                if paper_id not in self.papers:
                    self.papers[paper_id] = paper
                    self.field_map[paper_id] = field_name
                    count += 1

            print(f"  {field_name}: {count} new papers loaded")

        print(f"  Total unique papers: {len(self.papers)}")

    def find_cross_field_pairs(self):
        """Find paper pairs linked by citation across different fields.

        Returns:
            List of (anchor_text, positive_text, anchor_field, pos_field)
            tuples where the two papers are from different fields and share
            a citation link.
        """
        print("Finding cross-field citation pairs...")

        positive_pairs = []
        seen_pairs = set()

        for paper_id, paper in tqdm(self.papers.items(), desc="  Scanning citations"):
            source_field = self.field_map.get(paper_id)
            if not source_field:
                continue

            for ref_id in paper.get("referenced_works", []):
                if ref_id not in self.papers:
                    continue

                target_field = self.field_map.get(ref_id)
                if not target_field or target_field == source_field:
                    continue

                pair_key = tuple(sorted([paper_id, ref_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                anchor_text = self._format_text(paper)
                positive_text = self._format_text(self.papers[ref_id])

                if anchor_text and positive_text:
                    positive_pairs.append({
                        "anchor": anchor_text,
                        "positive": positive_text,
                        "anchor_field": source_field,
                        "positive_field": target_field,
                    })

        print(f"  Found {len(positive_pairs)} cross-field positive pairs")

        field_pair_counts = defaultdict(int)
        for pair in positive_pairs:
            key = tuple(sorted([pair["anchor_field"], pair["positive_field"]]))
            field_pair_counts[key] += 1
        print("  Cross-field pair distribution:")
        for (f1, f2), count in sorted(field_pair_counts.items(), key=lambda x: -x[1]):
            print(f"    {f1} <-> {f2}: {count}")

        return positive_pairs

    def mine_hard_negatives(self, num_negatives):
        """Mine hard negative pairs from within the same field.

        Hard negatives are same-field papers with no citation link.
        These teach the model that surface similarity within a field
        does not imply cross-field relevance.

        Args:
            num_negatives: Number of hard negative pairs to generate.

        Returns:
            List of dictionaries with anchor and negative texts.
        """
        print(f"Mining {num_negatives} hard negative pairs...")

        by_field = defaultdict(list)
        for paper_id, field_name in self.field_map.items():
            by_field[field_name].append(paper_id)

        negatives = []
        attempts = 0
        max_attempts = num_negatives * 10

        while len(negatives) < num_negatives and attempts < max_attempts:
            attempts += 1

            field_name = random.choice(list(by_field.keys()))
            paper_ids = by_field[field_name]
            if len(paper_ids) < 2:
                continue

            id_a, id_b = random.sample(paper_ids, 2)
            paper_a = self.papers[id_a]
            paper_b = self.papers[id_b]

            refs_a = set(paper_a.get("referenced_works", []))
            refs_b = set(paper_b.get("referenced_works", []))
            if id_b in refs_a or id_a in refs_b:
                continue

            text_a = self._format_text(paper_a)
            text_b = self._format_text(paper_b)

            if text_a and text_b:
                negatives.append({
                    "anchor": text_a,
                    "negative": text_b,
                    "field": field_name,
                })

        print(f"  Generated {len(negatives)} hard negative pairs")
        return negatives

    def _format_text(self, paper):
        """Format a paper as a single text string for embedding.

        Combines title and abstract with a separator, which is the standard
        input format for sentence-transformer models.

        Args:
            paper: Paper dictionary with 'title' and 'abstract' fields.

        Returns:
            Formatted string, or None if essential fields are missing.
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not title or not abstract:
            return None

        return f"{title} [SEP] {abstract}"

    def build_and_save(self, val_fraction=0.1):
        """Run the full pair construction pipeline and save results.

        Args:
            val_fraction: Fraction of pairs to hold out for validation.
        """
        self.load_papers()

        positives_raw = self.find_cross_field_pairs()
        if not positives_raw:
            print("ERROR: No cross-field pairs found. Check data/raw/ files.")
            return

        negatives_raw = self.mine_hard_negatives(len(positives_raw))

        pos_df = pd.DataFrame(positives_raw)
        neg_df = pd.DataFrame(negatives_raw)

        pos_df = pos_df.sample(frac=1, random_state=42).reset_index(drop=True)
        neg_df = neg_df.sample(frac=1, random_state=42).reset_index(drop=True)

        pos_val_size = max(1, int(len(pos_df) * val_fraction))
        neg_val_size = max(1, int(len(neg_df) * val_fraction))

        pos_train = pos_df.iloc[pos_val_size:]
        pos_val = pos_df.iloc[:pos_val_size]
        neg_train = neg_df.iloc[neg_val_size:]
        neg_val = neg_df.iloc[:neg_val_size]

        pos_train.to_csv(self.output_dir / "train_positives.csv", index=False)
        pos_val.to_csv(self.output_dir / "val_positives.csv", index=False)
        neg_train.to_csv(self.output_dir / "train_negatives.csv", index=False)
        neg_val.to_csv(self.output_dir / "val_negatives.csv", index=False)

        papers_meta = []
        for pid, p in self.papers.items():
            text = self._format_text(p)
            if text is None:
                continue
            papers_meta.append({
                "id": pid,
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "year": p.get("year"),
                "field": self.field_map.get(pid, "unknown"),
                "primary_field_name": p.get("primary_field_name", ""),
                "cited_by_count": p.get("cited_by_count", 0),
                "text": text,
            })

        papers_df = pd.DataFrame(papers_meta)
        papers_df.to_csv(self.output_dir / "papers.csv", index=False)

        print(f"\nSaved to {self.output_dir}/:")
        print(f"  train_positives.csv  ({len(pos_train)} pairs)")
        print(f"  val_positives.csv    ({len(pos_val)} pairs)")
        print(f"  train_negatives.csv  ({len(neg_train)} pairs)")
        print(f"  val_negatives.csv    ({len(neg_val)} pairs)")
        print(f"  papers.csv           ({len(papers_df)} papers)")


def main():
    """Run the training pair construction pipeline."""
    random.seed(42)
    builder = TrainingPairBuilder()
    builder.build_and_save()


if __name__ == "__main__":
    main()
