"""
Upload inference artifacts to HuggingFace Hub. Slims metadata and stages models/indexes before upload.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi


DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
STAGING_DIR = Path("data/outputs/hub_staging")

DEFAULT_REPO_ID = "zkmine/crosspaper-artifacts"

# Abstract characters kept per paper. main.py renders abstract[:300], so this
# leaves headroom without shipping full texts.
ABSTRACT_CHARS = 400

# Files copied verbatim into the staging directory before upload
ARTIFACT_FILES = [
    (DATA_PROCESSED / "base.index", "data/base.index"),
    (DATA_PROCESSED / "finetuned.index", "data/finetuned.index"),
    (DATA_PROCESSED / "base_embeddings.npy", "data/base_embeddings.npy"),
    (DATA_PROCESSED / "finetuned_embeddings.npy", "data/finetuned_embeddings.npy"),
]

ARTIFACT_DIRS = [
    (MODELS_DIR / "base", "models/base"),
    (MODELS_DIR / "fine_tuned", "models/fine_tuned"),
]

# Subdirectories skipped when copying model folders. The Trainer writes
# intermediate checkpoints (model weights plus optimizer state, roughly three
# times the size of the model) next to the final checkpoint. Inference only
# needs the final weights.
MODEL_IGNORE_PATTERNS = ("checkpoint*", "checkpoints", "runs", "eval")


class ArtifactUploader:
    """Stages and uploads artifacts to HuggingFace Hub dataset repo."""

    def __init__(self, repo_id=DEFAULT_REPO_ID, staging_dir=STAGING_DIR):
        """Initialize the uploader with target repo and staging directory."""
        self.repo_id = repo_id
        self.staging_dir = Path(staging_dir)
        self.api = HfApi()

    def slim_metadata(self):
        """Slim metadata by dropping text column and truncating abstracts."""
        source = DATA_PROCESSED / "paper_metadata.pkl"
        print(f"Slimming {source}...")

        metadata = pd.read_pickle(str(source))
        original_cols = list(metadata.columns)

        if "text" in metadata.columns:
            metadata = metadata.drop(columns=["text"])
        if "abstract" in metadata.columns:
            metadata["abstract"] = (
                metadata["abstract"].astype(str).str.slice(0, ABSTRACT_CHARS)
            )

        target_dir = self.staging_dir / "data"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "paper_metadata.pkl"
        metadata.to_pickle(str(target))

        source_mb = source.stat().st_size / 1e6
        target_mb = target.stat().st_size / 1e6
        print(f"  Columns: {original_cols} -> {list(metadata.columns)}")
        print(f"  Size:    {source_mb:.1f} MB -> {target_mb:.1f} MB")
        print(f"  Rows:    {len(metadata)} (order preserved)")

        return target

    def stage(self):
        """Assemble artifacts into staging directory, return total size."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        self.slim_metadata()

        print("\nStaging artifacts...")
        for source, relative_target in ARTIFACT_FILES:
            if not source.exists():
                raise FileNotFoundError(
                    f"{source} not found. Run build_index.py first."
                )
            target = self.staging_dir / relative_target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"  {relative_target}  ({source.stat().st_size / 1e6:.1f} MB)")

        for source, relative_target in ARTIFACT_DIRS:
            if not source.exists():
                raise FileNotFoundError(
                    f"{source} not found. Run fine_tune.py first."
                )
            target = self.staging_dir / relative_target
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*MODEL_IGNORE_PATTERNS),
            )
            size_mb = sum(
                f.stat().st_size for f in target.rglob("*") if f.is_file()
            ) / 1e6
            print(f"  {relative_target}/  ({size_mb:.1f} MB)")

            if not (target / "model.safetensors").exists():
                raise RuntimeError(
                    f"{target}/model.safetensors is missing after staging. "
                    "The final checkpoint may not have been saved correctly."
                )

        total_mb = sum(
            f.stat().st_size for f in self.staging_dir.rglob("*") if f.is_file()
        ) / 1e6
        print(f"\nTotal staged: {total_mb:.1f} MB")
        return total_mb

    def upload(self):
        """Create the Hub repo if needed and upload the staged artifacts."""
        print(f"\nCreating/verifying repo: {self.repo_id}")
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )

        print("Uploading (this may take several minutes)...")
        self.api.upload_folder(
            folder_path=str(self.staging_dir),
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Upload CrossPaper inference artifacts",
        )
        print(f"\nDone: https://huggingface.co/datasets/{self.repo_id}")

    def run(self):
        """Stage and upload all artifacts."""
        self.stage()
        self.upload()


def main():
    """Run the artifact upload pipeline."""
    parser = argparse.ArgumentParser(
        description="Upload CrossPaper inference artifacts to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hub dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Assemble the staging directory without uploading",
    )
    args = parser.parse_args()

    uploader = ArtifactUploader(repo_id=args.repo_id)
    if args.stage_only:
        uploader.stage()
        print("\nStaging complete. Rerun without --stage-only to upload.")
    else:
        uploader.run()


if __name__ == "__main__":
    main()
