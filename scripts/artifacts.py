"""
Download inference artifacts from HuggingFace Hub at startup if not available locally.
"""

import os
import shutil
from pathlib import Path


DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")

DEFAULT_REPO_ID = os.environ.get(
    "CROSSPAPER_ARTIFACTS_REPO", "zkmine/crosspaper-artifacts"
)

# Local paths that must exist before the app can serve recommendations
REQUIRED_PATHS = [
    DATA_PROCESSED / "base.index",
    DATA_PROCESSED / "finetuned.index",
    DATA_PROCESSED / "base_embeddings.npy",
    DATA_PROCESSED / "finetuned_embeddings.npy",
    DATA_PROCESSED / "paper_metadata.pkl",
    MODELS_DIR / "base" / "model.safetensors",
    MODELS_DIR / "fine_tuned" / "model.safetensors",
]

# Mapping from the layout inside the Hub repo to the local layout
HUB_LAYOUT = [
    ("data/base.index", DATA_PROCESSED / "base.index"),
    ("data/finetuned.index", DATA_PROCESSED / "finetuned.index"),
    ("data/base_embeddings.npy", DATA_PROCESSED / "base_embeddings.npy"),
    ("data/finetuned_embeddings.npy", DATA_PROCESSED / "finetuned_embeddings.npy"),
    ("data/paper_metadata.pkl", DATA_PROCESSED / "paper_metadata.pkl"),
]

HUB_MODEL_DIRS = [
    ("models/base", MODELS_DIR / "base"),
    ("models/fine_tuned", MODELS_DIR / "fine_tuned"),
]


def artifacts_present():
    """Check if all required artifacts exist locally."""
    return all(path.exists() for path in REQUIRED_PATHS)


def ensure_artifacts(repo_id=DEFAULT_REPO_ID):
    """Download artifacts from Hub if not present, otherwise skip."""
    if artifacts_present():
        print("Artifacts found locally, skipping download.")
        return

    # Imported lazily so local runs do not require huggingface_hub
    from huggingface_hub import snapshot_download

    print(f"Artifacts not found locally. Downloading from {repo_id}...")
    print("This runs once per deployment and takes a few minutes.")

    snapshot_path = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
    ))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for hub_relative, local_path in HUB_LAYOUT:
        source = snapshot_path / hub_relative
        if not source.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            shutil.copy2(source, local_path)
            print(f"  {local_path}")

    for hub_relative, local_dir in HUB_MODEL_DIRS:
        source = snapshot_path / hub_relative
        if not source.exists():
            continue
        if not (local_dir / "model.safetensors").exists():
            shutil.copytree(source, local_dir, dirs_exist_ok=True)
            print(f"  {local_dir}/")

    missing = [str(p) for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        raise RuntimeError(
            "Artifact download finished but these paths are still missing:\n  "
            + "\n  ".join(missing)
            + f"\nCheck the layout of {repo_id} on the Hub."
        )

    print("All artifacts ready.")
