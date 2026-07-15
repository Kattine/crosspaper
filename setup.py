"""
Pipeline orchestrator: fetch data, build pairs, mine negatives, fine-tune, build indexes.
"""

import argparse
import subprocess
import sys


PIPELINE_STEPS = [
    ("fetch", "scripts/fetch_openalex.py", "Fetching papers from OpenAlex API"),
    ("pairs", "scripts/build_training_pairs.py", "Constructing cross-field citation pairs"),
    ("index", "scripts/build_index.py", "Building FAISS search index"),
    ("mine", "scripts/mine_hard_negatives.py", "Mining embedding-based hard negatives"),
    ("train", "scripts/fine_tune.py", "Fine-tuning sentence-transformer"),
]

# The index step runs twice: once before mining (the base index is the input to
# hard negative mining) and once after training (to index with the fine-tuned
# model). The full pipeline appends the second pass automatically.
FINAL_INDEX_STEP = (
    "scripts/build_index.py",
    "Rebuilding FAISS index with the fine-tuned model",
)


def run_step(script_path, description):
    """Run a pipeline step as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Running: python {script_path}")
    print(f"{'='*60}\n")

    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: Step failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n  Done: {description}\n")


def main():
    """Run the full pipeline or a specific step."""
    parser = argparse.ArgumentParser(description="CrossPaper setup pipeline")
    parser.add_argument(
        "--step",
        choices=[s[0] for s in PIPELINE_STEPS],
        help="Run a specific step only (default: run all steps in order)",
    )
    args = parser.parse_args()

    if args.step:
        step = next(s for s in PIPELINE_STEPS if s[0] == args.step)
        run_step(step[1], step[2])
    else:
        print("\n  CrossPaper Setup Pipeline")
        print("  Running all steps...\n")
        for _, script_path, description in PIPELINE_STEPS:
            run_step(script_path, description)
        run_step(*FINAL_INDEX_STEP)
        print("\n  All steps completed. Run 'python main.py' to start the app.\n")


if __name__ == "__main__":
    main()
