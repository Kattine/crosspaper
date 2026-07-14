"""
Setup script for CrossPaper.
Runs the full pipeline: fetch data, build training pairs, fine-tune model, build FAISS index.

Usage:
    python setup.py              # Run full pipeline
    python setup.py --step fetch # Run a specific step only

AI Attribution: Project structure and pipeline design assisted by Claude (Anthropic).
"""

import argparse
import subprocess
import sys


PIPELINE_STEPS = [
    ("fetch", "scripts/fetch_openalex.py", "Fetching papers from OpenAlex API"),
    ("pairs", "scripts/build_training_pairs.py", "Constructing cross-disciplinary training pairs"),
    ("train", "scripts/fine_tune.py", "Fine-tuning sentence-transformer"),
    ("index", "scripts/build_index.py", "Building FAISS search index"),
]


def run_step(script_path, description):
    """Run a single pipeline step as a subprocess.

    Args:
        script_path: Relative path to the Python script.
        description: Human-readable description for logging.
    """
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
    """Run the full setup pipeline or a specific step."""
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
        print("\n  All steps completed. Run 'python main.py' to start the app.\n")


if __name__ == "__main__":
    main()
