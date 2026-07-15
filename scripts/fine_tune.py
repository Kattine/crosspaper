"""
Fine-tune the sentence-transformer on cross-field citation triplets.

Anchor + positive are papers linked by a cross-field citation; the negative
is an embedding-mined same-field paper that looks similar but has no citation
link to the anchor.

One thing that tripped me up: MPS recompiles its graph every time batch shapes
change, which tanks throughput badly. The fix is uniform triplets, a fixed
max_seq_length, and drop_last=True so every batch is identical.
assisted by Claude (Anthropic, https://claude.ai)
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback


DATA_DIR = Path("data/processed")
BASE_MODEL_DIR = Path("models/base")
FINETUNED_MODEL_DIR = Path("models/fine_tuned")
BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 128
EVAL_STEPS = 100
CACHE_CLEAR_STEPS = 50


class MPSCacheCallback(TrainerCallback):
    """Periodically clears the MPS allocator cache during training.

    MPS does not aggressively free cached blocks, which causes memory
    pressure and swapping on long runs. Clearing every N steps keeps
    step time stable.
    """

    def __init__(self, every_n_steps=CACHE_CLEAR_STEPS):
        """Initialize the callback.

        Args:
            every_n_steps: How often (in optimizer steps) to clear the cache.
        """
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Clear the MPS cache at a fixed step interval.

        Args:
            args: Trainer arguments.
            state: Trainer state (provides global_step).
            control: Trainer control object.
            **kwargs: Additional trainer context (unused).

        Returns:
            The unmodified control object.
        """
        if state.global_step % self.every_n_steps == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        return control


class CrossPaperTrainer:
    """Fine-tunes a sentence-transformer on cross-field citation triplets.

    Manages device selection, data loading, training, evaluation, and
    checkpoint saving for both base and fine-tuned models.
    """

    def __init__(
        self,
        base_model_name=BASE_MODEL_NAME,
        data_dir=DATA_DIR,
        base_dir=BASE_MODEL_DIR,
        output_dir=FINETUNED_MODEL_DIR,
        device=None,
    ):
        """Initialize the trainer.

        Args:
            base_model_name: HuggingFace model identifier for the base model.
            data_dir: Directory containing training pair CSV files.
            base_dir: Directory to save the base model checkpoint.
            output_dir: Directory to save the fine-tuned model checkpoint.
            device: Force a specific device ('mps', 'cuda', 'cpu'). If None,
                the best available device is auto-detected.
        """
        self.base_model_name = base_model_name
        self.data_dir = Path(data_dir)
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.device = device or self._detect_device()

    def _detect_device(self):
        """Detect the best available compute device.

        Prefers Apple Silicon GPU (MPS), then CUDA, then CPU.

        Returns:
            Device string: 'mps', 'cuda', or 'cpu'.
        """
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_datasets(self, max_train=None):
        """Load triplet training and validation data.

        Builds uniform (anchor, positive, negative) triplets. Uniformity is
        required: mixing pair and triplet examples produces variable batch
        shapes, which forces MPS to recompile its graph every step.

        Args:
            max_train: Optional cap on training triplets (used for smoke tests).

        Returns:
            Tuple of (train_dataset, val_evaluator).
        """
        print("Loading training data...")

        train_path = self.data_dir / "train_triplets.csv"
        val_path = self.data_dir / "val_triplets.csv"

        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} not found. Run mine_hard_negatives.py first "
                "to generate embedding-mined hard negative triplets."
            )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        n_train = len(train_df)
        if max_train:
            n_train = min(n_train, max_train)

        train_dataset = Dataset.from_dict({
            "anchor": train_df["anchor"].iloc[:n_train].tolist(),
            "positive": train_df["positive"].iloc[:n_train].tolist(),
            "negative": train_df["negative"].iloc[:n_train].tolist(),
        })

        # Validation: TripletEvaluator measures whether the anchor is closer
        # to the positive than to the negative. Unlike the similarity
        # evaluator, this needs no continuous labels and cannot produce nan.
        val_evaluator = TripletEvaluator(
            anchors=val_df["anchor"].tolist(),
            positives=val_df["positive"].tolist(),
            negatives=val_df["negative"].tolist(),
            name="cross_field_val",
            batch_size=BATCH_SIZE,
        )

        print(f"  Training triplets:   {len(train_dataset)}")
        print(f"  Validation triplets: {len(val_df)}")

        return train_dataset, val_evaluator

    def train(self, smoke_test=False):
        """Run the full fine-tuning pipeline.

        Loads the base model, saves it as the "before" checkpoint, fine-tunes
        on cross-field triplets, and saves the result.

        Args:
            smoke_test: If True, train on 200 triplets for a few steps to
                verify speed and correctness before committing to a full run.
        """
        print(f"\nLoading base model: {self.base_model_name}")
        print(f"  Device: {self.device}")

        model = SentenceTransformer(self.base_model_name, device=self.device)

        model.max_seq_length = MAX_SEQ_LENGTH
        print(f"  Max sequence length: {MAX_SEQ_LENGTH}")

        self.base_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.base_dir))
        print(f"  Base model saved to {self.base_dir}")

        max_train = 200 if smoke_test else None
        train_dataset, val_evaluator = self.load_datasets(max_train=max_train)

        print("\nBaseline evaluation (base model, before training)...")
        baseline = val_evaluator(model)
        baseline_acc = baseline.get("cross_field_val_cosine_accuracy", 0.0)
        print(f"  Base triplet accuracy: {baseline_acc:.4f}")

        epochs = 1 if smoke_test else EPOCHS
        max_steps = 20 if smoke_test else -1

        args = SentenceTransformerTrainingArguments(
            output_dir=str(Path("data/outputs/training_checkpoints")),
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
            eval_strategy="steps" if not smoke_test else "no",
            eval_steps=EVAL_STEPS,
            save_strategy="steps" if not smoke_test else "no",
            save_steps=EVAL_STEPS,
            save_total_limit=1,
            load_best_model_at_end=not smoke_test,
            metric_for_best_model="eval_cross_field_val_cosine_accuracy",
            greater_is_better=True,
            logging_steps=25,
            report_to=[],
            fp16=False,
            bf16=False,
        )

        loss = MultipleNegativesRankingLoss(model)

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=val_evaluator if not smoke_test else None,
            callbacks=[MPSCacheCallback()],
        )

        total_steps = (
            max_steps if max_steps > 0
            else (len(train_dataset) // BATCH_SIZE) * epochs
        )
        print(f"\nTraining configuration:")
        print(f"  Epochs:        {epochs}")
        print(f"  Batch size:    {BATCH_SIZE}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Total steps:   {total_steps}")
        print(f"  Eval every:    {EVAL_STEPS} steps")
        print()

        trainer.train()

        if smoke_test:
            print("\nSmoke test complete. If step time was under ~2 s/it,")
            print("rerun without --smoke for the full training run.")
            return

        print("\nFinal evaluation (fine-tuned model)...")
        final = val_evaluator(model)
        final_acc = final.get("cross_field_val_cosine_accuracy", 0.0)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.output_dir))

        print("\n" + "=" * 55)
        print("  TRAINING COMPLETE")
        print("=" * 55)
        print(f"  Triplet accuracy before: {baseline_acc:.4f}")
        print(f"  Triplet accuracy after:  {final_acc:.4f}")
        print(f"  Improvement:             {final_acc - baseline_acc:+.4f}")
        print(f"\n  Fine-tuned model saved to {self.output_dir}")


def main():
    """Run the fine-tuning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune sentence-transformer on cross-field citations"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 20-step smoke test to verify speed before the full run",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="Force a specific device (default: auto-detect)",
    )
    args = parser.parse_args()

    trainer = CrossPaperTrainer(device=args.device)
    trainer.train(smoke_test=args.smoke)


if __name__ == "__main__":
    main()
