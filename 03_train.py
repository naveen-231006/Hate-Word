"""
Phase 3: Model Fine-Tuning
============================
Fine-tunes MuRIL, XLM-RoBERTa, and mBERT on the Tamil hate speech dataset.
Designed to work on both CPU (slower) and GPU.
For Colab: just run this script after mounting drive and installing requirements.

Usage:
    python 03_train.py --model muril
    python 03_train.py --model xlm-roberta
    python 03_train.py --model mbert
    python 03_train.py --model all
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.trainer_utils import (
    WeightedTrainer,
    compute_metrics,
    compute_class_weights,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
)

# ──────────────────────────────────────────────
# Model configurations
# ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "muril": {
        "name": "MuRIL",
        "hf_id": "google/muril-base-cased",
        "max_length": 128,
    },
    "xlm-roberta": {
        "name": "XLM-RoBERTa",
        "hf_id": "xlm-roberta-base",
        "max_length": 128,
    },
    "mbert": {
        "name": "mBERT",
        "hf_id": "bert-base-multilingual-cased",
        "max_length": 128,
    },
}

# ──────────────────────────────────────────────
# Training configuration (adjusts for CPU vs GPU)
# ──────────────────────────────────────────────

def get_training_args(model_key, output_dir, use_gpu):
    """Get training arguments adapted for available hardware."""

    if use_gpu:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,  # effective batch = 32
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=100,
            report_to="none",
            seed=42,
        )
    else:
        # CPU-friendly: smaller batch, fewer epochs
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,  # effective batch = 32
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=50,
            report_to="none",
            seed=42,
        )


# ──────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize a HuggingFace Dataset."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


# ──────────────────────────────────────────────
# Training pipeline for a single model
# ──────────────────────────────────────────────

def train_model(model_key, preprocessed_dir="outputs/preprocessed", output_base="outputs/models"):
    """Train a single model."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    hf_id = config["hf_id"]
    max_length = config["max_length"]

    output_dir = os.path.join(output_base, model_key)
    os.makedirs(output_dir, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    device_info = "GPU (CUDA)" if use_gpu else "CPU"

    print("\n" + "=" * 60)
    print(f"  Training: {model_name} ({hf_id})")
    print(f"  Device:   {device_info}")
    print("=" * 60)

    # 1. Load preprocessed data
    print(f"\n[1/5] Loading preprocessed data...")
    hf_path = os.path.join(preprocessed_dir, "hf_dataset")

    if os.path.exists(hf_path):
        dataset = load_from_disk(hf_path)
    else:
        # Fallback: load from CSVs
        train_df = pd.read_csv(os.path.join(preprocessed_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(preprocessed_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(preprocessed_dir, "test.csv"))
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })

    print(f"  ✓ Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,} | Test: {len(dataset['test']):,}")

    # 2. Tokenize
    print(f"\n[2/5] Tokenizing with {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    train_tok = tokenize_dataset(dataset["train"], tokenizer, max_length)
    val_tok = tokenize_dataset(dataset["validation"], tokenizer, max_length)
    print(f"  ✓ Tokenized (max_length={max_length})")

    # 3. Compute class weights
    print(f"\n[3/5] Computing class weights...")
    train_labels = dataset["train"]["label"]
    if isinstance(train_labels, list):
        train_labels_np = np.array(train_labels)
    else:
        train_labels_np = np.array(train_labels)
    class_weights = compute_class_weights(train_labels_np)
    print(f"  ✓ Class weights: {[f'{w:.3f}' for w in class_weights]}")

    # 4. Load model
    print(f"\n[4/5] Loading {model_name} model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")

    # 5. Train
    print(f"\n[5/5] Starting training...")
    training_args = get_training_args(model_key, output_dir, use_gpu)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_result = trainer.train()

    # Save best model + tokenizer
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Save training history
    history = {
        "model": model_name,
        "hf_id": hf_id,
        "device": device_info,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "best_metric": trainer.state.best_metric,
    }

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    history["eval_metrics"] = eval_results

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ {model_name} training complete!")
    print(f"  ✓ Best model saved to: {best_model_dir}")
    print(f"  ✓ Best {training_args.metric_for_best_model}: {trainer.state.best_metric:.4f}")
    print(f"  ✓ Training time: {train_result.metrics.get('train_runtime', 0):.0f}s")

    return history


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train hate speech detection models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["muril", "xlm-roberta", "mbert", "all"],
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="outputs/preprocessed",
        help="Path to preprocessed data directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PHASE 3: Model Fine-Tuning")
    print("=" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")

    models_to_train = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    all_histories = {}

    for model_key in models_to_train:
        history = train_model(model_key, preprocessed_dir=args.preprocessed_dir)
        all_histories[model_key] = history

    # Save combined summary
    summary_path = "outputs/models/training_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_histories, f, indent=2)

    print("\n" + "=" * 60)
    print("  ✅ Phase 3 Complete! All models trained.")
    print("=" * 60)

    # Print comparison
    print("\n  Model Comparison (Validation):")
    print("  " + "-" * 60)
    print(f"  {'Model':<20s} {'F1-Weighted':>12s} {'F1-Macro':>12s} {'Accuracy':>12s}")
    print("  " + "-" * 60)
    for key, hist in all_histories.items():
        metrics = hist.get("eval_metrics", {})
        print(f"  {hist['model']:<20s} "
              f"{metrics.get('eval_f1_weighted', 0):>12.4f} "
              f"{metrics.get('eval_f1_macro', 0):>12.4f} "
              f"{metrics.get('eval_accuracy', 0):>12.4f}")


if __name__ == "__main__":
    main()
