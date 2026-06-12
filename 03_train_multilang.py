"""
03_train_multilang.py — Multi-Language & Cross-Lingual Training
================================================================
Fine-tunes MuRIL, XLM-RoBERTa, and mBERT across all experiment configurations.

Optimized for RTX 3050 (4GB VRAM): batch_size=8, gradient_accumulation=4, fp16.

Usage:
    # Monolingual experiments (Step 3: A, B, C)
    python 03_train_multilang.py --experiment mono_tamil --model all
    python 03_train_multilang.py --experiment mono_malayalam --model mbert
    python 03_train_multilang.py --experiment mono_kannada --model all

    # Multilingual (Step 3: D)
    python 03_train_multilang.py --experiment multi_all --model all

    # Cross-lingual (Step 4)
    python 03_train_multilang.py --experiment cross_ta_ml --model all
    python 03_train_multilang.py --experiment cross_ta_kn --model all
    python 03_train_multilang.py --experiment cross_ml_kn --model all

    # Run ALL experiments sequentially
    python 03_train_multilang.py --experiment all --model all
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
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_config import (
    MODEL_CONFIGS, EXPERIMENTS, UNIFIED_LABELS,
    LABEL2ID, ID2LABEL, NUM_LABELS, GPU_3050_CONFIG,
)
from utils.trainer_utils import (
    FocalLossTrainer,
    compute_class_weights,
)

# ──────────────────────────────────────────────
# Focal Loss hyperparameters
# ──────────────────────────────────────────────

FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
CLASSIFIER_DROPOUT = 0.3


# ──────────────────────────────────────────────
# Metrics (using unified labels)
# ──────────────────────────────────────────────

def compute_metrics_multilang(eval_pred):
    """Compute accuracy + F1 scores using unified labels."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    p_w, r_w, _, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # Per-class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0,
        labels=list(range(NUM_LABELS))
    )

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
    }

    short_names = ["not_off", "untarg", "t_indiv", "t_group", "t_other", "not_lang"]
    for i, short in enumerate(short_names):
        metrics[f"f1_{short}"] = f1_per_class[i]

    return metrics


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
    return tokenized


# ──────────────────────────────────────────────
# Training arguments (optimized for RTX 3050)
# ──────────────────────────────────────────────

def get_training_args(output_dir, use_gpu):
    """Training arguments optimized for RTX 3050 (4GB VRAM)."""
    if use_gpu:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=8,     # RTX 3050: 4GB VRAM
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,     # effective batch = 32
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            fp16=True,
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
    else:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
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
# Train a single model for a single experiment
# ──────────────────────────────────────────────

def train_single(experiment_name, model_key, preprocessed_base="outputs/preprocessed",
                 output_base="outputs/models"):
    """Train one model for one experiment configuration."""
    exp_cfg = EXPERIMENTS[experiment_name]
    model_cfg = MODEL_CONFIGS[model_key]
    model_name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    max_length = model_cfg["max_length"]

    output_dir = os.path.join(output_base, experiment_name, model_key)
    os.makedirs(output_dir, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    device_info = f"GPU ({torch.cuda.get_device_name(0)})" if use_gpu else "CPU"

    print("\n" + "=" * 60)
    print(f"  Experiment: {exp_cfg['description']}")
    print(f"  Model:      {model_name} ({hf_id})")
    print(f"  Device:     {device_info}")
    print("=" * 60)

    # 1. Determine which preprocessed data to load
    train_langs = exp_cfg["train_langs"]

    if len(train_langs) == 1:
        # Monolingual: load from that language's directory
        data_dir = os.path.join(preprocessed_base, train_langs[0])
    elif len(train_langs) == 3:
        # Multi-all: load from combined
        data_dir = os.path.join(preprocessed_base, "combined")
    else:
        # Cross-lingual: load from cross_ directory
        data_dir = os.path.join(preprocessed_base, experiment_name)

    hf_path = os.path.join(data_dir, "hf_dataset")

    print(f"\n[1/5] Loading data from {data_dir}...")
    if os.path.exists(hf_path):
        dataset = load_from_disk(hf_path)
    else:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df[["text", "label"]]),
            "validation": Dataset.from_pandas(val_df[["text", "label"]]),
        })

    print(f"  ✓ Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    # 2. Tokenize
    print(f"\n[2/5] Tokenizing with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    train_tok = tokenize_dataset(dataset["train"], tokenizer, max_length)
    val_tok = tokenize_dataset(dataset["validation"], tokenizer, max_length)
    print(f"  ✓ Tokenized (max_length={max_length})")

    # 3. Class weights
    print(f"\n[3/5] Computing class weights...")
    train_labels = np.array(dataset["train"]["label"])
    class_weights = compute_class_weights(train_labels, num_classes=NUM_LABELS)
    print(f"  ✓ Weights: {[f'{w:.3f}' for w in class_weights]}")

    # 4. Load model
    print(f"\n[4/5] Loading {model_name}...")
    model_config = AutoConfig.from_pretrained(
        hf_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        classifier_dropout=CLASSIFIER_DROPOUT,
    )
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, config=model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {total_params:,}")

    # 5. Train
    print(f"\n[5/5] Training with FocalLoss (gamma={FOCAL_GAMMA})...")
    training_args = get_training_args(output_dir, use_gpu)

    trainer = FocalLossTrainer(
        class_weights=class_weights,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics_multilang,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_result = trainer.train()

    # Save
    best_model_dir = os.path.join(output_dir, "best_model")
    print(f"\n  💾 Saving to {best_model_dir}...")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Eval on validation
    eval_results = trainer.evaluate()

    # Save history
    history = {
        "experiment": experiment_name,
        "description": exp_cfg["description"],
        "model": model_name,
        "hf_id": hf_id,
        "device": device_info,
        "train_langs": exp_cfg["train_langs"],
        "test_langs": exp_cfg["test_langs"],
        "train_samples": len(dataset["train"]),
        "focal_gamma": FOCAL_GAMMA,
        "label_smoothing": LABEL_SMOOTHING,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "best_metric": trainer.state.best_metric,
        "eval_metrics": eval_results,
    }

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    runtime_min = train_result.metrics.get("train_runtime", 0) / 60
    print(f"\n  ✅ {model_name} complete!")
    print(f"  ✓ Best F1-W: {trainer.state.best_metric:.4f}")
    print(f"  ✓ Time: {runtime_min:.1f} min")

    return history


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-language training pipeline")
    parser.add_argument(
        "--experiment",
        type=str,
        default="mono_tamil",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Which model to train",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MULTI-LANGUAGE TRAINING PIPELINE")
    print("=" * 60)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  VRAM:     {vram:.1f} GB")

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]
    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    total_runs = len(experiments) * len(models)
    print(f"\n  Total training runs: {total_runs}")
    print(f"  Experiments: {experiments}")
    print(f"  Models: {[MODEL_CONFIGS[m]['name'] for m in models]}")

    all_histories = {}
    run_idx = 0

    for exp in experiments:
        all_histories[exp] = {}
        for model_key in models:
            run_idx += 1
            print(f"\n{'#'*60}")
            print(f"  RUN {run_idx}/{total_runs}")
            print(f"{'#'*60}")

            history = train_single(exp, model_key)
            all_histories[exp][model_key] = history

    # Save master summary
    summary_path = "outputs/models/training_summary_multilang.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_histories, f, indent=2)

    # Print comparison
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 70)
    print(f"{'Experiment':<25s} {'Model':<12s} {'F1-W':>8s} {'F1-M':>8s} {'Acc':>8s} {'Time':>8s}")
    print("-" * 70)
    for exp, models_dict in all_histories.items():
        for mk, hist in models_dict.items():
            m = hist.get("eval_metrics", {})
            runtime = hist.get("train_runtime", 0) / 60
            print(f"{exp:<25s} {hist['model']:<12s} "
                  f"{m.get('eval_f1_weighted', 0):>8.4f} "
                  f"{m.get('eval_f1_macro', 0):>8.4f} "
                  f"{m.get('eval_accuracy', 0):>8.4f} "
                  f"{runtime:>7.1f}m")

    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
