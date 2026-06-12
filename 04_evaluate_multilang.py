"""
04_evaluate_multilang.py — Multi-Experiment Evaluation
=======================================================
Evaluates all trained models across all experiments.
Generates per-class metrics, confusion matrices, and comparison tables.

Usage:
    python 04_evaluate_multilang.py --experiment mono_tamil
    python 04_evaluate_multilang.py --experiment all
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_config import (
    MODEL_CONFIGS, EXPERIMENTS, UNIFIED_LABELS, SHORT_LABELS,
    LABEL2ID, ID2LABEL, NUM_LABELS,
)


def predict_batch(model, tokenizer, texts, device, batch_size=32, max_length=128):
    """Run inference on a list of texts, return predicted label IDs and probabilities."""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors='pt', truncation=True,
            max_length=max_length, padding='max_length',
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_probs)


def evaluate_experiment(experiment_name, model_base="outputs/models",
                        data_base="outputs/preprocessed", out_base="outputs/evaluation"):
    """Evaluate all trained models for one experiment."""
    exp_cfg = EXPERIMENTS[experiment_name]
    test_langs = exp_cfg["test_langs"]

    out_dir = os.path.join(out_base, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Evaluating: {exp_cfg['description']}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load test data for each test language
    test_sets = {}
    for lang in test_langs:
        test_csv = os.path.join(data_base, lang, "test.csv")
        if os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            test_sets[lang] = test_df
            print(f"  Test set ({lang}): {len(test_df):,} samples")
        else:
            print(f"  ⚠ No test set found for {lang} at {test_csv}")

    if not test_sets:
        print("  ✗ No test data available. Skipping.")
        return {}

    # Load and evaluate each model
    all_metrics = {}
    all_predictions = {}

    for model_key, model_cfg in MODEL_CONFIGS.items():
        model_dir = os.path.join(model_base, experiment_name, model_key, "best_model")
        if not os.path.exists(model_dir):
            print(f"\n  ⚠ Model not found: {model_dir} — skipping {model_cfg['name']}")
            continue

        print(f"\n  Loading {model_cfg['name']} from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

        all_metrics[model_key] = {}

        for lang, test_df in test_sets.items():
            texts = test_df["text"].fillna("").tolist()
            true_labels = test_df["label"].values

            preds, probs = predict_batch(model, tokenizer, texts, device)

            # Metrics
            report = classification_report(
                true_labels, preds,
                target_names=UNIFIED_LABELS, output_dict=True, zero_division=0,
            )

            metrics = {
                "accuracy": report["accuracy"],
                "f1_weighted": report["weighted avg"]["f1-score"],
                "f1_macro": report["macro avg"]["f1-score"],
                "precision_weighted": report["weighted avg"]["precision"],
                "recall_weighted": report["weighted avg"]["recall"],
            }
            for i, name in enumerate(UNIFIED_LABELS):
                metrics[f"f1_{name}"] = report.get(name, {}).get("f1-score", 0)

            all_metrics[model_key][lang] = metrics

            # Save predictions
            pred_key = f"{model_key}_{lang}"
            all_predictions[pred_key] = {
                "true": true_labels.tolist(),
                "pred": preds.tolist(),
                "probs": probs.tolist(),
            }

            print(f"    {model_cfg['name']} on {lang}: "
                  f"Acc={metrics['accuracy']:.4f}, F1-W={metrics['f1_weighted']:.4f}, "
                  f"F1-M={metrics['f1_macro']:.4f}")

            # Confusion matrix
            cm = confusion_matrix(true_labels, preds, labels=list(range(NUM_LABELS)))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                        xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=ax)
            ax.set_title(f'{model_cfg["name"]} — {lang} (Test Set)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.tight_layout()
            cm_path = os.path.join(out_dir, f'cm_{model_key}_{lang}.png')
            plt.savefig(cm_path, dpi=150)
            plt.close()

        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved to {metrics_path}")

    # Save predictions
    preds_path = os.path.join(out_dir, "predictions.json")
    with open(preds_path, "w") as f:
        json.dump(all_predictions, f, indent=2)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
    )
    args = parser.parse_args()

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    master_results = {}
    for exp in experiments:
        metrics = evaluate_experiment(exp)
        if metrics:
            master_results[exp] = metrics

    # Save master comparison
    master_path = "outputs/evaluation/all_experiments_comparison.json"
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    with open(master_path, "w") as f:
        json.dump(master_results, f, indent=2)

    # Print master table
    print("\n" + "=" * 80)
    print("  MASTER COMPARISON")
    print("=" * 80)
    print(f"{'Experiment':<22s} {'Model':<12s} {'TestLang':<10s} "
          f"{'F1-W':>8s} {'F1-M':>8s} {'Acc':>8s}")
    print("-" * 80)
    for exp, models in master_results.items():
        for mk, langs in models.items():
            for lang, m in langs.items():
                print(f"{exp:<22s} {MODEL_CONFIGS[mk]['name']:<12s} {lang:<10s} "
                      f"{m['f1_weighted']:>8.4f} {m['f1_macro']:>8.4f} "
                      f"{m['accuracy']:>8.4f}")

    print(f"\n  Saved to {master_path}")


if __name__ == "__main__":
    main()
