"""
Phase 4: Evaluation & Comparison
==================================
Evaluates all trained models on the test set and generates
comparison tables, confusion matrices, and classification reports.

Usage:
    python 04_evaluate.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.trainer_utils import (
    LABEL_NAMES,
    NUM_LABELS,
    get_classification_report,
    get_confusion_matrix,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "outputs/evaluation"
MODELS_DIR = "outputs/models"
PREPROCESSED_DIR = "outputs/preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_KEYS = ["muril", "xlm-roberta", "mbert"]
MODEL_DISPLAY = {"muril": "MuRIL", "xlm-roberta": "XLM-RoBERTa", "mbert": "mBERT"}

# Short label names for confusion matrix readability
SHORT_LABELS = [
    "Not Off.",
    "Off. Untarg.",
    "Off. Indiv.",
    "Off. Group",
    "Off. Other",
    "not-Tamil",
]


def set_plot_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "savefig.facecolor": "#1a1a2e",
        "savefig.dpi": 150,
    })


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict(model, tokenizer, texts, batch_size=32, max_length=128):
    """Run inference on a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits.cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            all_preds.extend(preds.tolist())
            all_logits.extend(logits.tolist())

    return np.array(all_preds), np.array(all_logits)


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = get_confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax1,
        xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
        linewidths=0.5, linecolor="#333",
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"{model_name} — Confusion Matrix (Counts)")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2,
        xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
        linewidths=0.5, linecolor="#333",
        vmin=0, vmax=1,
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"{model_name} — Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(all_metrics, save_path):
    """Create a grouped bar chart comparing models."""
    metric_names = ["Accuracy", "F1 (Weighted)", "F1 (Macro)", "Precision (W)", "Recall (W)"]
    metric_keys = ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#00d2ff", "#f5af19", "#fc466b"]

    for i, (model_key, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=MODEL_DISPLAY.get(model_key, model_key),
                      color=colors[i], edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="white")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Performance")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    set_plot_style()

    print("=" * 60)
    print("  PHASE 4: Evaluation & Comparison")
    print("=" * 60)

    # Load test data
    print("\n[1/4] Loading test data...")
    hf_path = os.path.join(PREPROCESSED_DIR, "hf_dataset")
    if os.path.exists(hf_path):
        dataset = load_from_disk(hf_path)
        test_data = dataset["test"]
    else:
        test_df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "test.csv"))
        test_data = Dataset.from_pandas(test_df)

    test_texts = test_data["text"]
    test_labels = np.array(test_data["label"])
    print(f"  ✓ Test samples: {len(test_texts):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Device: {device}")

    # Evaluate each model
    all_metrics = {}
    all_predictions = {}

    available_models = [k for k in MODEL_KEYS if os.path.exists(os.path.join(MODELS_DIR, k, "best_model"))]
    if not available_models:
        print("\n  ⚠ No trained models found in outputs/models/")
        print("  Run '03_train.py' first to train models.")
        return

    print(f"\n[2/4] Evaluating {len(available_models)} models...")

    for model_key in available_models:
        model_path = os.path.join(MODELS_DIR, model_key, "best_model")
        model_name = MODEL_DISPLAY.get(model_key, model_key)
        print(f"\n  → Evaluating {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

        preds, logits = predict(model, tokenizer, test_texts)
        all_predictions[model_key] = preds

        # Classification report
        report = get_classification_report(test_labels, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_key}_classification_report.csv"))

        # Full text report
        print(f"\n  {model_name} Classification Report:")
        print(get_classification_report(test_labels, preds))

        # Store metrics
        all_metrics[model_key] = {
            "accuracy": report["accuracy"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "f1_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
        }

        # Per-class F1
        for label_id, label_name in enumerate(LABEL_NAMES):
            all_metrics[model_key][f"f1_{label_name}"] = report.get(label_name, {}).get("f1-score", 0)

        # Confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, f"{model_key}_confusion_matrix.png")
        plot_confusion_matrix(test_labels, preds, model_name, cm_path)
        print(f"  ✓ Confusion matrix saved: {cm_path}")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Comparison table
    print(f"\n[3/4] Generating comparison table...")
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df.index.name = "Model"
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_metrics.csv"))

    print("\n  Model Comparison (Test Set):")
    print("  " + "-" * 70)
    print(f"  {'Model':<20s} {'Accuracy':>10s} {'F1-W':>10s} {'F1-M':>10s} {'Prec-W':>10s} {'Rec-W':>10s}")
    print("  " + "-" * 70)
    for model_key, metrics in all_metrics.items():
        print(f"  {MODEL_DISPLAY[model_key]:<20s} "
              f"{metrics['accuracy']:>10.4f} "
              f"{metrics['f1_weighted']:>10.4f} "
              f"{metrics['f1_macro']:>10.4f} "
              f"{metrics['precision_weighted']:>10.4f} "
              f"{metrics['recall_weighted']:>10.4f}")

    # 4. Comparison chart
    print(f"\n[4/4] Generating comparison chart...")
    chart_path = os.path.join(OUTPUT_DIR, "model_comparison_table.png")
    plot_model_comparison(all_metrics, chart_path)
    print(f"  ✓ Saved: {chart_path}")

    # Save predictions for error analysis
    pred_df = pd.DataFrame({
        "text": test_texts,
        "true_label": test_labels,
        "true_label_name": [LABEL_NAMES[l] for l in test_labels],
    })
    for model_key in available_models:
        pred_df[f"pred_{model_key}"] = all_predictions[model_key]
        pred_df[f"pred_{model_key}_name"] = [LABEL_NAMES[p] for p in all_predictions[model_key]]
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "all_predictions.csv"), index=False)
    print(f"  ✓ All predictions saved to: {OUTPUT_DIR}/all_predictions.csv")

    # Save metrics as JSON
    with open(os.path.join(OUTPUT_DIR, "comparison_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Identify best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]["f1_weighted"])
    print(f"\n  🏆 Best model: {MODEL_DISPLAY[best_model[0]]} (F1-Weighted: {best_model[1]['f1_weighted']:.4f})")

    print("\n" + "=" * 60)
    print("  ✅ Phase 4 Complete! Evaluation results saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
