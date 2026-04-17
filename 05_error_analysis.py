"""
Phase 5: Error Analysis
========================
Analyzes misclassified samples from the best-performing model.
Categorizes error types and identifies patterns.

Usage:
    python 05_error_analysis.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.trainer_utils import LABEL_NAMES

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "outputs/analysis"
EVAL_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PRIORITY = ["muril", "xlm-roberta", "mbert"]  # preference order for "best"


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


def detect_script(text):
    """Classify text script type."""
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', str(text)))
    latin_chars = len(re.findall(r'[a-zA-Z]', str(text)))
    total = tamil_chars + latin_chars
    if total == 0:
        return "Other"
    ratio = tamil_chars / total
    if ratio > 0.7:
        return "Tamil"
    elif ratio < 0.3:
        return "Latin"
    return "Mixed"


def main():
    set_plot_style()

    print("=" * 60)
    print("  PHASE 5: Error Analysis")
    print("=" * 60)

    # 1. Load predictions
    print("\n[1/5] Loading predictions...")
    pred_path = os.path.join(EVAL_DIR, "all_predictions.csv")
    if not os.path.exists(pred_path):
        print("  ⚠ No predictions found. Run 04_evaluate.py first.")
        return

    df = pd.read_csv(pred_path)
    print(f"  ✓ Loaded {len(df):,} predictions")

    # Determine best model
    metrics_path = os.path.join(EVAL_DIR, "comparison_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_model = max(metrics.items(), key=lambda x: x[1]["f1_weighted"])[0]
    else:
        # Fallback: use first available
        pred_cols = [c for c in df.columns if c.startswith("pred_") and not c.endswith("_name")]
        best_model = pred_cols[0].replace("pred_", "") if pred_cols else None

    if best_model is None:
        print("  ⚠ No model predictions found.")
        return

    pred_col = f"pred_{best_model}"
    pred_name_col = f"pred_{best_model}_name"
    print(f"  ✓ Best model: {best_model}")

    # 2. Identify misclassified samples
    print("\n[2/5] Analyzing misclassifications...")
    df["correct"] = df["true_label"] == df[pred_col]
    df["script"] = df["text"].apply(detect_script)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    total = len(df)
    correct = df["correct"].sum()
    incorrect = total - correct
    print(f"  ✓ Correct:   {correct:,} ({correct/total*100:.1f}%)")
    print(f"  ✓ Incorrect: {incorrect:,} ({incorrect/total*100:.1f}%)")

    misclassified = df[~df["correct"]].copy()
    misclassified.to_csv(os.path.join(OUTPUT_DIR, "misclassified_samples.csv"), index=False)
    print(f"  ✓ Saved {len(misclassified)} misclassified samples")

    # 3. Error type categorization
    print("\n[3/5] Categorizing error types...")

    # Dangerous false negatives: Offensive predicted as Not_offensive
    false_neg = misclassified[
        (misclassified["true_label"] != 0) &
        (misclassified["true_label"] != 5) &  # not "not-Tamil"
        (misclassified[pred_col] == 0)
    ]
    print(f"  ⚠ Dangerous false negatives (offensive → not_offensive): {len(false_neg)}")

    # False positives: Not_offensive predicted as Offensive
    false_pos = misclassified[
        (misclassified["true_label"] == 0) &
        (misclassified[pred_col] != 0) &
        (misclassified[pred_col] != 5)
    ]
    print(f"  ⚠ False positives (not_offensive → offensive): {len(false_pos)}")

    # Cross-category offensive confusion
    cross_off = misclassified[
        (misclassified["true_label"].isin([1, 2, 3, 4])) &
        (misclassified[pred_col].isin([1, 2, 3, 4]))
    ]
    print(f"  ⚠ Cross-category offensive confusion: {len(cross_off)}")

    # not-Tamil errors
    not_tamil_errors = misclassified[
        (misclassified["true_label"] == 5) | (misclassified[pred_col] == 5)
    ]
    print(f"  ⚠ not-Tamil related errors: {len(not_tamil_errors)}")

    # 4. Visualizations
    print("\n[4/5] Generating error analysis visualizations...")

    # a) Error distribution by true label
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Error rate per class
    error_rates = []
    for label_id, label_name in enumerate(LABEL_NAMES):
        class_total = (df["true_label"] == label_id).sum()
        class_errors = ((df["true_label"] == label_id) & ~df["correct"]).sum()
        rate = class_errors / class_total * 100 if class_total > 0 else 0
        error_rates.append({"label": label_name, "error_rate": rate, "total": class_total, "errors": class_errors})

    er_df = pd.DataFrame(error_rates)
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#f39c12", "#95a5a6"]
    axes[0, 0].barh(er_df["label"], er_df["error_rate"], color=colors, edgecolor="white", linewidth=0.5)
    axes[0, 0].set_xlabel("Error Rate (%)")
    axes[0, 0].set_title("Error Rate by True Label")
    axes[0, 0].invert_yaxis()
    for i, row in er_df.iterrows():
        axes[0, 0].text(row["error_rate"] + 0.5, i,
                        f"{row['error_rate']:.1f}% ({row['errors']}/{row['total']})",
                        va="center", fontsize=10, color="white")

    # Error by script type
    script_errors = df.groupby("script")["correct"].agg(["count", "sum"])
    script_errors["errors"] = script_errors["count"] - script_errors["sum"]
    script_errors["error_rate"] = script_errors["errors"] / script_errors["count"] * 100
    script_colors = ["#00d2ff", "#f5af19", "#fc466b", "#95a5a6"]
    axes[0, 1].bar(script_errors.index, script_errors["error_rate"],
                   color=script_colors[:len(script_errors)], edgecolor="white", linewidth=0.5)
    axes[0, 1].set_ylabel("Error Rate (%)")
    axes[0, 1].set_title("Error Rate by Script Type")

    # Error by text length
    df["length_bin"] = pd.cut(df["text_length"], bins=[0, 50, 100, 200, 500, float("inf")],
                              labels=["<50", "50-100", "100-200", "200-500", "500+"])
    len_errors = df.groupby("length_bin", observed=True)["correct"].agg(["count", "sum"])
    len_errors["error_rate"] = (len_errors["count"] - len_errors["sum"]) / len_errors["count"] * 100
    axes[1, 0].bar(len_errors.index.astype(str), len_errors["error_rate"],
                   color="#00d2ff", edgecolor="white", linewidth=0.5)
    axes[1, 0].set_ylabel("Error Rate (%)")
    axes[1, 0].set_xlabel("Text Length (chars)")
    axes[1, 0].set_title("Error Rate by Text Length")

    # Confusion flow: most common error pairs
    if len(misclassified) > 0:
        error_pairs = misclassified.groupby(["true_label_name", pred_name_col]).size()
        error_pairs = error_pairs.sort_values(ascending=False).head(10)
        pair_labels = [f"{t} →\n{p}" for t, p in error_pairs.index]
        axes[1, 1].barh(pair_labels, error_pairs.values, color="#fc466b", edgecolor="white", linewidth=0.5)
        axes[1, 1].set_xlabel("Count")
        axes[1, 1].set_title("Top 10 Error Pairs (True → Predicted)")
        axes[1, 1].invert_yaxis()
    else:
        axes[1, 1].text(0.5, 0.5, "No errors!", transform=axes[1, 1].transAxes,
                        ha="center", va="center", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/error_distribution.png")

    # 5. Print sample errors
    print("\n[5/5] Sample misclassified texts:")
    print("=" * 80)

    for error_type, error_df, emoji in [
        ("False Negatives (Offensive → Not_offensive)", false_neg, "🔴"),
        ("False Positives (Not_offensive → Offensive)", false_pos, "🟡"),
        ("Cross-category Confusion", cross_off, "🟠"),
    ]:
        print(f"\n  {emoji} {error_type}:")
        print("  " + "-" * 70)
        for _, row in error_df.head(5).iterrows():
            print(f"  TRUE: {row['true_label_name']}")
            print(f"  PRED: {row[pred_name_col]}")
            print(f"  TEXT: {row['text'][:100]}{'...' if len(str(row['text'])) > 100 else ''}")
            print()

    # Summary statistics
    summary = {
        "total_test_samples": total,
        "correct": int(correct),
        "incorrect": int(incorrect),
        "accuracy": correct / total,
        "false_negatives_dangerous": len(false_neg),
        "false_positives": len(false_pos),
        "cross_category_confusion": len(cross_off),
        "not_tamil_errors": len(not_tamil_errors),
        "best_model": best_model,
        "error_rates_by_class": {r["label"]: r["error_rate"] for _, r in er_df.iterrows()},
    }
    with open(os.path.join(OUTPUT_DIR, "error_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ Error summary saved to: {OUTPUT_DIR}/error_summary.json")

    print("\n" + "=" * 60)
    print("  ✅ Phase 5 Complete! Error analysis saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
