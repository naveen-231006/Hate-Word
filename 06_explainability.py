"""
Phase 5b: Explainability with LIME
====================================
Generates LIME explanations for the best model's predictions
on selected misclassified and correctly classified samples.

Usage:
    python 06_explainability.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.trainer_utils import LABEL_NAMES

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "outputs/analysis/lime_explanations"
EVAL_DIR = "outputs/evaluation"
MODELS_DIR = "outputs/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 10  # number of samples to explain
NUM_FEATURES = 15  # top features in LIME explanation


# ──────────────────────────────────────────────
# Model predictor wrapper for LIME
# ──────────────────────────────────────────────

class ModelPredictor:
    """Wraps a HuggingFace model for LIME's predict_proba interface."""

    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device

    def predict_proba(self, texts):
        """Return probability predictions for a list of texts."""
        self.model.eval()
        all_probs = []

        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            encodings = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE 5b: Explainability (LIME)")
    print("=" * 60)

    # 1. Determine best model
    print("\n[1/4] Loading best model...")
    metrics_path = os.path.join(EVAL_DIR, "comparison_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_model_key = max(metrics.items(), key=lambda x: x[1]["f1_weighted"])[0]
    else:
        # Fallback
        for key in ["muril", "xlm-roberta", "mbert"]:
            if os.path.exists(os.path.join(MODELS_DIR, key, "best_model")):
                best_model_key = key
                break
        else:
            print("  ⚠ No trained models found.")
            return

    model_path = os.path.join(MODELS_DIR, best_model_key, "best_model")
    print(f"  ✓ Using model: {best_model_key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    predictor = ModelPredictor(model, tokenizer)

    # 2. Load samples
    print("\n[2/4] Selecting samples for explanation...")
    pred_path = os.path.join(EVAL_DIR, "all_predictions.csv")
    if not os.path.exists(pred_path):
        print("  ⚠ No predictions found. Run 04_evaluate.py first.")
        return

    df = pd.read_csv(pred_path)
    pred_col = f"pred_{best_model_key}"
    pred_name_col = f"pred_{best_model_key}_name"
    df["correct"] = df["true_label"] == df[pred_col]

    # Select samples: mix of correct and incorrect, diverse labels
    selected = []

    # Misclassified samples (more interesting for analysis)
    misclassified = df[~df["correct"]].sample(n=min(NUM_SAMPLES // 2, len(df[~df["correct"]])),
                                                random_state=42)
    selected.append(misclassified)

    # Correctly classified offensive samples
    correct_offensive = df[(df["correct"]) & (df["true_label"].isin([1, 2, 3, 4]))].sample(
        n=min(NUM_SAMPLES // 4, len(df[(df["correct"]) & (df["true_label"].isin([1, 2, 3, 4]))])),
        random_state=42
    )
    selected.append(correct_offensive)

    # Correctly classified non-offensive
    correct_safe = df[(df["correct"]) & (df["true_label"] == 0)].sample(
        n=min(NUM_SAMPLES // 4, len(df[(df["correct"]) & (df["true_label"] == 0)])),
        random_state=42
    )
    selected.append(correct_safe)

    samples = pd.concat(selected).head(NUM_SAMPLES).reset_index(drop=True)
    print(f"  ✓ Selected {len(samples)} samples ({(~samples['correct']).sum()} misclassified)")

    # 3. Generate LIME explanations
    print("\n[3/4] Generating LIME explanations...")
    explainer = LimeTextExplainer(class_names=LABEL_NAMES, verbose=False)

    explanation_summaries = []

    for idx, row in samples.iterrows():
        text = str(row["text"])
        true_label = int(row["true_label"])
        pred_label = int(row[pred_col])
        is_correct = row["correct"]

        print(f"\n  [{idx+1}/{len(samples)}] {'✓' if is_correct else '✗'} "
              f"TRUE: {LABEL_NAMES[true_label]} | PRED: {LABEL_NAMES[pred_label]}")
        print(f"  TEXT: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            explanation = explainer.explain_instance(
                text,
                predictor.predict_proba,
                num_features=NUM_FEATURES,
                num_samples=500,  # reduced for CPU speed
                labels=list(range(len(LABEL_NAMES))),
            )

            # Save HTML
            html_path = os.path.join(OUTPUT_DIR, f"sample_{idx:03d}.html")
            explanation.save_to_file(html_path)

            # Get top features for predicted class
            top_features = explanation.as_list(label=pred_label)
            print(f"  Top features: {top_features[:5]}")

            explanation_summaries.append({
                "sample_idx": idx,
                "text": text[:200],
                "true_label": LABEL_NAMES[true_label],
                "predicted_label": LABEL_NAMES[pred_label],
                "correct": is_correct,
                "top_features": str(top_features[:10]),
                "html_file": f"sample_{idx:03d}.html",
            })

        except Exception as e:
            print(f"  ⚠ Error explaining sample {idx}: {e}")
            continue

    # 4. Save summary
    print(f"\n[4/4] Saving explanation summary...")
    summary_df = pd.DataFrame(explanation_summaries)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "explanation_summary.csv"), index=False)
    print(f"  ✓ {len(explanation_summaries)} explanations saved to {OUTPUT_DIR}/")

    print("\n" + "=" * 60)
    print("  ✅ Phase 5b Complete! LIME explanations saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
