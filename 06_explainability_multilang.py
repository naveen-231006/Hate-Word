"""
06_explainability_multilang.py — LIME Explainability for All Languages
======================================================================
Generates word-importance highlighted figures for paper using LIME.
Selects representative examples from Tamil, Malayalam, and Kannada.

Produces publication-ready figures:
    Input Comment → Prediction → Highlighted Important Words

Usage:
    python 06_explainability_multilang.py --experiment mono_tamil --model mbert
    python 06_explainability_multilang.py --experiment all --model mbert
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
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_config import (
    MODEL_CONFIGS, EXPERIMENTS, UNIFIED_LABELS, NUM_LABELS,
)

try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("⚠ LIME not installed. Install with: pip install lime")


def make_predictor(model, tokenizer, device, max_length=128):
    """Create a prediction function for LIME."""
    def predict_proba(texts):
        all_probs = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch, return_tensors='pt', truncation=True,
                max_length=max_length, padding='max_length',
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)
    return predict_proba


def generate_lime_figure(text, prediction, confidence, word_weights,
                         lang, model_name, output_path,
                         top_n=10):
    """Generate a publication-ready LIME figure.

    Args:
        word_weights: list of (word, weight) tuples from explanation.as_list().
    """
    # Limit to top_n
    word_weights = word_weights[:top_n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={'height_ratios': [1, 2]})

    # Top panel: Input text + prediction
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis('off')

    pred_label = UNIFIED_LABELS[prediction]
    color = '#16a34a' if prediction == 0 else '#dc2626' if prediction >= 2 else '#d97706'

    axes[0].text(0.02, 0.75, f"Input ({lang}):", fontsize=10, fontweight='bold',
                 color='#555', transform=axes[0].transAxes)
    axes[0].text(0.02, 0.45, text[:120] + ("..." if len(text) > 120 else ""),
                 fontsize=11, wrap=True, transform=axes[0].transAxes,
                 fontfamily='serif')
    axes[0].text(0.02, 0.10, f"Prediction: {pred_label}  ({confidence:.1%})",
                 fontsize=11, fontweight='bold', color=color,
                 transform=axes[0].transAxes)
    axes[0].text(0.70, 0.10, f"Model: {model_name}",
                 fontsize=9, color='#888', transform=axes[0].transAxes)

    # Bottom panel: Word importance bars
    words = [w for w, _ in word_weights]
    weights = [w for _, w in word_weights]

    colors = ['#16a34a' if w > 0 else '#dc2626' for w in weights]

    y_pos = range(len(words))
    axes[1].barh(y_pos, weights, color=colors, edgecolor='white', height=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(words, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Feature Importance (LIME)', fontsize=10)
    axes[1].axvline(x=0, color='#ccc', linewidth=0.5)
    axes[1].set_title('Word Importance', fontsize=11, fontweight='bold')

    # Legend
    green_patch = mpatches.Patch(color='#16a34a', label='Supports prediction')
    red_patch = mpatches.Patch(color='#dc2626', label='Opposes prediction')
    axes[1].legend(handles=[green_patch, red_patch], loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def explain_experiment(experiment_name, model_key, n_examples=3,
                       model_base="outputs/models", data_base="outputs/preprocessed",
                       output_base="outputs/analysis/lime_figures"):
    """Generate LIME explanations for representative examples."""
    exp_cfg = EXPERIMENTS[experiment_name]
    model_cfg = MODEL_CONFIGS[model_key]

    model_dir = os.path.join(model_base, experiment_name, model_key, "best_model")
    if not os.path.exists(model_dir):
        print(f"  ⚠ Model not found: {model_dir}")
        return

    out_dir = os.path.join(output_base, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n  Loading {model_cfg['name']} from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    predict_fn = make_predictor(model, tokenizer, device)

    explainer = LimeTextExplainer(class_names=UNIFIED_LABELS, split_expression=r'\s+')

    # Get test data for each test language
    for lang in exp_cfg["test_langs"]:
        test_csv = os.path.join(data_base, lang, "test.csv")
        if not os.path.exists(test_csv):
            continue

        test_df = pd.read_csv(test_csv)
        print(f"\n  Explaining {lang} examples ({model_cfg['name']})...")

        # Select diverse examples: 1 not-offensive, 1-2 offensive
        examples = []

        # Get one correct Not_offensive
        not_off = test_df[test_df["label"] == 0].sample(n=min(1, len(test_df[test_df["label"] == 0])),
                                                         random_state=42)
        examples.append(not_off)

        # Get offensive examples (different sub-types)
        for label_id in [1, 2, 3]:
            subset = test_df[test_df["label"] == label_id]
            if len(subset) > 0:
                examples.append(subset.sample(n=1, random_state=42))
                if len(examples) >= n_examples:
                    break

        if len(examples) < n_examples:
            # Fill with random samples
            remaining = test_df.sample(n=min(n_examples - len(examples), len(test_df)),
                                       random_state=42)
            examples.append(remaining)

        example_df = pd.concat(examples).head(n_examples).reset_index(drop=True)

        for idx, row in example_df.iterrows():
            text = str(row["text"])
            true_label = int(row["label"])

            if not text.strip():
                continue

            print(f"    [{idx+1}/{n_examples}] Explaining: {text[:60]}...")

            # Get prediction
            probs = predict_fn([text])[0]
            pred_label = int(np.argmax(probs))
            confidence = float(probs[pred_label])

            # LIME explanation
            try:
                explanation = explainer.explain_instance(
                    text, predict_fn,
                    num_features=10,
                    num_samples=500,
                    labels=[pred_label],
                )

                output_path = os.path.join(
                    out_dir, f'lime_{lang}_{model_key}_{idx}.png'
                )
                # as_list() returns [(word, weight), ...] sorted by |weight|
                word_weights = explanation.as_list(label=pred_label)
                generate_lime_figure(
                    text, pred_label, confidence,
                    word_weights,
                    lang, model_cfg["name"], output_path,
                )

                # Also save the LIME HTML
                html_path = os.path.join(out_dir, f'lime_{lang}_{model_key}_{idx}.html')
                explanation.save_to_file(html_path)

            except Exception as e:
                print(f"    ⚠ LIME failed: {e}")
                continue

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    if not HAS_LIME:
        print("Please install LIME: pip install lime")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="mono_tamil",
                        choices=list(EXPERIMENTS.keys()) + ["all"])
    parser.add_argument("--model", type=str, default="mbert",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args()

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    print("=" * 60)
    print("  LIME EXPLAINABILITY — Multi-Language")
    print("=" * 60)

    for exp in experiments:
        explain_experiment(exp, args.model, n_examples=args.n_examples)

    print(f"\n  ✅ All LIME explanations complete!")
    print(f"  Figures saved to outputs/analysis/lime_figures/")


if __name__ == "__main__":
    main()
