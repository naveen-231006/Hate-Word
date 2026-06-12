"""
07_ensemble_multilang.py — Ensemble for All Experiments
========================================================
Runs F1-weighted ensemble for each experiment.
For multi-lang and cross-lingual: evaluates on each test language separately.

Usage:
    python 07_ensemble_multilang.py --experiment mono_tamil
    python 07_ensemble_multilang.py --experiment all
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_config import (
    MODEL_CONFIGS, EXPERIMENTS, UNIFIED_LABELS, SHORT_LABELS, NUM_LABELS,
)


def run_ensemble(experiment_name, eval_base="outputs/evaluation"):
    """Run F1-weighted ensemble for one experiment."""
    exp_cfg = EXPERIMENTS[experiment_name]
    eval_dir = os.path.join(eval_base, experiment_name)

    preds_path = os.path.join(eval_dir, "predictions.json")
    metrics_path = os.path.join(eval_dir, "metrics.json")

    if not os.path.exists(preds_path) or not os.path.exists(metrics_path):
        print(f"  ⚠ No evaluation data for {experiment_name}. Run 04_evaluate_multilang.py first.")
        return None

    with open(preds_path) as f:
        all_predictions = json.load(f)
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Ensemble: {exp_cfg['description']}")
    print(f"{'='*60}")

    test_langs = exp_cfg["test_langs"]
    ensemble_results = {}

    for lang in test_langs:
        print(f"\n  --- Test language: {lang} ---")

        # Gather predictions from each model for this test language
        model_keys = []
        model_preds_list = []
        model_probs_list = []

        for mk in MODEL_CONFIGS:
            pred_key = f"{mk}_{lang}"
            if pred_key in all_predictions:
                model_keys.append(mk)
                model_preds_list.append(np.array(all_predictions[pred_key]["pred"]))
                model_probs_list.append(np.array(all_predictions[pred_key]["probs"]))

        if len(model_keys) < 2:
            print(f"    Need ≥2 models for ensemble, found {len(model_keys)}. Skipping.")
            continue

        true_labels = np.array(all_predictions[f"{model_keys[0]}_{lang}"]["true"])
        n_samples = len(true_labels)

        # Build per-class F1 weight matrix
        f1_weights = np.ones((len(model_keys), NUM_LABELS))
        for i, mk in enumerate(model_keys):
            mk_metrics = all_metrics.get(mk, {}).get(lang, {})
            for j, name in enumerate(UNIFIED_LABELS):
                f1_val = mk_metrics.get(f"f1_{name}", 0.0)
                f1_weights[i, j] = max(f1_val, 0.01)

        # Method 1: F1-weighted probability averaging
        weighted_probs = np.zeros((n_samples, NUM_LABELS))
        total_weight = np.zeros(NUM_LABELS)
        for i, mk in enumerate(model_keys):
            probs = model_probs_list[i]
            w = f1_weights[i]
            weighted_probs += probs * w[np.newaxis, :]
            total_weight += w

        avg_probs = weighted_probs / np.maximum(total_weight[np.newaxis, :], 1e-8)
        avg_probs = avg_probs / avg_probs.sum(axis=1, keepdims=True)
        ensemble_preds = np.argmax(avg_probs, axis=1)

        # Metrics
        report = classification_report(
            true_labels, ensemble_preds,
            target_names=UNIFIED_LABELS, output_dict=True, zero_division=0,
        )
        print(classification_report(
            true_labels, ensemble_preds,
            target_names=UNIFIED_LABELS, zero_division=0,
        ))

        ens_metrics = {
            "accuracy": report["accuracy"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "f1_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
        }
        for i, name in enumerate(UNIFIED_LABELS):
            ens_metrics[f"f1_{name}"] = report.get(name, {}).get("f1-score", 0)

        ensemble_results[lang] = ens_metrics

        print(f"    Ensemble Acc={ens_metrics['accuracy']:.4f}, "
              f"F1-W={ens_metrics['f1_weighted']:.4f}, "
              f"F1-M={ens_metrics['f1_macro']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, ensemble_preds, labels=list(range(NUM_LABELS)))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=ax)
        ax.set_title(f'Ensemble — {experiment_name} — {lang}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(eval_dir, f'cm_ensemble_{lang}.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # Comparison with individual models
        print(f"\n    {'Model':<15s} {'Acc':>8s} {'F1-W':>8s} {'F1-M':>8s}")
        print(f"    {'-'*45}")
        for mk in model_keys:
            m = all_metrics.get(mk, {}).get(lang, {})
            print(f"    {MODEL_CONFIGS[mk]['name']:<15s} "
                  f"{m.get('accuracy',0):>8.4f} "
                  f"{m.get('f1_weighted',0):>8.4f} "
                  f"{m.get('f1_macro',0):>8.4f}")
        print(f"    {'ENSEMBLE':<15s} "
              f"{ens_metrics['accuracy']:>8.4f} "
              f"{ens_metrics['f1_weighted']:>8.4f} "
              f"{ens_metrics['f1_macro']:>8.4f}")

        # Improvement
        best_individual = max(
            all_metrics.get(mk, {}).get(lang, {}).get("f1_weighted", 0)
            for mk in model_keys
        )
        improvement = ens_metrics["f1_weighted"] - best_individual
        print(f"\n    Improvement over best single: {improvement:+.4f} F1-W")

    # Save ensemble metrics
    ens_path = os.path.join(eval_dir, "ensemble_metrics.json")
    with open(ens_path, "w") as f:
        json.dump(ensemble_results, f, indent=2)
    print(f"\n  ✓ Ensemble metrics saved to {ens_path}")

    return ensemble_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
    )
    args = parser.parse_args()

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    all_ensemble = {}
    for exp in experiments:
        result = run_ensemble(exp)
        if result:
            all_ensemble[exp] = result

    # Save master ensemble comparison
    master_path = "outputs/evaluation/all_ensemble_comparison.json"
    with open(master_path, "w") as f:
        json.dump(all_ensemble, f, indent=2)

    # Print master table
    print("\n" + "=" * 70)
    print("  ENSEMBLE COMPARISON — ALL EXPERIMENTS")
    print("=" * 70)
    print(f"{'Experiment':<22s} {'TestLang':<10s} {'F1-W':>8s} {'F1-M':>8s} {'Acc':>8s}")
    print("-" * 60)
    for exp, langs in all_ensemble.items():
        for lang, m in langs.items():
            print(f"{exp:<22s} {lang:<10s} "
                  f"{m['f1_weighted']:>8.4f} {m['f1_macro']:>8.4f} {m['accuracy']:>8.4f}")

    print(f"\n  Saved to {master_path}")


if __name__ == "__main__":
    main()
