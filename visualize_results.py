"""
visualize_results.py — Publication-Ready Figures for the Paper
===============================================================
Generates all comparison charts, heatmaps, and transfer matrices.

Usage:
    python visualize_results.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_config import MODEL_CONFIGS, EXPERIMENTS, UNIFIED_LABELS, SHORT_LABELS

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Colors
MODEL_COLORS = {
    'muril': '#00d2ff',
    'xlm-roberta': '#f5af19',
    'mbert': '#fc466b',
    'ensemble': '#7c3aed',
}
LANG_COLORS = {
    'tamil': '#2563eb',
    'malayalam': '#16a34a',
    'kannada': '#d97706',
}


def load_results():
    """Load all evaluation results."""
    results = {}

    # Individual model metrics
    master_path = "outputs/evaluation/all_experiments_comparison.json"
    if os.path.exists(master_path):
        with open(master_path) as f:
            results["individual"] = json.load(f)

    # Ensemble metrics
    ens_path = "outputs/evaluation/all_ensemble_comparison.json"
    if os.path.exists(ens_path):
        with open(ens_path) as f:
            results["ensemble"] = json.load(f)

    return results


def figure1_monolingual_comparison(results):
    """
    Figure 1: Monolingual results — bar chart comparing 3 models + ensemble
    across Tamil, Malayalam, Kannada.
    """
    if "individual" not in results:
        print("  ⚠ No individual results. Skipping Figure 1.")
        return

    individual = results["individual"]
    ensemble = results.get("ensemble", {})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    mono_exps = {
        "tamil": "mono_tamil",
        "malayalam": "mono_malayalam",
        "kannada": "mono_kannada",
    }

    for idx, (lang, exp_name) in enumerate(mono_exps.items()):
        ax = axes[idx]
        exp_data = individual.get(exp_name, {})
        ens_data = ensemble.get(exp_name, {}).get(lang, {})

        models_list = list(MODEL_CONFIGS.keys()) + ["ensemble"]
        model_names = [MODEL_CONFIGS.get(m, {}).get("name", "Ensemble") for m in models_list]

        metrics = ["accuracy", "f1_weighted", "f1_macro"]
        metric_labels = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]

        x = np.arange(len(metrics))
        width = 0.18

        for i, mk in enumerate(models_list):
            if mk == "ensemble":
                m = ens_data
            else:
                m = exp_data.get(mk, {}).get(lang, {})

            values = [m.get(k, 0) for k in metrics]
            color = MODEL_COLORS.get(mk, '#999')
            bars = ax.bar(x + i * width, values, width,
                          label=MODEL_CONFIGS.get(mk, {}).get("name", "Ensemble"),
                          color=color, edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                            fontweight='bold')

        ax.set_title(f'{lang.capitalize()}', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.2)

        if idx == 0:
            ax.set_ylabel('Score', fontsize=11)
            ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('Monolingual Results: MuRIL vs XLM-R vs mBERT vs Ensemble',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_monolingual_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1 saved: {path}")


def figure2_cross_lingual_transfer(results):
    """
    Figure 2: Cross-lingual transfer matrix heatmap.
    Rows = train language(s), Columns = test language.
    """
    if "individual" not in results:
        print("  ⚠ No results for Figure 2.")
        return

    individual = results["individual"]
    ensemble = results.get("ensemble", {})

    # Build transfer matrix: best F1-W for each (train_config, test_lang) pair
    train_configs = [
        ("mono_tamil", "Tamil only"),
        ("mono_malayalam", "Malayalam only"),
        ("mono_kannada", "Kannada only"),
        ("cross_ta_ml", "Tamil+Malayalam"),
        ("cross_ta_kn", "Tamil+Kannada"),
        ("cross_ml_kn", "Mal+Kannada"),
        ("multi_all", "All three"),
    ]
    test_langs = ["tamil", "malayalam", "kannada"]
    test_lang_labels = ["Tamil", "Malayalam", "Kannada"]

    matrix = np.zeros((len(train_configs), len(test_langs)))

    for i, (exp, _) in enumerate(train_configs):
        # Use ensemble if available, else best individual
        ens = ensemble.get(exp, {})
        for j, lang in enumerate(test_langs):
            if lang in ens:
                matrix[i, j] = ens[lang].get("f1_weighted", 0)
            else:
                # Check individual models
                exp_data = individual.get(exp, {})
                best_f1 = 0
                for mk in MODEL_CONFIGS:
                    f1 = exp_data.get(mk, {}).get(lang, {}).get("f1_weighted", 0)
                    best_f1 = max(best_f1, f1)
                matrix[i, j] = best_f1

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=test_lang_labels,
                yticklabels=[label for _, label in train_configs],
                ax=ax, vmin=0, vmax=1,
                linewidths=0.5, linecolor='white')
    ax.set_title('Cross-Lingual Transfer Matrix (F1-Weighted)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Test Language', fontsize=11)
    ax.set_ylabel('Training Configuration', fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_cross_lingual_transfer.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2 saved: {path}")


def figure3_per_class_f1(results):
    """
    Figure 3: Per-class F1 across languages (monolingual ensemble).
    """
    ensemble = results.get("ensemble", {})
    if not ensemble:
        print("  ⚠ No ensemble results for Figure 3.")
        return

    mono_exps = {
        "Tamil": ("mono_tamil", "tamil"),
        "Malayalam": ("mono_malayalam", "malayalam"),
        "Kannada": ("mono_kannada", "kannada"),
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(UNIFIED_LABELS))
    width = 0.25

    for i, (lang_label, (exp, lang)) in enumerate(mono_exps.items()):
        ens = ensemble.get(exp, {}).get(lang, {})
        f1_vals = [ens.get(f"f1_{name}", 0) for name in UNIFIED_LABELS]
        ax.bar(x + i * width, f1_vals, width,
               label=lang_label, color=list(LANG_COLORS.values())[i],
               edgecolor='white', linewidth=0.5)
        for xi, val in zip(x + i * width, f1_vals):
            if val > 0:
                ax.text(xi, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Per-Class F1 Score by Language (Ensemble)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(SHORT_LABELS, fontsize=9, rotation=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_per_class_f1.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 3 saved: {path}")


def figure4_multilingual_vs_monolingual(results):
    """
    Figure 4: Multi-lingual (Exp D) vs Monolingual — does combining help?
    """
    ensemble = results.get("ensemble", {})
    if not ensemble:
        print("  ⚠ No ensemble results for Figure 4.")
        return

    langs = ["tamil", "malayalam", "kannada"]
    lang_labels = ["Tamil", "Malayalam", "Kannada"]
    mono_exps = ["mono_tamil", "mono_malayalam", "mono_kannada"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(langs))
    width = 0.3

    # Monolingual F1-W
    mono_vals = []
    for exp, lang in zip(mono_exps, langs):
        f1w = ensemble.get(exp, {}).get(lang, {}).get("f1_weighted", 0)
        mono_vals.append(f1w)

    # Multilingual F1-W
    multi_vals = []
    for lang in langs:
        f1w = ensemble.get("multi_all", {}).get(lang, {}).get("f1_weighted", 0)
        multi_vals.append(f1w)

    bars1 = ax.bar(x - width/2, mono_vals, width, label='Monolingual',
                   color='#2563eb', edgecolor='white')
    bars2 = ax.bar(x + width/2, multi_vals, width, label='Multilingual (All 3)',
                   color='#7c3aed', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_ylabel('F1 (Weighted)', fontsize=11)
    ax.set_title('Monolingual vs Multilingual Training (Ensemble)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_multilingual_vs_monolingual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 4 saved: {path}")


def main():
    print("=" * 60)
    print("  GENERATING PAPER FIGURES")
    print("=" * 60)

    results = load_results()

    if not results:
        print("  ⚠ No results found. Run evaluation scripts first.")
        return

    figure1_monolingual_comparison(results)
    figure2_cross_lingual_transfer(results)
    figure3_per_class_f1(results)
    figure4_multilingual_vs_monolingual(results)

    print(f"\n  ✅ All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
