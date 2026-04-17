"""
07_ensemble.py — Ensemble the 3 models using majority voting.
This requires no GPU and uses only the already-generated predictions.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

# ============================================================
#  Configuration
# ============================================================
LABEL_NAMES = [
    'Not_offensive', 'Offensive_Untargeted', 'Offensive_Targeted_Individual',
    'Offensive_Targeted_Group', 'Offensive_Targeted_Other', 'not-Tamil'
]
SHORT_LABELS = ['Not Off.', 'Off. Untarg.', 'Off. Indiv.', 'Off. Group', 'Off. Other', 'not-Tamil']

MODEL_CONFIGS = {
    'muril': 'MuRIL',
    'xlm-roberta': 'XLM-RoBERTa',
    'mbert': 'mBERT',
}

PRED_CSV = 'outputs/evaluation/all_predictions.csv'
OUT_DIR  = 'outputs/evaluation'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
#  Load predictions
# ============================================================
print("Loading predictions from:", PRED_CSV)
df = pd.read_csv(PRED_CSV)

true_labels = df['true_label'].values
model_preds = {}
for mk in MODEL_CONFIGS:
    col = f'pred_{mk}'
    if col in df.columns:
        model_preds[mk] = df[col].values
        print(f"  {MODEL_CONFIGS[mk]}: {len(model_preds[mk])} predictions loaded")
    else:
        print(f"  WARNING: Column '{col}' not found!")

# ============================================================
#  Majority Voting Ensemble
# ============================================================
print("\n" + "="*60)
print("  ENSEMBLE — Majority Voting")
print("="*60)

# Stack predictions: shape (n_models, n_samples)
pred_stack = np.array([model_preds[mk] for mk in model_preds])

# Majority vote (axis=0 → vote across models for each sample)
ensemble_preds, _ = mode(pred_stack, axis=0, keepdims=False)
ensemble_preds = ensemble_preds.astype(int)

# ============================================================
#  Evaluation
# ============================================================
print("\n--- Classification Report (Ensemble) ---")
print(classification_report(
    true_labels, ensemble_preds,
    target_names=LABEL_NAMES, zero_division=0
))

# Compute metrics
report = classification_report(
    true_labels, ensemble_preds,
    target_names=LABEL_NAMES, output_dict=True, zero_division=0
)

ensemble_metrics = {
    'accuracy': report['accuracy'],
    'f1_weighted': report['weighted avg']['f1-score'],
    'f1_macro': report['macro avg']['f1-score'],
    'precision_weighted': report['weighted avg']['precision'],
    'recall_weighted': report['weighted avg']['recall'],
}

# Per-class F1
for i, name in enumerate(LABEL_NAMES):
    ensemble_metrics[f'f1_{name}'] = report.get(name, {}).get('f1-score', 0)

print(f"\nEnsemble Accuracy:    {ensemble_metrics['accuracy']:.4f}")
print(f"Ensemble F1-Weighted: {ensemble_metrics['f1_weighted']:.4f}")
print(f"Ensemble F1-Macro:    {ensemble_metrics['f1_macro']:.4f}")

# ============================================================
#  Compare against individual models
# ============================================================
# Load individual model metrics
with open('outputs/evaluation/comparison_metrics.json', 'r') as f:
    individual_metrics = json.load(f)

print("\n" + "="*60)
print("  COMPARISON: Individual Models vs Ensemble")
print("="*60)
print(f"{'Model':<20s} {'Accuracy':>10s} {'F1-W':>10s} {'F1-M':>10s}")
print("-"*50)
for mk, name in MODEL_CONFIGS.items():
    m = individual_metrics.get(mk, {})
    print(f"{name:<20s} {m.get('accuracy',0):>10.4f} {m.get('f1_weighted',0):>10.4f} {m.get('f1_macro',0):>10.4f}")
print("-"*50)
print(f"{'ENSEMBLE (Vote)':<20s} {ensemble_metrics['accuracy']:>10.4f} {ensemble_metrics['f1_weighted']:>10.4f} {ensemble_metrics['f1_macro']:>10.4f}")

# Calculate improvement over best single model
best_single_f1w = max(m.get('f1_weighted', 0) for m in individual_metrics.values())
improvement = ensemble_metrics['f1_weighted'] - best_single_f1w
print(f"\n{'Improvement over best single model:':<40s} {improvement:+.4f} F1-W")

# ============================================================
#  Confusion Matrix
# ============================================================
cm = confusion_matrix(true_labels, ensemble_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=ax)
ax.set_title('Ensemble (Majority Vote) — Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, 'ensemble_confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
print(f"\nConfusion matrix saved: {cm_path}")
plt.close()

# ============================================================
#  Updated Comparison Chart (with Ensemble)
# ============================================================
all_metrics = dict(individual_metrics)
all_metrics['ensemble'] = ensemble_metrics
model_names = {**MODEL_CONFIGS, 'ensemble': 'Ensemble (Vote)'}

metric_labels = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'Precision (W)', 'Recall (W)']
metric_keys = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']

fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(metric_labels))
width = 0.18
colors = ['#00d2ff', '#f5af19', '#fc466b', '#7c3aed']

for i, (mk, name) in enumerate(model_names.items()):
    m = all_metrics.get(mk, {})
    values = [m.get(k, 0) for k in metric_keys]
    bars = ax.bar(x + i * width, values, width, label=name,
                  color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Score')
ax.set_title('Model Comparison — Including Ensemble (Test Set)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'model_comparison_with_ensemble.png')
plt.savefig(chart_path, dpi=150)
print(f"Comparison chart saved: {chart_path}")
plt.close()

# ============================================================
#  Save all results
# ============================================================
# Save ensemble metrics
all_metrics_path = os.path.join(OUT_DIR, 'comparison_metrics_with_ensemble.json')
with open(all_metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Save ensemble predictions
df['pred_ensemble'] = ensemble_preds
df['pred_ensemble_name'] = [LABEL_NAMES[p] for p in ensemble_preds]
df.to_csv(os.path.join(OUT_DIR, 'all_predictions_with_ensemble.csv'), index=False)

print(f"Metrics saved: {all_metrics_path}")
print(f"Predictions saved: {os.path.join(OUT_DIR, 'all_predictions_with_ensemble.csv')}")

# ============================================================
#  Per-class F1 comparison
# ============================================================
print("\n" + "="*60)
print("  PER-CLASS F1 COMPARISON")
print("="*60)
print(f"{'Class':<35s} {'MuRIL':>8s} {'XLM-R':>8s} {'mBERT':>8s} {'Ensem.':>8s} {'Best':>8s}")
print("-"*75)
for i, name in enumerate(LABEL_NAMES):
    f1s = {}
    for mk in list(MODEL_CONFIGS.keys()) + ['ensemble']:
        f1s[mk] = all_metrics.get(mk, {}).get(f'f1_{name}', 0)
    best = max(f1s.values())
    best_mk = max(f1s, key=lambda k: f1s[k])
    star = lambda mk: '*' if mk == best_mk else ' '
    print(f"{name:<35s} {f1s.get('muril',0):>7.3f}{star('muril')} "
          f"{f1s.get('xlm-roberta',0):>7.3f}{star('xlm-roberta')} "
          f"{f1s.get('mbert',0):>7.3f}{star('mbert')} "
          f"{f1s.get('ensemble',0):>7.3f}{star('ensemble')} "
          f"{'<-- ' + model_names.get(best_mk, best_mk)}")

print("\n[DONE] Ensemble analysis complete!")
