"""Generate LIME explanation figures for the paper from saved data."""
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# LIME data from explanation_summary.csv
samples = [
    {
        'text': '1k dislikes yaaru da nengala',
        'true': 'Not_offensive', 'pred': 'Offensive_Untargeted', 'correct': False,
        'features': [('dislikes', 0.427), ('da', 0.117), ('1k', 0.074), ('yaaru', 0.057), ('nengala', 0.054)]
    },
    {
        'text': 'ithu sunda podara neram illa moviekku support neram',
        'true': 'Not_offensive', 'pred': 'Offensive_Untargeted', 'correct': False,
        'features': [('sunda', 0.239), ('podara', 0.074), ('ithu', 0.060), ('illa', 0.031), ('neram', -0.106), ('support', -0.097), ('moviekku', -0.061)]
    },
    {
        'text': 'yeanda ivlo stylish aana role ku punch dialogue kuduthu saavadippeenga',
        'true': 'Offensive_Untargeted', 'pred': 'Offensive_Untargeted', 'correct': True,
        'features': [('saavadippeenga', 0.333), ('yeanda', 0.150), ('dialogue', 0.124), ('aana', 0.043), ('ivlo', 0.037), ('role', 0.029), ('punch', -0.015), ('kuduthu', -0.048), ('stylish', -0.062), ('ku', -0.087)]
    }
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (sample, ax) in enumerate(zip(samples, axes)):
    features = sorted(sample['features'], key=lambda x: x[1])
    words = [f[0] for f in features]
    weights = [f[1] for f in features]
    colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]
    
    bars = ax.barh(words, weights, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Feature Weight', fontsize=10)
    
    status = 'CORRECT' if sample['correct'] else 'MISCLASSIFIED'
    status_color = '#27ae60' if sample['correct'] else '#c0392b'
    
    ax.set_title(f'({chr(97+idx)}) {status}\nTrue: {sample["true"]}\nPred: {sample["pred"]}',
                 fontsize=9, color=status_color, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x', alpha=0.2)
    ax.set_xlim(-0.15, 0.5)

plt.suptitle('LIME Word-Level Feature Importance for mBERT Predictions', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/lime_explanations.png', dpi=200, bbox_inches='tight')
print("Saved: figures/lime_explanations.png")
plt.close()

# Also generate training comparison chart
models = ['MuRIL', 'XLM-RoBERTa', 'mBERT']
val_f1w = [0.7309, 0.7287, 0.7322]
test_f1w = [0.7310, 0.7275, 0.7505]
runtimes = [1605, 2365, 1603]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(len(models))
width = 0.3
bars1 = ax1.bar(x - width/2, val_f1w, width, label='Validation F1-W', color='#3498db', edgecolor='white')
bars2 = ax1.bar(x + width/2, test_f1w, width, label='Test F1-W', color='#e74c3c', edgecolor='white')

for bar, val in zip(bars1, val_f1w):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, test_f1w):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax1.set_ylabel('F1-Weighted Score')
ax1.set_title('Validation vs Test F1-Weighted')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.set_ylim(0.70, 0.77)
ax1.grid(axis='y', alpha=0.2)

# Runtime comparison
bars3 = ax2.bar(models, [r/60 for r in runtimes], color=['#3498db', '#e67e22', '#e74c3c'], edgecolor='white')
for bar, val in zip(bars3, runtimes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val/60:.1f} min', ha='center', va='bottom', fontsize=9)
ax2.set_ylabel('Training Time (minutes)')
ax2.set_title('Training Time Comparison (Tesla T4)')
ax2.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('figures/training_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: figures/training_analysis.png")
plt.close()
