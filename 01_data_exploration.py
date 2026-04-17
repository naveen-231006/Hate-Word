"""
Phase 1: Dataset Download & Exploration
========================================
Downloads the OffensEval Dravidian (Tamil) dataset from Hugging Face
and generates exploration visualizations.

Usage:
    python 01_data_exploration.py
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "outputs/exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = [
    "Not_offensive",
    "Offensive_Untargeted",
    "Offensive_Targeted_Individual",
    "Offensive_Targeted_Group",
    "Offensive_Targeted_Other",
    "not-Tamil",
]

# Colors for consistent plotting
COLORS = {
    "Not_offensive": "#2ecc71",
    "Offensive_Untargeted": "#e74c3c",
    "Offensive_Targeted_Individual": "#e67e22",
    "Offensive_Targeted_Group": "#9b59b6",
    "Offensive_Targeted_Other": "#f39c12",
    "not-Tamil": "#95a5a6",
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def detect_script(text):
    """Classify text as Tamil, Latin (English/Tanglish), or Mixed based on character frequency."""
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total = tamil_chars + latin_chars
    if total == 0:
        return "Other"
    tamil_ratio = tamil_chars / total
    if tamil_ratio > 0.7:
        return "Tamil"
    elif tamil_ratio < 0.3:
        return "Latin (Tanglish/English)"
    else:
        return "Mixed (Code-Switched)"


def set_plot_style():
    """Configure matplotlib for publication-quality dark-themed plots."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "savefig.facecolor": "#1a1a2e",
        "savefig.dpi": 150,
        "figure.figsize": (12, 7),
    })


# ──────────────────────────────────────────────
# Main exploration
# ──────────────────────────────────────────────

def main():
    set_plot_style()

    print("=" * 60)
    print("  PHASE 1: Dataset Download & Exploration")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/6] Loading dataset from Hugging Face...")
    dataset = load_dataset("community-datasets/offenseval_dravidian", "tamil")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    print(f"  ✓ Train samples: {len(train_data):,}")
    print(f"  ✓ Validation samples: {len(val_data):,}")
    print(f"  ✓ Features: {train_data.features}")

    # 2. Label distribution
    print("\n[2/6] Analyzing label distribution...")
    train_labels = train_data["label"]
    label_counts = Counter(train_labels)

    print("\n  Label Distribution (Train):")
    print("  " + "-" * 55)
    total = len(train_labels)
    for label_id in sorted(label_counts.keys()):
        count = label_counts[label_id]
        pct = count / total * 100
        name = LABEL_NAMES[label_id]
        bar = "█" * int(pct)
        print(f"  {name:<35s} {count:>6,d}  ({pct:5.1f}%) {bar}")

    # 3. Plot label distribution
    print("\n[3/6] Generating label distribution plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    names = [LABEL_NAMES[lid] for lid, _ in sorted_labels]
    counts = [c for _, c in sorted_labels]
    colors = [COLORS[n] for n in names]

    bars = ax1.barh(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Number of Samples")
    ax1.set_title("Label Distribution (Train Set)")
    ax1.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                 f"{count:,}", va="center", fontsize=11, color="white")

    # Pie chart
    ax2.pie(counts, labels=names, colors=colors, autopct="%1.1f%%",
            startangle=140, textprops={"fontsize": 10, "color": "white"},
            wedgeprops={"edgecolor": "white", "linewidth": 0.5})
    ax2.set_title("Label Proportions")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/label_distribution.png")

    # 4. Text length analysis
    print("\n[4/6] Analyzing text lengths...")
    df_train = pd.DataFrame({"text": train_data["text"], "label": train_data["label"]})
    df_train["label_name"] = df_train["label"].map(lambda x: LABEL_NAMES[x])
    df_train["char_length"] = df_train["text"].str.len()
    df_train["word_count"] = df_train["text"].str.split().str.len()

    print(f"\n  Text Length Statistics:")
    print(f"  {'Metric':<20s} {'Chars':>10s} {'Words':>10s}")
    print(f"  {'-'*40}")
    print(f"  {'Mean':<20s} {df_train['char_length'].mean():>10.1f} {df_train['word_count'].mean():>10.1f}")
    print(f"  {'Median':<20s} {df_train['char_length'].median():>10.1f} {df_train['word_count'].median():>10.1f}")
    print(f"  {'Max':<20s} {df_train['char_length'].max():>10d} {df_train['word_count'].max():>10d}")
    print(f"  {'Min':<20s} {df_train['char_length'].min():>10d} {df_train['word_count'].min():>10d}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Character length distribution
    for label_name in LABEL_NAMES:
        subset = df_train[df_train["label_name"] == label_name]["char_length"]
        if len(subset) > 0:
            ax1.hist(subset, bins=50, alpha=0.6, label=label_name,
                     color=COLORS[label_name], edgecolor="none")
    ax1.set_xlabel("Character Length")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Character Length Distribution by Label")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(0, 500)

    # Word count distribution
    for label_name in LABEL_NAMES:
        subset = df_train[df_train["label_name"] == label_name]["word_count"]
        if len(subset) > 0:
            ax2.hist(subset, bins=50, alpha=0.6, label=label_name,
                     color=COLORS[label_name], edgecolor="none")
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Word Count Distribution by Label")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "text_length_distribution.png"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/text_length_distribution.png")

    # 5. Script detection
    print("\n[5/6] Detecting script types (Tamil / Latin / Mixed)...")
    df_train["script"] = df_train["text"].apply(detect_script)
    script_counts = df_train["script"].value_counts()

    print("\n  Script Distribution:")
    print("  " + "-" * 40)
    for script, count in script_counts.items():
        pct = count / len(df_train) * 100
        print(f"  {script:<30s} {count:>6,d}  ({pct:5.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    script_colors = ["#00d2ff", "#f5af19", "#fc466b", "#95a5a6"]
    bars = ax.bar(script_counts.index, script_counts.values,
                  color=script_colors[:len(script_counts)],
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Script Type Distribution in Training Data")
    for bar, count in zip(bars, script_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{count:,}", ha="center", fontsize=12, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "script_distribution.png"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/script_distribution.png")

    # 6. Sample rows
    print("\n[6/6] Sample rows per label:")
    print("=" * 80)
    for label_id in range(len(LABEL_NAMES)):
        label_name = LABEL_NAMES[label_id]
        samples = df_train[df_train["label"] == label_id]["text"].head(3).tolist()
        print(f"\n  📌 {label_name} (label={label_id}):")
        for i, s in enumerate(samples, 1):
            print(f"     {i}. {s[:120]}{'...' if len(s) > 120 else ''}")

    # Cross-tabulation: script vs label
    print("\n\n  Script × Label Cross-Tabulation:")
    ct = pd.crosstab(df_train["script"], df_train["label_name"])
    print(ct.to_string(index=True))

    fig, ax = plt.subplots(figsize=(14, 7))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", stacked=True, ax=ax,
                color=[COLORS[c] for c in ct_pct.columns],
                edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Label Distribution by Script Type")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "script_vs_label.png"), bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved: {OUTPUT_DIR}/script_vs_label.png")

    print("\n" + "=" * 60)
    print("  ✅ Phase 1 Complete! All plots saved to outputs/exploration/")
    print("=" * 60)


if __name__ == "__main__":
    main()
