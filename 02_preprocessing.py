"""
Phase 2: Text Preprocessing
============================
Cleans and preprocesses the Tamil hate speech dataset.
Handles code-mixed text, creates train/val/test splits.

Usage:
    python 02_preprocessing.py
"""

import os
import re
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "outputs/preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = [
    "Not_offensive",
    "Offensive_Untargeted",
    "Offensive_Targeted_Individual",
    "Offensive_Targeted_Group",
    "Offensive_Targeted_Other",
    "not-Tamil",
]


# ──────────────────────────────────────────────
# Preprocessing functions
# ──────────────────────────────────────────────

def clean_text(text):
    """
    Clean a single text sample:
    - Remove URLs
    - Remove @mentions
    - Convert #hashtags to words
    - Remove special characters but keep Tamil Unicode + alphanumeric
    - Normalize whitespace
    - Lowercase Latin characters (preserve Tamil script as-is)
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Convert hashtags to words
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove non-alphanumeric, non-Tamil characters (keep spaces)
    # Tamil Unicode range: \u0B80-\u0BFF
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', ' ', text)

    # Lowercase only Latin characters (Tamil doesn't have case)
    result = []
    for char in text:
        if 'A' <= char <= 'Z':
            result.append(char.lower())
        else:
            result.append(char)
    text = ''.join(result)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset(dataset_split):
    """Apply cleaning to a dataset split."""
    cleaned_texts = [clean_text(text) for text in dataset_split["text"]]
    return dataset_split.remove_columns("text").add_column("text", cleaned_texts)


# ──────────────────────────────────────────────
# Main preprocessing pipeline
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE 2: Text Preprocessing")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset("community-datasets/offenseval_dravidian", "tamil")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    print(f"  ✓ Train: {len(train_data):,} | Validation: {len(val_data):,}")

    # 2. Show before/after examples
    print("\n[2/5] Preprocessing examples:")
    print("  " + "-" * 70)
    for i in range(5):
        original = train_data["text"][i]
        cleaned = clean_text(original)
        print(f"  ORIGINAL: {original[:80]}{'...' if len(original) > 80 else ''}")
        print(f"  CLEANED:  {cleaned[:80]}{'...' if len(cleaned) > 80 else ''}")
        print()

    # 3. Apply preprocessing
    print("[3/5] Applying preprocessing to all splits...")
    train_cleaned = preprocess_dataset(train_data)
    val_cleaned = preprocess_dataset(val_data)
    print(f"  ✓ Cleaned {len(train_cleaned):,} train + {len(val_cleaned):,} validation samples")

    # Remove empty texts after cleaning
    train_df = pd.DataFrame({"text": train_cleaned["text"], "label": train_cleaned["label"]})
    val_df = pd.DataFrame({"text": val_cleaned["text"], "label": val_cleaned["label"]})

    before_train = len(train_df)
    before_val = len(val_df)
    train_df = train_df[train_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    val_df = val_df[val_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"  ✓ Removed {before_train - len(train_df)} empty train samples")
    print(f"  ✓ Removed {before_val - len(val_df)} empty validation samples")

    # 4. Create train/val/test split (split validation 50/50)
    print("\n[4/5] Creating train/val/test split...")
    val_df_shuffled = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = len(val_df_shuffled) // 2
    new_val_df = val_df_shuffled[:split_idx].reset_index(drop=True)
    test_df = val_df_shuffled[split_idx:].reset_index(drop=True)

    print(f"  ✓ Train: {len(train_df):,}")
    print(f"  ✓ Val:   {len(new_val_df):,}")
    print(f"  ✓ Test:  {len(test_df):,}")

    # Verify no data leakage
    train_texts = set(train_df["text"].tolist())
    val_texts = set(new_val_df["text"].tolist())
    test_texts = set(test_df["text"].tolist())
    leakage_train_val = train_texts & val_texts
    leakage_train_test = train_texts & test_texts
    leakage_val_test = val_texts & test_texts
    print(f"\n  Data leakage check:")
    print(f"    Train ∩ Val:  {len(leakage_train_val)} overlapping samples")
    print(f"    Train ∩ Test: {len(leakage_train_test)} overlapping samples")
    print(f"    Val ∩ Test:   {len(leakage_val_test)} overlapping samples")

    # 5. Save preprocessed data
    print("\n[5/5] Saving preprocessed data...")

    # Save as CSV
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    new_val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    print(f"  ✓ CSVs saved to {OUTPUT_DIR}/")

    # Save as HuggingFace Dataset
    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(new_val_df),
        "test": Dataset.from_pandas(test_df),
    })
    hf_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "hf_dataset"))
    print(f"  ✓ HuggingFace Dataset saved to {OUTPUT_DIR}/hf_dataset/")

    # Print label distribution for each split
    print("\n  Label distributions per split:")
    for split_name, df in [("Train", train_df), ("Val", new_val_df), ("Test", test_df)]:
        print(f"\n  {split_name}:")
        for label_id in range(len(LABEL_NAMES)):
            count = (df["label"] == label_id).sum()
            pct = count / len(df) * 100
            print(f"    {LABEL_NAMES[label_id]:<35s} {count:>5,d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("  ✅ Phase 2 Complete! Preprocessed data saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
