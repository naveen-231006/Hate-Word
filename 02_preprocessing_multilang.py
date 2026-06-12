"""
02_preprocessing_multilang.py — Multi-Language Preprocessing
=============================================================
Cleans and preprocesses Tamil, Malayalam, and Kannada datasets.
Supports all three languages + combined multilingual preprocessing.

Usage:
    python 02_preprocessing_multilang.py --lang tamil
    python 02_preprocessing_multilang.py --lang malayalam
    python 02_preprocessing_multilang.py --lang kannada
    python 02_preprocessing_multilang.py --lang all
"""

import os
import re
import argparse
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.utils import resample

from lang_config import (
    LANGUAGE_CONFIGS, UNIFIED_LABELS, LABEL2ID, NUM_LABELS,
    ALL_UNICODE_RANGES, normalize_label_name,
)

# ──────────────────────────────────────────────
# Preprocessing functions
# ──────────────────────────────────────────────

def clean_text(text):
    """
    Clean a single text sample.
    Preserves Tamil, Malayalam, AND Kannada Unicode characters.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Convert hashtags to words
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove non-alphanumeric, non-Dravidian characters (keep spaces)
    # Preserves: Tamil (0B80-0BFF), Kannada (0C80-0CFF), Malayalam (0D00-0D7F)
    text = re.sub(r'[^\w\s\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F]', ' ', text)

    # Lowercase only Latin characters
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
# Label normalization
# ──────────────────────────────────────────────

def normalize_labels(df, raw_label_names):
    """
    Convert raw integer labels → unified integer labels.
    raw_label_names: list of label strings from the HF dataset config.
    """
    def map_label(raw_id):
        raw_name = raw_label_names[raw_id]
        unified_name = normalize_label_name(raw_name)
        return LABEL2ID[unified_name]

    df["label"] = df["label"].apply(map_label)
    return df


# ──────────────────────────────────────────────
# Oversampling for minority classes
# ──────────────────────────────────────────────

def oversample_minority_classes(df, min_ratio=0.15, random_state=42):
    """
    Oversample minority classes so each has at least `min_ratio` of the
    majority class count.
    """
    majority_count = df["label"].value_counts().max()
    target_min = int(majority_count * min_ratio)

    parts = []
    for label_id in sorted(df["label"].unique()):
        class_df = df[df["label"] == label_id]
        label_name = UNIFIED_LABELS[label_id] if label_id < len(UNIFIED_LABELS) else f"class_{label_id}"
        if len(class_df) == 0:
            print(f"    ✗ {label_name}: 0 samples (skipping)")
            continue
        if len(class_df) < target_min:
            oversampled = resample(
                class_df,
                replace=True,
                n_samples=target_min,
                random_state=random_state,
            )
            parts.append(oversampled)
            print(f"    ↑ {label_name}: {len(class_df):,} → {target_min:,}")
        else:
            parts.append(class_df)
            print(f"    = {label_name}: {len(class_df):,} (unchanged)")

    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


# ──────────────────────────────────────────────
# Process a single language
# ──────────────────────────────────────────────

def process_language(lang_key, output_base="outputs/preprocessed"):
    """Preprocess one language and save to disk."""
    lang_cfg = LANGUAGE_CONFIGS[lang_key]
    lang_name = lang_cfg["name"]
    hf_config = lang_cfg["hf_config"]
    output_dir = os.path.join(output_base, lang_key)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Preprocessing: {lang_name}")
    print(f"{'='*60}")

    # 1. Load dataset
    print(f"\n[1/6] Loading {lang_name} dataset...")
    dataset = load_dataset("community-datasets/offenseval_dravidian", hf_config)
    train_data = dataset["train"]
    val_data = dataset["validation"]
    print(f"  ✓ Train: {len(train_data):,} | Validation: {len(val_data):,}")

    # Get raw label names from the dataset features
    raw_label_names = train_data.features["label"].names
    print(f"  Raw labels: {raw_label_names}")

    # 2. Clean text
    print(f"\n[2/6] Cleaning text...")
    train_cleaned = preprocess_dataset(train_data)
    val_cleaned = preprocess_dataset(val_data)

    # 3. Convert to DataFrame and normalize labels
    print(f"\n[3/6] Normalizing labels to unified schema...")
    train_df = pd.DataFrame({"text": train_cleaned["text"], "label": train_cleaned["label"]})
    val_df = pd.DataFrame({"text": val_cleaned["text"], "label": val_cleaned["label"]})

    train_df = normalize_labels(train_df, raw_label_names)
    val_df = normalize_labels(val_df, raw_label_names)

    # Remove empty texts
    before_train = len(train_df)
    before_val = len(val_df)
    train_df = train_df[train_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    val_df = val_df[val_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"  ✓ Removed {before_train - len(train_df)} empty train samples")
    print(f"  ✓ Removed {before_val - len(val_df)} empty validation samples")

    # 4. Split validation into val/test (50/50)
    print(f"\n[4/6] Creating val/test split...")
    val_df_shuffled = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = len(val_df_shuffled) // 2
    new_val_df = val_df_shuffled[:split_idx].reset_index(drop=True)
    test_df = val_df_shuffled[split_idx:].reset_index(drop=True)

    print(f"  ✓ Train: {len(train_df):,}")
    print(f"  ✓ Val:   {len(new_val_df):,}")
    print(f"  ✓ Test:  {len(test_df):,}")

    # 5. Oversample minority classes in training data
    print(f"\n[5/6] Oversampling minority classes...")
    train_df_orig_len = len(train_df)
    train_df = oversample_minority_classes(train_df, min_ratio=0.15)
    print(f"  ✓ Training: {train_df_orig_len:,} → {len(train_df):,}")

    # 6. Save
    print(f"\n[6/6] Saving to {output_dir}/...")

    # CSV
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    new_val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"  ✓ CSVs saved")

    # HuggingFace Dataset
    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(new_val_df),
        "test": Dataset.from_pandas(test_df),
    })
    hf_dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
    print(f"  ✓ HuggingFace Dataset saved")

    # Print distribution
    print(f"\n  Label distribution ({lang_name}):")
    for split_name, df in [("Train (oversampled)", train_df), ("Val", new_val_df), ("Test", test_df)]:
        print(f"\n  {split_name}:")
        for label_id in range(NUM_LABELS):
            count = (df["label"] == label_id).sum()
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"    {UNIFIED_LABELS[label_id]:<35s} {count:>5,d} ({pct:5.1f}%)")

    print(f"\n  ✅ {lang_name} preprocessing complete!")
    return train_df, new_val_df, test_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess Dravidian hate speech datasets")
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=["tamil", "malayalam", "kannada", "all"],
        help="Which language to preprocess (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MULTI-LANGUAGE PREPROCESSING")
    print("=" * 60)

    langs = list(LANGUAGE_CONFIGS.keys()) if args.lang == "all" else [args.lang]

    all_data = {}
    for lang in langs:
        train_df, val_df, test_df = process_language(lang)
        all_data[lang] = {"train": train_df, "val": val_df, "test": test_df}

    # If processing all, also create a combined dataset
    if args.lang == "all":
        print(f"\n{'='*60}")
        print(f"  Creating COMBINED multilingual dataset")
        print(f"{'='*60}")

        combined_dir = "outputs/preprocessed/combined"
        os.makedirs(combined_dir, exist_ok=True)

        combined_train = []
        combined_val = []
        combined_test = []

        for lang in langs:
            t = all_data[lang]["train"].copy()
            v = all_data[lang]["val"].copy()
            te = all_data[lang]["test"].copy()
            t["language"] = lang
            v["language"] = lang
            te["language"] = lang
            combined_train.append(t)
            combined_val.append(v)
            combined_test.append(te)

        combined_train_df = pd.concat(combined_train).sample(frac=1, random_state=42).reset_index(drop=True)
        combined_val_df = pd.concat(combined_val).sample(frac=1, random_state=42).reset_index(drop=True)
        combined_test_df = pd.concat(combined_test).reset_index(drop=True)

        combined_train_df.to_csv(os.path.join(combined_dir, "train.csv"), index=False)
        combined_val_df.to_csv(os.path.join(combined_dir, "val.csv"), index=False)
        combined_test_df.to_csv(os.path.join(combined_dir, "test.csv"), index=False)

        # HF Dataset (drop language column for tokenization)
        hf_train = combined_train_df[["text", "label"]].copy()
        hf_val = combined_val_df[["text", "label"]].copy()
        hf_test = combined_test_df[["text", "label"]].copy()

        hf_dataset = DatasetDict({
            "train": Dataset.from_pandas(hf_train),
            "validation": Dataset.from_pandas(hf_val),
            "test": Dataset.from_pandas(hf_test),
        })
        hf_dataset.save_to_disk(os.path.join(combined_dir, "hf_dataset"))

        print(f"  ✓ Combined Train: {len(combined_train_df):,}")
        print(f"  ✓ Combined Val:   {len(combined_val_df):,}")
        print(f"  ✓ Combined Test:  {len(combined_test_df):,}")
        print(f"  ✓ Saved to {combined_dir}/")

        # Also create cross-lingual combined datasets
        cross_configs = {
            "cross_ta_ml": ["tamil", "malayalam"],
            "cross_ta_kn": ["tamil", "kannada"],
            "cross_ml_kn": ["malayalam", "kannada"],
        }

        for cross_name, cross_langs in cross_configs.items():
            cross_dir = os.path.join("outputs/preprocessed", cross_name)
            os.makedirs(cross_dir, exist_ok=True)

            cross_train = []
            cross_val = []
            for lang in cross_langs:
                t = all_data[lang]["train"].copy()
                v = all_data[lang]["val"].copy()
                t["language"] = lang
                v["language"] = lang
                cross_train.append(t)
                cross_val.append(v)

            cross_train_df = pd.concat(cross_train).sample(frac=1, random_state=42).reset_index(drop=True)
            cross_val_df = pd.concat(cross_val).sample(frac=1, random_state=42).reset_index(drop=True)

            cross_train_df.to_csv(os.path.join(cross_dir, "train.csv"), index=False)
            cross_val_df.to_csv(os.path.join(cross_dir, "val.csv"), index=False)

            hf_t = cross_train_df[["text", "label"]].copy()
            hf_v = cross_val_df[["text", "label"]].copy()
            hf_dataset = DatasetDict({
                "train": Dataset.from_pandas(hf_t),
                "validation": Dataset.from_pandas(hf_v),
            })
            hf_dataset.save_to_disk(os.path.join(cross_dir, "hf_dataset"))

            print(f"  ✓ {cross_name} ({'+'.join(cross_langs)}): "
                  f"Train={len(cross_train_df):,}, Val={len(cross_val_df):,}")

    print(f"\n{'='*60}")
    print(f"  ✅ All preprocessing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
