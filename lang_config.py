"""
lang_config.py — Shared configuration for Tamil, Malayalam, and Kannada
======================================================================
Provides unified label mapping across all three OffensEval Dravidian configs,
Unicode ranges, and experiment definitions.
"""

# ──────────────────────────────────────────────
# Unified 6-class label schema
# ──────────────────────────────────────────────
# The raw dataset has slight naming differences across languages:
#   - "Offensive_Untargetede" (typo in dataset, with trailing 'e')
#   - "Offensive_Targeted_Insult_Individual" vs our "Offensive_Targeted_Individual"
#   - "not-Tamil" / "not-malayalam" / "not-Kannada" (inconsistent casing)
# We normalize everything to these 6 labels:

UNIFIED_LABELS = [
    "Not_offensive",
    "Offensive_Untargeted",
    "Offensive_Targeted_Individual",
    "Offensive_Targeted_Group",
    "Offensive_Targeted_Other",
    "not-target-language",
]

LABEL2ID = {name: idx for idx, name in enumerate(UNIFIED_LABELS)}
ID2LABEL = {idx: name for idx, name in enumerate(UNIFIED_LABELS)}
NUM_LABELS = len(UNIFIED_LABELS)

# Short display names for plots and tables
SHORT_LABELS = [
    "Not Off.", "Off. Untarg.", "Off. Indiv.",
    "Off. Group", "Off. Other", "not-LangX",
]

# ──────────────────────────────────────────────
# Raw dataset label → unified label mapping
# ──────────────────────────────────────────────
# Maps every raw label string from any language config to our unified label name.

RAW_TO_UNIFIED = {
    # Common across all languages
    "Not_offensive": "Not_offensive",
    "Offensive_Untargetede": "Offensive_Untargeted",        # typo in dataset
    "Offensive_Untargeted": "Offensive_Untargeted",         # in case it's fixed
    "Offensive_Targeted_Insult_Individual": "Offensive_Targeted_Individual",
    "Offensive_Targeted_Insult_Group": "Offensive_Targeted_Group",
    "Offensive_Targeted_Insult_Other": "Offensive_Targeted_Other",
    # Per-language "not-X" labels
    "not-Tamil": "not-target-language",
    "not-malayalam": "not-target-language",
    "not-Kannada": "not-target-language",
    # Fallbacks (your existing code's label names)
    "Offensive_Targeted_Individual": "Offensive_Targeted_Individual",
    "Offensive_Targeted_Group": "Offensive_Targeted_Group",
    "Offensive_Targeted_Other": "Offensive_Targeted_Other",
    "not-target-language": "not-target-language",
}


def normalize_label_name(raw_label_name):
    """Convert a raw dataset label string to our unified label name."""
    unified = RAW_TO_UNIFIED.get(raw_label_name)
    if unified is None:
        raise ValueError(f"Unknown label: '{raw_label_name}'. "
                         f"Known labels: {list(RAW_TO_UNIFIED.keys())}")
    return unified


def normalize_label_id(raw_label_id, raw_label_names):
    """Convert a raw integer label + the config's label names list → unified integer ID."""
    raw_name = raw_label_names[raw_label_id]
    unified_name = normalize_label_name(raw_name)
    return LABEL2ID[unified_name]


# ──────────────────────────────────────────────
# Language configurations
# ──────────────────────────────────────────────

LANGUAGE_CONFIGS = {
    "tamil": {
        "name": "Tamil",
        "hf_config": "tamil",                    # HuggingFace dataset config name
        "unicode_range": r"\u0B80-\u0BFF",       # Tamil Unicode block
        "not_label": "not-Tamil",                # raw dataset label
    },
    "malayalam": {
        "name": "Malayalam",
        "hf_config": "malayalam",
        "unicode_range": r"\u0D00-\u0D7F",       # Malayalam Unicode block
        "not_label": "not-malayalam",
    },
    "kannada": {
        "name": "Kannada",
        "hf_config": "kannada",
        "unicode_range": r"\u0C80-\u0CFF",       # Kannada Unicode block
        "not_label": "not-Kannada",
    },
}

# All three Unicode ranges combined (for clean_text to preserve any Dravidian script)
ALL_UNICODE_RANGES = r"\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F"

# ──────────────────────────────────────────────
# Model configurations (same for all languages)
# ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "muril": {
        "name": "MuRIL",
        "hf_id": "google/muril-base-cased",
        "max_length": 128,
    },
    "xlm-roberta": {
        "name": "XLM-RoBERTa",
        "hf_id": "xlm-roberta-base",
        "max_length": 128,
    },
    "mbert": {
        "name": "mBERT",
        "hf_id": "bert-base-multilingual-cased",
        "max_length": 128,
    },
}

# ──────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────

EXPERIMENTS = {
    # Monolingual (Step 3)
    "mono_tamil": {
        "description": "Exp A: Train Tamil → Test Tamil",
        "train_langs": ["tamil"],
        "test_langs": ["tamil"],
    },
    "mono_malayalam": {
        "description": "Exp B: Train Malayalam → Test Malayalam",
        "train_langs": ["malayalam"],
        "test_langs": ["malayalam"],
    },
    "mono_kannada": {
        "description": "Exp C: Train Kannada → Test Kannada",
        "train_langs": ["kannada"],
        "test_langs": ["kannada"],
    },
    # Multilingual (Step 3, Experiment D)
    "multi_all": {
        "description": "Exp D: Train All → Test Each",
        "train_langs": ["tamil", "malayalam", "kannada"],
        "test_langs": ["tamil", "malayalam", "kannada"],
    },
    # Cross-lingual (Step 4)
    "cross_ta_ml": {
        "description": "Cross: Train Tamil+Malayalam → Test Kannada",
        "train_langs": ["tamil", "malayalam"],
        "test_langs": ["kannada"],
    },
    "cross_ta_kn": {
        "description": "Cross: Train Tamil+Kannada → Test Malayalam",
        "train_langs": ["tamil", "kannada"],
        "test_langs": ["malayalam"],
    },
    "cross_ml_kn": {
        "description": "Cross: Train Malayalam+Kannada → Test Tamil",
        "train_langs": ["malayalam", "kannada"],
        "test_langs": ["tamil"],
    },
}

# ──────────────────────────────────────────────
# RTX 3050 (4GB VRAM) optimized training config
# ──────────────────────────────────────────────

GPU_3050_CONFIG = {
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 4,   # effective batch = 32
    "fp16": True,
    "max_length": 128,
}
