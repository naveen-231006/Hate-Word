# Offensive Language Detection in Tamil Social Media Text

A comparative study of multilingual transformer models (MuRIL, XLM-RoBERTa, mBERT) for detecting offensive language in Tamil and code-mixed Tamil-English (Tanglish) social media text, with majority-voting ensemble and LIME-based explainability.

## Key Results

| Model | Accuracy | F1-Weighted | F1-Macro |
|-------|----------|-------------|----------|
| MuRIL | 0.7115 | 0.7310 | 0.4405 |
| XLM-RoBERTa | 0.7019 | 0.7275 | 0.4743 |
| mBERT | 0.7288 | 0.7505 | 0.4972 |
| **Ensemble (Vote)** | **0.7502** | **0.7620** | 0.4947 |

**Key Finding:** MuRIL outperforms on pure Tamil Unicode text (F1-W: 0.771), but mBERT dominates on Latin/Tanglish (80% of data), making it the best overall model.

## Project Structure

```
├── 01_data_exploration.py    # Phase 1: EDA and distribution plots
├── 02_preprocessing.py       # Phase 2: Text cleaning and train/val/test split
├── 03_train.py               # Phase 3: Fine-tune MuRIL, XLM-R, mBERT
├── 04_evaluate.py            # Phase 4: Evaluation and confusion matrices
├── 05_error_analysis.py      # Phase 5: Error categorization
├── 06_explainability.py      # Phase 6: LIME word-level explanations
├── 07_ensemble.py            # Phase 7: Majority-voting ensemble
├── train_colab.ipynb         # Google Colab notebook (runs Phases 1-6)
├── utils/
│   └── trainer_utils.py      # Custom WeightedTrainer for class imbalance
├── paper/
│   ├── paper.tex             # IEEE-format LaTeX paper
│   ├── paper_draft.md        # Markdown draft
│   └── figures/              # All figures for the paper
├── outputs/
│   ├── evaluation/           # Metrics, confusion matrices, predictions
│   ├── analysis/             # Error analysis, LIME explanations
│   ├── exploration/          # EDA plots
│   └── models/               # Model checkpoints (not in repo)
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run training (Google Colab recommended)
Upload `train_colab.ipynb` to Google Colab with a T4 GPU runtime. The notebook runs Phases 1-6 automatically (~100 min total).

### 3. Run ensemble (local, no GPU needed)
```bash
python 07_ensemble.py
```

## Dataset

[OffensEval Dravidian Tamil](https://huggingface.co/datasets/ai4bharat/OffensEval-Dravidian) — 39,526 YouTube comments across 6 classes:
- Not_offensive (72.4%)
- Offensive_Untargeted (8.3%)
- Offensive_Targeted_Group (7.3%)
- Offensive_Targeted_Individual (6.7%)
- not-Tamil (4.1%)
- Offensive_Targeted_Other (1.3%)

## Models

| Model | HuggingFace ID | Parameters |
|-------|---------------|------------|
| MuRIL | `google/muril-base-cased` | 236M |
| XLM-RoBERTa | `xlm-roberta-base` | 278M |
| mBERT | `bert-base-multilingual-cased` | 177M |

## Paper

The IEEE-format LaTeX paper is in `paper/paper.tex`. Compile on [Overleaf](https://overleaf.com) by uploading the `paper/` directory.

## License

This project is for academic/research purposes.
