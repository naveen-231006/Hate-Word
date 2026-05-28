"""
Utility functions for training Tamil hate speech detection models.
Includes custom weighted trainer, focal loss trainer, metrics computation,
and label mappings.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ──────────────────────────────────────────────
# Label definitions
# ──────────────────────────────────────────────

LABEL_NAMES = [
    "Not_offensive",
    "Offensive_Untargeted",
    "Offensive_Targeted_Individual",
    "Offensive_Targeted_Group",
    "Offensive_Targeted_Other",
    "not-Tamil",
]

LABEL2ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: name for idx, name in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)


# ──────────────────────────────────────────────
# Metrics (with per-class F1 tracking)
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 (macro & weighted) + per-class F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # Per-class F1 for tracking minority class performance during training
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0,
        labels=list(range(NUM_LABELS))
    )

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }

    # Add per-class F1 with short names for readable training logs
    short_names = ["not_off", "untarg", "t_indiv", "t_group", "t_other", "not_tam"]
    for i, short in enumerate(short_names):
        metrics[f"f1_{short}"] = f1_per_class[i]

    return metrics


# ──────────────────────────────────────────────
# Focal Loss (for extreme class imbalance)
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for handling class imbalance.

    Down-weights loss for well-classified (easy) examples and focuses
    training on hard, misclassified examples. Combined with class weights
    (alpha), this is far more effective than weighted CE alone for
    extreme imbalance (e.g. 72% vs 1.3% class distribution).

    Args:
        alpha: Per-class weights tensor (same as CE weights).
        gamma: Focusing parameter. gamma=0 is standard CE.
               gamma=2.0 is recommended for heavy imbalance.
        label_smoothing: Smoothing factor (0.0 = no smoothing).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FocalLossTrainer(Trainer):
    """Custom Trainer using Focal Loss for extreme class imbalance.

    Drop-in replacement for WeightedTrainer with significantly better
    performance on minority classes (Offensive_Targeted_Other, etc.).
    """

    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = FocalLoss(
            alpha=weight,
            gamma=self.gamma,
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────
# Legacy Weighted Trainer (kept for backward compatibility)
# ──────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Custom Trainer that uses weighted CrossEntropyLoss to handle class imbalance."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────
# Class weight computation
# ──────────────────────────────────────────────

def compute_class_weights(labels, num_classes=NUM_LABELS):
    """
    Compute inverse-frequency class weights for handling imbalanced datasets.
    Returns a list of weights, one per class.
    """
    label_counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Avoid division by zero
    label_counts = np.maximum(label_counts, 1.0)
    total = label_counts.sum()
    weights = total / (num_classes * label_counts)
    return weights.tolist()


def get_classification_report(y_true, y_pred, output_dict=False):
    """Generate a classification report with label names."""
    return classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=output_dict,
        zero_division=0,
    )


def get_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
