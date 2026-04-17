"""
Utility functions for training Tamil hate speech detection models.
Includes custom weighted trainer, metrics computation, and label mappings.
"""

import torch
import numpy as np
from torch import nn
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
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 (macro & weighted) for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }


# ──────────────────────────────────────────────
# Weighted Trainer (for class imbalance)
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
