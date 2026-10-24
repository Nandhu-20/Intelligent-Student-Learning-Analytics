"""
Reusable classification evaluation metrics.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_classifier(y_true, y_pred):
    """
    Return aggregate metrics, a per-class report,
    and a confusion matrix for a classification model.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred)

    return {
        "metrics": metrics,
        "classification_report": pd.DataFrame(report).transpose(),
        "confusion_matrix": matrix,
    }
