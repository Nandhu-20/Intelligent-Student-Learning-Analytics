import numpy as np
import pytest

from src.evaluation.metrics import evaluate_classifier


def test_perfect_predictions():
    actual = ["Dropout", "Enrolled", "Graduate"]
    predicted = ["Dropout", "Enrolled", "Graduate"]

    result = evaluate_classifier(actual, predicted)

    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["macro_f1"] == 1.0
    assert result["confusion_matrix"].shape == (3, 3)


def test_incorrect_predictions_are_penalized():
    actual = ["Dropout", "Graduate"]
    predicted = ["Graduate", "Graduate"]

    result = evaluate_classifier(actual, predicted)

    assert result["metrics"]["accuracy"] == 0.5
    assert result["metrics"]["macro_f1"] < 1.0


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        evaluate_classifier([], [])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_classifier(["Dropout"], ["Graduate", "Enrolled"])


def test_confusion_matrix_contains_all_classes():
    actual = ["Dropout", "Enrolled", "Graduate"]
    predicted = ["Dropout", "Dropout", "Graduate"]

    result = evaluate_classifier(actual, predicted)

    assert np.trace(result["confusion_matrix"]) == 2
