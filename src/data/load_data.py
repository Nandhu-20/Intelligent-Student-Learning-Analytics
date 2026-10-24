"""
Data loading utilities for the Intelligent Student Learning Analytics project.
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_student_dataset():
    """
    Download and load the UCI student dropout and academic success dataset.

    Returns:
        features (pd.DataFrame): The 36 original input features.
        target (pd.Series): The target outcome for each student.
    """
    dataset = fetch_ucirepo(id=697)

    features = dataset.data.features.copy()
    target_frame = dataset.data.targets.copy()

    if features.shape != (4424, 36):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}. "
            "Expected 4424 rows and 36 columns."
        )

    if len(target_frame) != len(features):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if target_frame.shape[1] != 1:
        raise ValueError(
            f"Expected one target column, got {target_frame.shape[1]}."
        )

    target = target_frame.iloc[:, 0].rename("Target")

    if target.isna().any():
        raise ValueError("The target contains missing values.")

    return features, target


if __name__ == "__main__":
    X, y = load_student_dataset()

    print("Dataset loaded successfully.")
    print("Features:", X.shape)
    print("Target:", y.shape)
    print("\nOutcome distribution:")
    print(y.value_counts())
