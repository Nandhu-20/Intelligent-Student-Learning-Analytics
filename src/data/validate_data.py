"""
Data quality validation for Intelligent Student Learning Analytics.
"""

import pandas as pd


EXPECTED_ROWS = 4424
EXPECTED_FEATURES = 36

VALID_TARGETS = {
    "Dropout",
    "Enrolled",
    "Graduate",
}


def validate_student_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict:
    """
    Validate the structure and basic quality of student data.

    Returns a dictionary containing validation results.
    Raises ValueError when a required check fails.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")

    if not isinstance(target, pd.Series):
        raise TypeError("Target must be a pandas Series.")

    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")

    if features.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}."
        )

    if features.isna().any().any():
        raise ValueError("Features contain missing values.")

    if target.isna().any():
        raise ValueError("Target contains missing values.")

    if features.duplicated().any():
        raise ValueError("Duplicate feature rows were found.")

    observed_targets = set(target.unique())
    if observed_targets != VALID_TARGETS:
        raise ValueError(
            f"Unexpected target classes: {observed_targets}"
        )

    return {
        "rows": len(features),
        "feature_count": features.shape[1],
        "missing_feature_values": int(
            features.isna().sum().sum()
        ),
        "duplicate_feature_rows": int(
            features.duplicated().sum()
        ),
        "target_classes": sorted(observed_targets),
        "status": "passed",
    }
