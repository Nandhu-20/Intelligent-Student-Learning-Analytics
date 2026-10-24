"""
Feature engineering utilities for student academic analytics.
"""

import numpy as np
import pandas as pd


def build_academic_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add seven academic-progress features to the original dataset.

    Note:
        These features use second-semester information.
        They are intended for academic-progress analysis,
        not admission-stage prediction.
    """
    required_columns = [
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = features.copy()

    result["Average_semester_grade"] = (
        result["Curricular units 1st sem (grade)"]
        + result["Curricular units 2nd sem (grade)"]
    ) / 2

    result["Grade_change"] = (
        result["Curricular units 2nd sem (grade)"]
        - result["Curricular units 1st sem (grade)"]
    )

    result["Total_enrolled_units"] = (
        result["Curricular units 1st sem (enrolled)"]
        + result["Curricular units 2nd sem (enrolled)"]
    )

    result["Total_evaluations"] = (
        result["Curricular units 1st sem (evaluations)"]
        + result["Curricular units 2nd sem (evaluations)"]
    )

    result["Total_approved_units"] = (
        result["Curricular units 1st sem (approved)"]
        + result["Curricular units 2nd sem (approved)"]
    )

    enrolled = result["Total_enrolled_units"]

    result["Approval_rate"] = np.divide(
        result["Total_approved_units"],
        enrolled,
        out=np.zeros(len(result), dtype=float),
        where=enrolled.to_numpy() > 0,
    )

    result["Evaluation_to_enrollment_ratio"] = np.divide(
        result["Total_evaluations"],
        enrolled,
        out=np.zeros(len(result), dtype=float),
        where=enrolled.to_numpy() > 0,
    )

    return result
