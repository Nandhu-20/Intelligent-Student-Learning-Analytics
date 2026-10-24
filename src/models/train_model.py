"""
Reproducible admission-stage model training pipeline.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_student_dataset
from src.data.validate_data import validate_student_data
from src.evaluation.metrics import evaluate_classifier


RANDOM_STATE = 42
TEST_SIZE = 0.20

ADMISSION_FEATURES = [
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Nacionality",
    "International",
    "Gender",
]

NUMERICAL_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
]

CATEGORICAL_FEATURES = [
    feature for feature in ADMISSION_FEATURES
    if feature not in NUMERICAL_FEATURES
]


def train_and_evaluate(output_dir=None):
    """Train and evaluate an admission-stage Logistic Regression model."""
    project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if output_dir is None:
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_student_dataset()
    validate_student_data(X, y)

    X_admission = X[ADMISSION_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_admission,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    evaluation = evaluate_classifier(y_test, predictions)

    model_path = output_dir / "admission_logistic_regression.pkl"
    joblib.dump(model, model_path)

    metrics_path = results_dir / "admission_model_metrics.json"
    metrics_path.write_text(
        json.dumps(evaluation["metrics"], indent=2),
        encoding="utf-8",
    )

    report_path = results_dir / "admission_classification_report.csv"
    evaluation["classification_report"].to_csv(report_path)

    confusion_path = results_dir / "admission_confusion_matrix.csv"
    pd.DataFrame(
        evaluation["confusion_matrix"],
        index=model.classes_,
        columns=model.classes_,
    ).to_csv(confusion_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_matrix_path": str(confusion_path),
        "test_rows": len(y_test),
        "metrics": evaluation["metrics"],
        "classification_report": evaluation["classification_report"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }


if __name__ == "__main__":
    result = train_and_evaluate()

    print("Training completed.")
    print("Test rows:", result["test_rows"])
    print("Metrics:", result["metrics"])
    print("Model saved:", result["model_path"])
    print("Metrics saved:", result["metrics_path"])
    print("\nClassification report:")
    print(result["classification_report"])
    print("\nConfusion matrix:")
    print(result["confusion_matrix"])
