
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "admission_logistic_regression.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "admission_model_metrics.json"
REPORT_PATH = PROJECT_ROOT / "results" / "admission_classification_report.csv"
MATRIX_PATH = PROJECT_ROOT / "results" / "admission_confusion_matrix.csv"

st.set_page_config(
    page_title="Intelligent Student Learning Analytics",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Intelligent Student Learning Analytics")
st.caption(
    "An interpretable, data-driven exploration of student outcomes "
    "and admission-stage predictions."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    report = pd.read_csv(REPORT_PATH, index_col=0)
    matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    return metrics, report, matrix


required_files = [MODEL_PATH, METRICS_PATH, REPORT_PATH, MATRIX_PATH]
missing_files = [
    str(path.relative_to(PROJECT_ROOT))
    for path in required_files
    if not path.is_file()
]

if missing_files:
    st.error("Required project files are missing:")
    for file_name in missing_files:
        st.write(f"- {file_name}")
    st.stop()

try:
    model = load_model()
    metrics, report, matrix = load_results()
except Exception as exc:
    st.error(f"Could not load the saved model or evaluation results: {exc}")
    st.stop()


overview_tab, performance_tab, prediction_tab, about_tab = st.tabs(
    ["Overview", "Model Performance", "Student Prediction", "About"]
)


with overview_tab:
    st.subheader("Project overview")
    st.write(
        "This application demonstrates student-outcome classification using "
        "admission-stage information. Predictions are estimates, not guarantees."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Original dataset rows", "4,424")
    col2.metric("Original features", "36")
    col3.metric("Outcome classes", "3")

    st.markdown("### Original dataset outcome distribution")
    outcome_counts = pd.Series({
        "Graduate": 2209,
        "Dropout": 1421,
        "Enrolled": 794,
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    outcome_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Student outcome")
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of outcomes in the original dataset")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The chart shows the dataset's observed outcomes, not predictions "
        "for new students."
    )


with performance_tab:
    st.subheader("Held-out test performance")

    metric_columns = st.columns(4)
    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1-score", "macro_f1"),
    ]

    for column, (label, key) in zip(metric_columns, metric_labels):
        column.metric(label, f"{metrics[key]:.1%}")

    st.markdown("### Classification report")
    st.dataframe(report, width="stretch")

    st.markdown("### Confusion matrix")
    st.caption("Rows are actual outcomes; columns are predicted outcomes.")
    st.dataframe(matrix, width="stretch")

    st.warning(
        "The model is a baseline. In particular, performance may be weaker "
        "for the Enrolled class. Evaluate class-wise metrics before using "
        "predictions to support real student interventions."
    )


with prediction_tab:
    st.subheader("Admission-stage prediction")
    st.write(
        "Enter information available at admission. The model returns a "
        "predicted outcome and class probabilities."
    )

    with st.form("student_prediction_form"):
        application_mode = st.number_input(
            "Application mode", min_value=1, max_value=100,
            value=1, step=1
        )
        application_order = st.number_input(
            "Application order", min_value=1, max_value=20,
            value=1, step=1
        )
        course = st.number_input(
            "Course code", min_value=1, max_value=9999,
            value=9238, step=1
        )
        attendance = st.selectbox(
            "Attendance schedule",
            options=[1, 0],
            format_func=lambda value: (
                "Daytime" if value == 1 else "Evening"
            ),
        )
        previous_qualification = st.number_input(
            "Previous qualification code", min_value=1,
            max_value=100, value=1, step=1
        )
        previous_grade = st.number_input(
            "Previous qualification grade",
            min_value=0.0, max_value=200.0, value=120.0
        )
        admission_grade = st.number_input(
            "Admission grade",
            min_value=0.0, max_value=200.0, value=120.0
        )
        age = st.number_input(
            "Age at enrollment", min_value=16, max_value=100,
            value=20, step=1
        )
        nationality = st.number_input(
            "Nationality code", min_value=1, max_value=200,
            value=1, step=1
        )
        international = st.selectbox(
            "International student",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No"
        )
        gender = st.selectbox(
            "Gender code",
            options=[0, 1],
            format_func=lambda value: str(value)
        )

        submitted = st.form_submit_button("Predict student outcome")

    if submitted:
        student = pd.DataFrame([{
            "Application mode": application_mode,
            "Application order": application_order,
            "Course": course,
            "Daytime/evening attendance": attendance,
            "Previous qualification": previous_qualification,
            "Previous qualification (grade)": previous_grade,
            "Admission grade": admission_grade,
            "Age at enrollment": age,
            "Nacionality": nationality,
            "International": international,
            "Gender": gender,
        }])

        try:
            prediction = model.predict(student)[0]
            st.success(f"Predicted outcome: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(student)[0]
                probability_table = pd.DataFrame({
                    "Outcome": model.classes_,
                    "Predicted probability": probabilities,
                }).sort_values(
                    "Predicted probability",
                    ascending=False,
                )

                st.markdown("### Model-estimated class probabilities")
                st.dataframe(
                    probability_table.style.format(
                        {"Predicted probability": "{:.1%}"}
                    ),
                    width="stretch",
                )
                st.caption(
                    "These are model-estimated probabilities, not guaranteed "
                    "real-world chances. They may not be well calibrated."
                )

                        # Explain the prediction using Logistic Regression coefficients.
            st.markdown("### Why did the model make this prediction?")
            if hasattr(model, "named_steps"):
                classifier = model.named_steps.get("classifier")
                preprocessor = model.named_steps.get("preprocessor")

                if (
                    classifier is not None
                    and hasattr(classifier, "coef_")
                    and preprocessor is not None
                    and hasattr(preprocessor, "get_feature_names_out")
                ):
                    feature_names = preprocessor.get_feature_names_out()
                    class_names = list(classifier.classes_)
                    predicted_class_index = class_names.index(prediction)

                    coefficients = pd.Series(
                        classifier.coef_[predicted_class_index],
                        index=feature_names,
                        name="Coefficient",
                    )

                    positive = coefficients.nlargest(10)
                    negative = coefficients.nsmallest(10)

                    left, right = st.columns(2)

                    with left:
                        st.markdown("#### Strongest positive coefficients")
                        st.dataframe(
                            positive.rename_axis(
                                "Encoded feature"
                            ).reset_index(),
                            width="stretch",
                        )

                    with right:
                        st.markdown("#### Strongest negative coefficients")
                        st.dataframe(
                            negative.rename_axis(
                                "Encoded feature"
                            ).reset_index(),
                            width="stretch",
                        )

                    st.caption(
                        "Coefficients indicate how encoded features influence "
                        "the predicted class score. They do not prove causation."
                    )
                else:
                    st.info(
                        "Coefficient explanations are unavailable for "
                        "the currently loaded model."
                    )
            else:
                st.info(
                    "This model does not expose a pipeline-based "
                    "coefficient explanation."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


with about_tab:
    st.subheader("About this project")
    st.write(
        "The project uses the UCI Predict Students' Dropout and Academic "
        "Success dataset, containing 4,424 records and 36 original features."
    )

    st.markdown(
        """
        **Current scope**
        - Admission-stage Logistic Regression baseline
        - Training-only preprocessing within a scikit-learn pipeline
        - Held-out evaluation with macro-averaged metrics
        - Confusion matrix and class-wise report
        - Interactive single-student prediction
        - Coefficient-based explanation of predictions

        **Limitations**
        - Predictions can be wrong and must not determine a student's future.
        - Input codes must match the dataset's category definitions.
        - Model-estimated probabilities may not be calibrated.
        - This is a demonstration, not a validated student intervention system.
        """
    )

    st.caption(
        "Dataset: Predict Students' Dropout and Academic Success, UCI "
        "Machine Learning Repository, DOI: 10.24432/C5MC89."
    )
