# Intelligent Student Learning Analytics

## Project Overview

Intelligent Student Learning Analytics is a machine learning project that analyzes student academic outcome data and estimates whether a student may graduate, remain enrolled, or drop out.

The project uses the UCI *Predict Students' Dropout and Academic Success* dataset.

## Dataset

- **Number of records:** 4,424
- **Original features:** 36
- **Target classes:** Dropout, Enrolled, Graduate
- **Source:** UCI Machine Learning Repository
- **DOI:** 10.24432/C5MC89

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Jupyter Notebook
- Visual Studio Code

## Project Features

- Exploratory data analysis
- Data preprocessing using a machine learning pipeline
- Admission-stage student outcome prediction
- Logistic Regression baseline model
- Classification report and confusion matrix
- Estimated probabilities for three outcomes
- Coefficient-based model explanations
- Interactive Streamlit dashboard

## Project Structure

```text
Intelligent_Student_Learning_Analytics/
├── app/
│   └── streamlit_app.py
├── data/
├── docs/
├── models/
├── notebooks/
├── results/
├── src/
├── tests/
├── visualizations/
├── load_data.py
└── README.md
```

## How to Run the Dashboard

### 1. Open the project folder

Open the project in VS Code.

### 2. Activate the virtual environment

In Windows PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies if needed

```powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 4. Run the application

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Model Evaluation

The admission-stage Logistic Regression model was evaluated using a held-out test set.

Previously recorded results for the balanced Logistic Regression model were approximately:

- **Accuracy:** 52.2%
- **Macro F1-score:** 48.4%

These results show that the model has limitations when distinguishing between all three student outcomes.

## Limitations

- Predictions can be incorrect and are not guarantees of future academic outcomes.
- Estimated probabilities may not be well calibrated.
- Input category codes must match the dataset definitions.
- Model coefficients do not establish causation.
- This project is a demonstration, not a validated student intervention system.
s
## Future Improvements

- Compare additional machine learning algorithms.
- Improve performance across all outcome classes.
- Evaluate probability calibration.
- Expand automated testing.
- Improve documentation and deployment.

## License

Add an appropriate license before distributing the project publicly.
