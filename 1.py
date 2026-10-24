import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Project started successfully!")
# Student performance dataset

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 3, 5, 7, 8, 4, 6, 9],
    "Attendance": [65, 70, 72, 75, 80, 85, 90, 95, 68, 78, 88, 92, 74, 82, 96],
    "Previous_Score": [50, 55, 60, 65, 70, 75, 80, 85, 52, 68, 77, 83, 62, 72, 90],
    "Assignments": [5, 6, 6, 7, 7, 8, 9, 10, 5, 7, 8, 9, 6, 8, 10],
    "Sleep_Hours": [5, 6, 6, 7, 7, 8, 8, 8, 5, 7, 7, 8, 6, 7, 8],
    "Final_Score": [52, 58, 62, 68, 72, 78, 84, 91, 55, 70, 80, 87, 64, 75, 94]
}

# Convert dictionary into a DataFrame
df = pd.DataFrame(data)

print("\nStudent Dataset:")
print(df)
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Students:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

print("\nAverage Student Performance:")
print(df.mean())

print("\nCorrelation with Final Score:")
print(df.corr()["Final_Score"].sort_values(ascending=False))
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.show()
print("\nCorrelation with Final Score:")
print(df.corr()["Final_Score"].sort_values(ascending=False))
# Correlation Heatmap

plt.figure(figsize=(10, 7))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()