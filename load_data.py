from ucimlrepo import fetch_ucirepo

# Load the UCI Student Dropout dataset
stusdent_data = fetch_ucirepo(id=697)

# Separate features and target
X = student_data.data.features
y = student_data.data.targets

print("Dataset loaded successfully!")

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nColumn names:")
print(X.columns.tolist())

print("\nData types:")
print(X.dtypes)

print("\nMissing values:")
print(X.isnull().sum())

print("\nTarget distribution:")
print(y["Target"].value_counts())

print("\nBasic statistics:")
print(X.describe())