import pandas as pd
import os

print("--- Stage 1: Data Cleaning ---")

# Define columns
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

# Check relative paths first (from notebooks/ or workspace root)
data_paths = [
    os.path.join("..", "data", "adult.data"),
    os.path.join("data", "adult.data")
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("Local file not found, loading from URL...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data = pd.read_csv(url, names=columns, skipinitialspace=True)
else:
    print(f"Loading dataset from {data_path}...")
    data = pd.read_csv(data_path, names=columns, skipinitialspace=True)

# Separate the target variable
y = data['income']
X = data.drop(columns=['income'])

# Handling Missing Values
# Replace '?' with NaN
X.replace("?", pd.NA, inplace=True)

# Check missing values before handling
print("\nMissing values before handling:")
print(X.isnull().sum())

# Drop rows with missing values
X.dropna(inplace=True)

# Check missing values after handling
print("\nMissing values after handling:")
print(X.isnull().sum())

# Drop duplicate rows
X.drop_duplicates(inplace=True)

# Drop rows with missing values from both X and y (align features and target)
X, y = X.align(y, axis=0, join='inner')

print(f"\nTarget aligned. X shape: {X.shape}, y shape: {y.shape}")
