from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

print("\n--- Stage 2: Feature Encoding & Split ---")

# Label Encoding for binary categorical variables
label_encoder_sex = LabelEncoder()
X['sex'] = label_encoder_sex.fit_transform(X['sex'])

# One-Hot Encoding for categorical variables with multiple categories
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Drop rows with missing values from both X and y (to align, as in the original code)
X, y = X.align(y, axis=0, join='inner')

# Check the new shapes
print("\nDataset shape after encoding:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

# Display sample
print("\nCleaned Data Sample:")
print(X.head())

# Initialize the label encoder for the target variable (income)
label_encoder_target = LabelEncoder()
y = label_encoder_target.fit_transform(y)

# Check the unique values in the target after encoding
print("\nEncoded Target Values:")
print(set(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting splits
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
