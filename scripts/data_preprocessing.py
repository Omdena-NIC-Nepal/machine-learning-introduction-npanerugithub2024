
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load housing dataset
df = pd.read_pickle("../data/eda_df.pkl")

# Check for missing values and drop if any
df = df.dropna()

# Handle outliers using Z-score (only for numeric columns)
z_scores = np.abs(zscore(df.select_dtypes(include='number')))
df = df[(z_scores < 3).all(axis=1)]

# Display data types
print(df.dtypes)

# Encode categorical 'chas' column if it's of object type
if df["chas"].dtype == 'object':
    df = pd.get_dummies(df, columns=["chas"], drop_first=True)

# Separate features and target
X = df.drop(columns=["medv"])  # Features
y = df["medv"]                 # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame for compatibility
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Save processed datasets
X_train.to_pickle("../data/X_train.pkl")
X_test.to_pickle("../data/X_test.pkl")
y_train.to_pickle("../data/y_train.pkl")
y_test.to_pickle("../data/y_test.pkl")
