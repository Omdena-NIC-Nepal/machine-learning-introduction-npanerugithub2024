#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
import joblib  # Used to save and load Python objects (like models)
loaded_model = joblib.load('../data/linear_regression_model.pkl')  # Load the saved model from 

X_train = pd.read_pickle("../data/X_train.pkl")
X_test = pd.read_pickle("../data/X_test.pkl")
y_train = pd.read_pickle("../data/y_train.pkl")
y_test = pd.read_pickle("../data/y_test.pkl")

# === Make predictions on the test set ===
from sklearn.metrics import mean_squared_error, r2_score

lr_preds = loaded_model.predict(X_test)


# === Evaluate the model ===
lr_mse = mean_squared_error(y_test, lr_preds)  # Mean Squared Error
lr_r2 = r2_score(y_test, lr_preds)             # R² Score

print("Mean Squared Error (MSE):", lr_mse)
print("R-squared (R²):", lr_r2)


# === Plot Predicted vs Actual ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=lr_preds)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Diagonal line
plt.title("Predicted vs Actual Values")
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.grid(True)
plt.tight_layout()
plt.show()


# === Plot residuals ===
residuals = y_test - lr_preds  # Calculate residuals (actual - predicted)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=lr_preds, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()


# === Plot histogram of residuals ===
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)  # Histogram with Kernel Density Estimate
plt.title("Histogram of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


