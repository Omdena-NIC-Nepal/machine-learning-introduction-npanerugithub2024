#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load saved dataset
X_train = pd.read_pickle("../data/X_train.pkl")
X_test = pd.read_pickle("../data/X_test.pkl")
y_train = pd.read_pickle("../data/y_train.pkl")
y_test = pd.read_pickle("../data/y_test.pkl")

#Train model
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression() # Create an instance of the LinearRegression model (nothing is trained yet)

lr_model.fit(X_train, y_train) # Train the model using the training data; this finds the best-fit line by minimizing the error


# To save your trained Linear Regression model in Python, the most common and reliable way is to use joblib from sklearn or directly from the joblib package.
import joblib  # Used to save and load Python objects (like models)
joblib.dump(lr_model, '../data/linear_regression_model.pkl')  # Save the trained model to a .pkl file