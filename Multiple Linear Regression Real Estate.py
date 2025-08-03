# Real Estate Price Prediction using Multiple Linear Regression
# Author: Kushagra Goyal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set()

# Load dataset
data = pd.read_csv("real estate price size year.csv")

# Define features and target variable
X = data[['size', 'year']]
y = data['price']

# Add a constant term to the model
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display model summary
print(model.summary())
