import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the housing dataset
data: DataFrame = pd.read_csv('housing.csv')

# Display the first few rows and info about the dataset
print(data.head())
print(data.info())

# Remove rows with missing values
data = data.dropna()

# Convert categorical variables to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']  # Target variable

# Split the data into training and testing sets (90% test, 10% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
lr_predictions = lr_model.predict(X_test)

# Evaluate the model's performance
print("Linear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, lr_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, lr_predictions)))

# Visualize actual vs. predicted house prices
plt.figure(figsize=(10,6))
plt.scatter(y_test, lr_predictions, label='Linear Regression', color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()