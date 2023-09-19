import numpy as np

from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression

# Load the dataset 'P1_2.txt'
linear_data = np.loadtxt('Reg_1.txt', delimiter='\t')

# Split the dataset into features (X) and labels (y)
X_linear = linear_data[:, :-1]
y_linear = linear_data[:, -1]

# Create a Linear Regression model
linear_model = LinearRegression(some=100000)  # You can adjust the 'some' parameter

# Fit the model on the data
linear_model.fit(X_linear, y_linear)

# Make predictions on the data
predictions = linear_model.predict(X_linear)

# Print or use the predictions as needed
print("Predictions:", predictions)

# Evaluation for Linear Regression
linear_predictions = linear_model.predict(X_linear)
linear_residuals = y_linear - linear_predictions
linear_mse = np.mean(linear_residuals ** 2)
linear_rmse = np.sqrt(linear_mse)
ss_total = np.sum((y_linear - np.mean(y_linear)) ** 2)
ss_residual = np.sum(linear_residuals ** 2)
linear_r2 = 1 - (ss_residual / ss_total)
print("Linear Regression Results:")
print(f"R2 Score: {linear_r2:.4f}")
print(f"Mean Squared Error (MSE): {linear_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {linear_rmse:.4f}")



# Load the dataset 'Reg_1.txt'
logistic_data = np.loadtxt('P1_2.txt', delimiter='\t')  # Assuming tab-separated values

# Split the dataset into features (X) and labels (y)
X = logistic_data[:, :-1]
y = logistic_data[:, -1] - 1

# Create a Logistic Regression model
model = LogisticRegression(some=5_000)  # You can adjust the 'some' parameter

# Fit the model on the data
model.fit(X, y)

# Make predictions on the data
predictions = model.predict(X)

# Print or use the predictions as needed
print("Predictions:", predictions)


# Evaluation for Logistic Regression
logistic_predictions = model.predict(X)
true_positives = np.sum((logistic_predictions == 1) & (y == 1))
false_positives = np.sum((logistic_predictions == 1) & (y == 0))
true_negatives = np.sum((logistic_predictions == 0) & (y == 0))
false_negatives = np.sum((logistic_predictions == 0) & (y == 1))

accuracy = (true_positives + true_negatives) / len(y)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")