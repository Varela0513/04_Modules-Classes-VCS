import numpy as np


class LinearRegression:
    def __init__(self, some, learning_rate=0.01, num_iterations=1000):
        self.some = some
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.some):  # Replace 'self.some' with the desired number of iterations
            linear_model = np.dot(X, self.weights) + self.bias
            gradient = self.calculate_gradient(X, y, linear_model, num_samples)
            self.weights -= self.learning_rate * gradient[0]
            self.bias -= self.learning_rate * gradient[1]

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def calculate_gradient(X, y, linear_model, num_samples):
        gradient_weights = (1 / num_samples) * np.dot(X.T, (linear_model - y))
        gradient_bias = (1 / num_samples) * np.sum(linear_model - y)
        return gradient_weights, gradient_bias
