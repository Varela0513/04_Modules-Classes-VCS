import numpy as np

class LogisticRegression:
    def __init__(self, some, learning_rate=0.0001, num_iterations=1000):
        self.some = some
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.some):  # Replace 'self.some' with the desired number of iterations
            linear_model = np.dot(X, self.weights) + self.bias
            predicted = self.sigmoid(linear_model)
            gradient_weights, gradient_bias = self.calculate_gradients(X, predicted, y, num_samples)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predicted = self.sigmoid(linear_model)
        predicted_classes = [1 if x > 0.5 else 0 for x in predicted]
        return np.array(predicted_classes)

    @staticmethod
    def calculate_gradients(X, predicted, y, num_samples):
        gradient_weights = np.dot(X.T, (predicted - y)) / num_samples
        gradient_bias = np.sum(predicted - y) / num_samples
        return gradient_weights, gradient_bias
