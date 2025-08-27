# %%
import numpy as np
# %%
def sigmoid(z):
    # Sigmoid function to map values to (0, 1)
    return 1 / (1 + np.exp(-z))
# %%
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # (Optional) Calculate loss for debugging
            # loss = -np.mean(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9))

    def predict_probability(self, X):
        # Predict probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        # Predict class labels (0 or 1) based on threshold
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)


# ------------------------------
# Testing with dummy data
# ------------------------------
# %%
# Dummy dataset (simple binary classification)
# Feature 1 = study hours, Feature 2 = hours of sleep
X = np.array([
    [2, 9],   # low study, high sleep
    [1, 5],   # low study, medium sleep
    [3, 6],   # moderate
    [5, 2],   # high study, low sleep
    [7, 1],   # very high study, low sleep
    [8, 2],   # very high study, low sleep
])
# Labels (0 = fail, 1 = pass)
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
model.fit(X, y)

# Test predictions
test_samples = np.array([[4, 5], [6, 1]])  # unseen samples
predictions = model.predict(test_samples)
probabilities = model.predict_probability(test_samples)

print("Predicted class labels:", predictions)
print("Predicted probabilities:", probabilities)
print("Learned weights:", model.weights)
print("Learned bias:", model.bias)


# %%

