# %%
import numpy as np
# %%
class SVM:
    """
    Linear Support Vector Machine (SVM) using gradient descent.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization strength
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train SVM using hinge loss.
        y should be -1 or +1 labels.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization term
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    # Misclassified -> include hinge loss gradient
                    dw = self.lambda_param * self.w - y[idx] * x_i
                    db = -y[idx]

                # Update parameters
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

    def predict(self, X):
        """Return class predictions (-1 or +1)."""
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)


# -------------------------
# Demo
# --------------------------
# %%
# Toy dataset (linearly separable)
X_train = np.array([[2, 3],
                    [1, 1],
                    [2, 0.5],
                    [6, 8],
                    [7, 9],
                    [8, 8]])
y_train = np.array([-1, -1, -1, 1, 1, 1])

svm = SVM(learning_rate=0.001, lambda_param=0.01, n_epochs=1000)
svm.fit(X_train, y_train)

X_test = np.array([[2, 2], [7, 7]])
preds = svm.predict(X_test)
print("SVM predictions:", preds)


# %%

