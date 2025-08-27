# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# %%
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []   # store loss values
        self.weight_history = [] # store weight values

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update params
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Loss (MSE)
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)
            # Store a copy of weights for tracking
            self.weight_history.append(self.weights.copy())

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



# %%
X, y = make_regression(
    n_samples=1000,   # number of samples
    n_features=5,     # number of features (multi-feature dataset)
    noise=15,         # add noise to make it realistic
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression(learning_rate=0.01, n_epochs=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R^2:", r2_score(y_test, y_pred))

# %%
plt.scatter(y_test, y_pred, alpha=0.2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted (Dummy Multi-Feature Data)")
plt.savefig("img.png")

# %%
plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Loss Decreasing on Dummy Multi-Feature Data")
plt.savefig("letter.png")

# %%

