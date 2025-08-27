# %%
import numpy as np

# %%
class NaiveBayes:
    """
    Gaussian Naive Bayes from scratch.
    Assumes features are continuous and normally distributed.
    """

    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """Fit model by computing mean, variance, and priors for each class."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9  # add small value to prevent /0
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_density(self, class_label, x):
        """Calculate Gaussian likelihood of the data point x given class parameters."""
        mean = self.mean[class_label]
        var = self.var[class_label]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _posterior(self, x):
        """Compute log-posterior probability for each class."""
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(np.log(self._gaussian_density(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """Predict class labels for samples in X."""
        X = np.asarray(X)
        if X.ndim == 1:
            return self._posterior(X)
        return np.array([self._posterior(x) for x in X])


# -------------------------
# Small usage example
# -------------------------
# %%
# Simple dataset
X_train = np.array([[1.0, 2.1],
                    [1.3, 1.8],
                    [2.0, 2.5],
                    [6.0, 7.1],
                    [6.5, 6.8],
                    [7.2, 7.9]])
y_train = np.array(["A", "A", "A", "B", "B", "B"])

model = NaiveBayes()
model.fit(X_train, y_train)

X_test = np.array([[1.5, 2.0], [6.8, 7.0]])
preds = model.predict(X_test)
print("Predictions:", preds)

