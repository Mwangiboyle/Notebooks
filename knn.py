# %%
import numpy as np
from collections import Counter, defaultdict

# %%
class KNN:
    """
    Simple K-Nearest Neighbors from scratch.

    Parameters
    ----------
    k : int
        Number of neighbors.
    task : str
        "classification" or "regression".
    metric : str
        "euclidean" or "manhattan".
    weights : str
        "uniform" or "distance" (distance-weighted voting).
    """
    def __init__(self, k=3, task="classification", metric="euclidean", weights="uniform"):
        assert task in ("classification", "regression")
        assert metric in ("euclidean", "manhattan")
        assert weights in ("uniform", "distance")
        self.k = int(k)
        self.task = task
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self._is_classification = self.task == "classification"

    def fit(self, X, y):
        """Store training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        self.X_train = X
        self.y_train = y
        # For classification, save sorted unique labels for stable tie-breaking
        if self._is_classification:
            self._labels = np.unique(y)

    def _distance(self, A, B):
        """Compute pairwise distances between A (m x d) and B (n x d)."""
        if self.metric == "euclidean":
            # Efficient broadcasting: sqrt(sum((a-b)^2, axis=1))
            dists = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
        else:  # manhattan
            dists = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
        return dists  # shape (m, n)

    def predict(self, X):
        """Predict labels (classification) or values (regression) for X."""
        if self.X_train is None:
            raise ValueError("Model has not been fitted. Call fit(X, y) first.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dists = self._distance(X, self.X_train)  # (m, n_train)
        n_queries = dists.shape[0]
        preds = []

        for i in range(n_queries):
            dist_row = dists[i]
            # indices of k nearest neighbors (stable tie-breaking by argsort)
            nn_idx = np.argsort(dist_row)[: self.k]
            nn_dists = dist_row[nn_idx]
            nn_labels = self.y_train[nn_idx]

            if self._is_classification:
                if self.weights == "uniform":
                    vote_counts = Counter(nn_labels)
                    # tie-breaking: choose label with highest count then smallest label (stable)
                    top_count = max(vote_counts.values())
                    candidates = [lab for lab, cnt in vote_counts.items() if cnt == top_count]
                    # if numeric labels, pick smallest; if strings, sort lexicographically
                    chosen = sorted(candidates)[0]
                    preds.append(chosen)
                else:  # distance-weighted voting
                    votes = defaultdict(float)
                    for lab, dist in zip(nn_labels, nn_dists):
                        w = 1.0 / (dist + 1e-9)  # add small epsilon to avoid div-by-zero
                        votes[lab] += w
                    # pick label with highest summed weight; tie-break deterministic
                    max_w = max(votes.values())
                    candidates = [lab for lab, w in votes.items() if w == max_w]
                    preds.append(sorted(candidates)[0])
            else:
                # regression
                if self.weights == "uniform":
                    preds.append(float(np.mean(nn_labels)))
                else:
                    # weighted mean
                    weights = 1.0 / (nn_dists + 1e-9)
                    preds.append(float(np.sum(weights * nn_labels) / np.sum(weights)))

        return np.array(preds) if preds and len(preds) > 1 else np.array(preds).reshape(-1)

    def predict_proba(self, X):
        """
        For classification only: return probability distribution over seen labels
        for each sample. Returns array shape (n_samples, n_labels).
        """
        if not self._is_classification:
            raise ValueError("predict_proba is only available for classification.")
        if self.X_train is None:
            raise ValueError("Model has not been fitted. Call fit(X, y) first.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dists = self._distance(X, self.X_train)
        probs = []
        for i in range(dists.shape[0]):
            dist_row = dists[i]
            nn_idx = np.argsort(dist_row)[: self.k]
            nn_dists = dist_row[nn_idx]
            nn_labels = self.y_train[nn_idx]

            label_scores = defaultdict(float)
            if self.weights == "uniform":
                for lab in nn_labels:
                    label_scores[lab] += 1.0
            else:
                for lab, dist in zip(nn_labels, nn_dists):
                    label_scores[lab] += 1.0 / (dist + 1e-9)

            # convert to probability over sorted labels
            total = sum(label_scores.values()) + 1e-12
            row = [label_scores.get(l, 0.0) / total for l in self._labels]
            probs.append(row)
        return np.array(probs)

# -------------------------
# Small usage example
# -------------------------
# %%
# Simple toy classification
X_train = np.array([[1, 1], [1, 2], [2, 2], [6, 5], [7, 7], [8, 6], [10,11], [11,13],
                    [14,16]])
y_train = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])

model = KNN(k=3, task="classification", metric="euclidean", weights="distance")
model.fit(X_train, y_train)
# %%
X_test = np.array([[1.5, 1.5], [7, 6], [13,15]])
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
print("Classification preds:", preds)
print("Probability distributions (rows -> samples):", probs)
# %%    # Simple regression
X_train_reg = np.array([[0], [1], [2], [3], [10]])
y_train_reg = np.array([0.0, 0.9, 1.9, 3.1, 10.5])
model_reg = KNN(k=2, task="regression", metric="euclidean", weights="distance")
model_reg.fit(X_train_reg, y_train_reg)

print("Regression preds:", model_reg.predict(np.array([[1.5], [8]])))


# %%

