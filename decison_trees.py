# %%
import numpy as np
# %%
class Node:
    """A single node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # only set for leaf nodes

    def is_leaf(self):
        return self.value is not None

# %%
class DecisionTree:
    """
    Decision Tree with explicit Node structure.
    Supports classification (Gini) and regression (MSE).
    """

    def __init__(self, max_depth=5, min_samples_split=2, task="classification"):
        assert task in ("classification", "regression")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1):
            return Node(value=self._leaf_value(y))

        # Find best split
        best_feature, best_thresh = self._best_split(X, y, n_features)
        if best_feature is None:
            return Node(value=self._leaf_value(y))

        # Split data
        left_idx = X[:, best_feature] <= best_thresh
        right_idx = ~left_idx
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=best_feature, threshold=best_thresh,
                    left=left_child, right=right_child)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                gain = self._information_gain(y, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_thresh = feature, t

        return split_idx, split_thresh

    def _information_gain(self, y, left_idx, right_idx):
        if self.task == "classification":
            return self._gini_gain(y, left_idx, right_idx)
        else:
            return self._mse_gain(y, left_idx, right_idx)

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _gini_gain(self, y, left_idx, right_idx):
        n = len(y)
        gini_parent = self._gini(y)
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        weighted = (len(y[left_idx]) / n) * gini_left + (len(y[right_idx]) / n) * gini_right
        return gini_parent - weighted

    def _mse(self, y):
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _mse_gain(self, y, left_idx, right_idx):
        n = len(y)
        mse_parent = self._mse(y)
        mse_left = self._mse(y[left_idx])
        mse_right = self._mse(y[right_idx])
        weighted = (len(y[left_idx]) / n) * mse_left + (len(y[right_idx]) / n) * mse_right
        return mse_parent - weighted

    def _leaf_value(self, y):
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return np.mean(y)

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return self._predict_one(X, self.root)
        return np.array([self._predict_one(x, self.root) for x in X])


# -------------------------
# Small usage example
# -------------------------
# %%    # Classification Example
X_train = np.array([[2, 3],
                    [1, 1],
                    [2, 1],
                    [6, 5],
                    [7, 7],
                    [8, 6]])
y_train = np.array(["A", "A", "A", "B", "B", "B"])

clf = DecisionTree(max_depth=3, task="classification")
clf.fit(X_train, y_train)

X_test = np.array([[2, 2], [7, 6]])
print("Classification preds:", clf.predict(X_test))

# Regression Example
X_reg = np.array([[1], [2], [3], [10]])
y_reg = np.array([1.2, 2.1, 2.9, 10.5])
reg_tree = DecisionTree(max_depth=2, task="regression")
reg_tree.fit(X_reg, y_reg)
print("Regression preds:", reg_tree.predict([[2], [9]]))



# %%

