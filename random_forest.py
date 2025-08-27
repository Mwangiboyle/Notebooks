# %%

import numpy as np
from collections import Counter

# %%
# --- Tree Node ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # feature index used for split
        self.threshold = threshold  # threshold value
        self.left = left            # left child (Node)
        self.right = right          # right child (Node)
        self.value = value          # if leaf, store predicted class

# %%
# --- Decision Tree (CART style) ---
class Tree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(set(y))

        # stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # choose random subset of features
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # split data
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feat], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh
        return split_idx, split_thresh

    def _information_gain(self, y, feature_col, threshold):
        # entropy
        def entropy(labels):
            counts = np.bincount(labels)
            probs = counts / len(labels)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])

        parent_entropy = entropy(y)

        # split
        left_idx = feature_col <= threshold
        right_idx = feature_col > threshold
        if sum(left_idx) == 0 or sum(right_idx) == 0:
            return 0

        # weighted avg entropy
        n = len(y)
        n_left, n_right = sum(left_idx), sum(right_idx)
        e_left, e_right = entropy(y[left_idx]), entropy(y[right_idx])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# %%
# --- Random Forest ---
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = Tree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.max_features
            )
            # bootstrap sample
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        # collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        final_preds = []
        for preds in tree_preds:
            final_preds.append(Counter(preds).most_common(1)[0][0])
        return np.array(final_preds)


# --- Example Usage ---
if __name__ == "__main__":
    # Simple dataset
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 1],
                  [6, 5],
                  [7, 7],
                  [8, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForest(n_estimators=5, max_depth=3, max_features=1)
    model.fit(X, y)
    preds = model.predict(np.array([[2, 2], [7, 6]]))
    print("Predictions:", preds)

