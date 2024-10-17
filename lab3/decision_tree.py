import numpy as np
import pandas as pd
from statistics import mode
from sklearn.base import BaseEstimator


def entropy(y):
    p = [len(y[y == k]) / len(y) for k in np.unique(y)]
    return -np.dot(p, np.log2(p))


def gini(y):
    p = [len(y[y == k]) / len(y) for k in np.unique(y)]
    return 1 - np.dot(p, p)


def variance(y):
    return np.var(y)


def mad_median(y):
    return np.mean(np.abs(y - np.median(y)))


def regression_leaf(y):
    return np.mean(y)


def classification_leaf(y):
    return mode(y)


class Node:
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


class DecisionTree(BaseEstimator):
    def __init__(self, max_depth=100, min_samples_split=2, min_samples_leaf=1, criterion="entropy",
                 leaf_func="classification_leaf"):
        params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "leaf_func": leaf_func
        }

        criteria_dict = {
            "variance": variance,
            "mad_median": mad_median,
            "gini": gini,
            "entropy": entropy
        }

        leaf_dict = {
            "regression_leaf": regression_leaf,
            "classification_leaf": classification_leaf
        }

        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

        super(DecisionTree, self).set_params(**params)
        self._criterion_function = criteria_dict[criterion]
        self._leaf_value = leaf_dict[leaf_func]
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.current_depth = 0

    def _functional(self, x_train: pd.DataFrame, y: pd.Series, feature_idx: int, threshold):
        mask = x_train.iloc[:, feature_idx] < threshold
        n_obj = x_train.shape[0]
        n_left = np.sum(mask)
        n_right = n_obj - n_left
        if n_left > 0 and n_right > 0:
            return (
                    self._criterion_function(y)
                    - (n_left / n_obj) * self._criterion_function(y.loc[mask])
                    - (n_right / n_obj) * self._criterion_function(y.loc[~mask])
            )
        else:
            return 0

    def _build_tree(self, x_train: pd.DataFrame, y: pd.Series, depth=1):
        """Train decision tree"""
        max_functional = 0
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = x_train.shape

        if len(np.unique(y)) == 1:
            return Node(labels=y)

        best_mask = None
        if depth < self.max_depth and n_samples >= self.min_samples_split and n_samples >= self.min_samples_leaf:
            for feature_idx in range(n_features):
                max_value = np.max(x_train.iloc[:, feature_idx])
                min_value = np.min(x_train.iloc[:, feature_idx])
                threshold_values = np.linspace(min_value, max_value, 5)
                functional_values = [
                    self._functional(x_train, y, feature_idx, threshold) for threshold in threshold_values
                ]

                best_threshold_idx = np.nanargmax(functional_values)

                if functional_values[best_threshold_idx] > max_functional:
                    max_functional = functional_values[best_threshold_idx]
                    best_threshold = threshold_values[best_threshold_idx]
                    best_feature_idx = feature_idx
                    best_mask = x_train.iloc[:, feature_idx] < best_threshold

        if best_feature_idx is not None and best_mask is not None:
            return Node(
                feature_idx=best_feature_idx,
                threshold=best_threshold,
                left=self._build_tree(x_train.loc[best_mask], y.loc[best_mask], depth + 1),
                right=self._build_tree(x_train.loc[~best_mask, :], y.loc[~best_mask], depth + 1),
            )
        else:
            self.current_depth = depth
            return Node(labels=y)

    def fit(self, x_train: pd.DataFrame, y: pd.Series):
        """Run training decision tree"""
        self.root = self._build_tree(x_train, y)
        self.max_depth = self.current_depth
        return self

    def _predict_object(self, x: pd.Series):
        """Prediction for one test object"""
        node = self.root
        while node.labels is None:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return self._leaf_value(node.labels)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Prediction for all test objects"""
        results = np.array([self._predict_object(x_test.iloc[i]) for i in range(0, x_test.shape[0])])
        return np.array(results)
