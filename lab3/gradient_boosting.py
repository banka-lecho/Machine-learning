import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm
from sklearn.base import BaseEstimator

from lab3.decision_tree import DecisionTree


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


class GradientBoosting(BaseEstimator):
    def __init__(self, n_estimators=10, learning_rate=0.01, max_depth=3, min_samples_split=5, criterion="entropy",
                 leaf_func="classification_leaf", random_state=17, loss_name="mse"):

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

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.leaf_func = leaf_func
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.loss_name = loss_name
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0], 1])

        if loss_name == "mse":
            self.objective = self.mse
            self.objective_grad = self.mse_grad

        elif loss_name == "rmsle":
            self.objective = self.rmsle
            self.objective_grad = self.rmsle_grad

        self.trees_ = []

    @staticmethod
    def mse(y, p):
        return np.mean((y - p) ** 2)

    @staticmethod
    def mse_grad(y: np.array, p: np.array):
        return 2 * (p - y) / y.shape[0]

    @staticmethod
    def rmsle(y, p):
        y = y.reshape([y.shape[0], 1])
        p = p.reshape([p.shape[0], 1])
        return np.mean(np.log((p + 1) / (y + 1)) ** 2) ** 0.5

    def rmsle_grad(self, y, p):
        y = y.reshape([y.shape[0], 1])
        p = p.reshape([p.shape[0], 1])
        return 1.0 / (y.shape[0] * (p + 1) * self.rmsle(y, p)) * np.log((p + 1) / (y + 1))

    def fit(self, X: np.array, y: np.array):
        b = self.initialization(y)
        prediction = b.copy()

        for t in tqdm(range(self.n_estimators)):
            if t == 0:
                resid = y
            else:
                resid = -self.objective_grad(y, prediction)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                leaf_func=self.leaf_func
            )
            tree.fit(X, pd.Series(resid))
            b = tree.predict(X).reshape([X.shape[0], 1])
            self.trees_.append(tree)
            prediction += self.learning_rate * b
        return self

    def predict(self, X):
        predictions = np.ones([X.shape[0], 1])
        for t in range(self.n_estimators):
            predictions += self.learning_rate * self.trees_[t].predict(X).reshape([X.shape[0], 1])
        return predictions
