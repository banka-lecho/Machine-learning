import warnings
import numpy as np
from decimal import Decimal
from sklearn.base import BaseEstimator
warnings.filterwarnings("ignore")


"""Ядра"""


def uniform(r: float) -> float:
    """ Uniform kernel """
    return 1 / 2


def triangular(r: float) -> float:
    """ Triangular kernel """
    return 1 - abs(r)


def quartic(r: float) -> float:
    """ Quartic kernel """
    return 15 / 16 * (1 - r ** 2) ** 2


def gaussian(r: float) -> float:
    """ Gaussian kernel """
    return np.exp(-0.5 * r ** 2) / np.sqrt(2 * np.pi)


"""Метрики"""


def euclidean(x: np.array, x_neighbor: np.array) -> float:
    """ Euclidean distance between two embeddings """
    return float(np.sqrt(np.sum(np.square(x - x_neighbor))))


def cosine(x: np.array, x_neighbor: np.array) -> float:
    """ Cosine similarity between two embeddings """
    result = float(1 - np.dot(x, x_neighbor.T) / (np.linalg.norm(x) * np.linalg.norm(x_neighbor)))
    return result


def minkowski_distance(x: np.array, x_neighbor: np.array) -> float:
    """ Minkowski distance between two embeddings """

    def p_root(sum_diff, root):
        root_value = 1 / float(root)
        return (sum_diff ** root_value).astype(float)
    p_value = 1
    sum_diff = sum(pow(abs(a - b), p_value) for a, b in zip(x, x_neighbor))
    return float(p_root(sum_diff, p_value)[0])


class K_Nearest_Neighbors_Classifier(BaseEstimator):
    def __init__(self, metric=None, kernel=None, window_type=None, window_width=None, k=None, p_value=None):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.m_test = None
        self.p_value = p_value
        self.metric = metric if metric is not None else euclidean
        self.kernel = kernel if kernel is not None else quartic
        self.window_type = window_type if window_type is not None else 'variable'
        self.window_width = window_width

    def fit(self, x_train, y_train):
        """ Function to store training set """
        self.x_train = x_train
        self.y_train = y_train

    def window_function(self, distances: list):
        """ Function for setting weights for objects with kernel """
        weights = []
        if self.window_type == 'fixed':
            for (dist, target) in distances:
                if dist < self.window_width:
                    weights.append((self.kernel(dist/self.window_width), target))
                else:
                    weights.append((0, target))

        elif self.window_type == 'variable':
            count = 0
            for (dist, target) in distances:
                if count < self.k:
                    weights.append((self.kernel(dist), target))
                else:
                    weights.append((0, target))
                count += 1

        else:
            raise ValueError("Window type should be 'fixed' or 'variable'")
        return weights

    def predict_for_one(self, emb):
        """ Predict class for one object"""
        distances = []
        for i, v in enumerate(self.x_train):
            dist = self.metric(emb, v)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        # вычисляем weights без ядра
        weights_targets = self.window_function(distances)
        # вычисляем weights с ядром
        weights = [weight for (weight, _) in weights_targets]
        targets = [target for (_, target) in weights_targets]

        prediction = np.argmax(np.bincount(targets, weights))
        return prediction

    def predict(self, x_test):
        predictions = []
        for emb in x_test:
            prediction = self.predict_for_one(emb)
            predictions.append(prediction)
        return predictions