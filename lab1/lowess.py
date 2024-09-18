import numpy as np
import pandas as pd

def gaussian(r):
    """Gaussian kernel"""
    return np.exp(-0.5 * r ** 2) / np.sqrt(2 * np.pi)


def cosine(x, x_neighbor) -> float:
    """ Cosine similarity """
    result = float(1 - np.dot(x, x_neighbor.T) / (np.linalg.norm(x) * np.linalg.norm(x_neighbor)))
    return result


class Lowess(object):
    def __init__(self, metric=None, kernel=None, window_type=None, window_width=None, k=None, p_value=None):
        self.x_train = None
        self.y_train = None
        self.metric = metric if metric is not None else cosine
        self.kernel = kernel if kernel is not None else gaussian
        self.window_type = window_type if window_type is not None else 'variable'
        self.window_width = window_width
        self.k = k
        self.p_value = p_value

    def fit(self, x_train, y_train):
        """ Function to store training set """
        self.x_train = x_train
        self.y_train = y_train

    def window_function(self, distances: list):
        weights = []
        if self.window_type == 'fixed':
            for (dist, target) in distances:
                if dist < self.window_width:
                    weights.append((self.kernel(dist), target))
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

    def get_one_prediction(self, emb, delta):
        distances = []
        for i, v in enumerate(self.x_train):
            dist = self.metric(emb, v)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        weights_targets = self.window_function(distances)

        y_w = [y_i * w_i for (y_i, w_i) in weights_targets]
        y_w_delta = [y_w_i * delta_i for (y_w_i, delta_i) in zip(y_w, delta)]
        prediction = sum(y_w) / sum(y_w_delta)
        return prediction

    def get_all_predictions(self, iter):
        predictions = np.zeros(len(self.x_train))
        delta = np.ones(len(self.x_train))
        for iteration in range(iter):
            for i in range(0, len(self.x_train)):
                prediction = self.get_one_prediction(self.x_train[i], delta)
                predictions[i] = prediction

            residuals = self.y_train - predictions
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2

        return predictions


def run_my_lowess(x, y):
    model_lowess = Lowess(metric=cosine,
                          kernel=gaussian,
                          window_type='variable',
                          window_width=10,
                          k=5)
    model_lowess.fit(x, y)
    predictions = model_lowess.get_all_predictions(3)
    return predictions