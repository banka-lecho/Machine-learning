import warnings
from metrics import *
from kernels import *
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")


class K_Nearest_Neighbors_Classifier(BaseEstimator):
    def __init__(self, metric=None, kernel=None, window_type=None, window_width=None, k=None, p_value=None,
                 lowess_weights=None):

        dict_kernels = {'gaussian': gaussian, 'quartic': quartic, 'triangular': triangular}

        dict_metrics = {'euclidean': euclidean, 'minkowski_distance': minkowski_distance, 'cosine': cosine}

        self.k = k
        self.x_train = None
        self.y_train = None
        self.m_test = None
        self.p_value = p_value
        self.metric = dict_metrics[metric] if dict_metrics[metric] is not None else euclidean
        self.kernel = dict_kernels[kernel] if dict_kernels[kernel] is not None else quartic
        self.window_type = window_type if window_type is not None else 'variable'
        self.window_width = window_width
        self.lowess_weights = lowess_weights

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
                    weights.append((self.kernel(dist / self.window_width), target))
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

    def predict_for_one(self, u, index_of_u):
        """ Predict class for one object"""

        if self.lowess_weights is None:
            distances = []
            for i, v in enumerate(self.x_train):
                dist = self.metric(u, v)
                distances.append((dist, self.y_train[i]))

            distances.sort(key=lambda x: x[0])
            weights_targets = self.window_function(distances)
            weights = [weight for (weight, _) in weights_targets]
            targets = [target for (_, target) in weights_targets]
        else:
            targets, weights = self.lowess_weights[index_of_u]

        prediction = np.argmax(np.bincount(targets, weights))
        return prediction

    def predict(self, x_test):
        predictions = []
        for i, u in enumerate(x_test):
            prediction = self.predict_for_one(u, i)
            predictions.append(prediction)
        return predictions
