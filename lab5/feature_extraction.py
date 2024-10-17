import numpy as np
from scipy.stats import entropy
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


class WrapperFeatureSelector:
    def __init__(self):
        self.classifier = SGDClassifier()
        self.features = None

    def get_best_feature(self, X, y, selected_features):
        best_feature = None
        max_quality = 0
        for feature in self.features:
            X_subset = X[selected_features + [feature]]
            self.classifier.fit(X_subset, y)
            quality = f1_score(y, self.classifier.predict(X_subset))
            if quality > max_quality:
                max_quality = quality
                best_feature = feature
        return best_feature

    def select_features(self, X, y, n_features):
        selected_features = []
        self.features = set(X.columns.values)
        for i in range(n_features):
            best_feature = self.get_best_feature(X, y, selected_features)
            selected_features.append(best_feature)
            self.features.remove(best_feature)
        return selected_features


class EmbeddedFeatureSelector(BaseEstimator):
    def __init__(self, n_features=30, random_state=42):
        self.n_features = n_features
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.selected_features_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-self.n_features:]
        self.selected_features_ = X.columns[indices]
        return self


class FilterFeatureSelector(BaseEstimator):
    def __init__(self, n_features=1):
        self.n_features = n_features
        self.mutual_infos = None

    def get_n_features(self, columns):
        """
        Выборка N наиболее важных признаков
        """
        if self.mutual_infos is None:
            raise RuntimeError("Сначала нужно вызывать метод mutual_information")

        feature_importances = sorted(zip(columns, self.mutual_infos), key=lambda x: x[1], reverse=True)
        top_n_features = [feature for feature, _ in feature_importances[:self.n_features]]
        return top_n_features

    def mutual_information(self, X, y):
        """
        Вычисляет взаимную информацию между признаками X и целевой переменной y.
        """
        n_features = X.shape[1]
        self.mutual_infos = np.zeros(n_features)
        for i in range(n_features):
            feature_entropy = entropy(np.unique(X[:, i], return_counts=True)[1])
            feature_conditional_entropy = np.mean(
                [entropy(np.unique(X[y == cls, i], return_counts=True)[1]) for cls in np.unique(y)])
            self.mutual_infos[i] = feature_entropy - feature_conditional_entropy

        return self.mutual_infos
