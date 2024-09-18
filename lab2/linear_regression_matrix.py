import numpy as np
class RidgeRegressionMatrix:

    def __init__(self, lambda_: float):
        self.lambda_ = lambda_
        self.weights = []

    def fit(self, X_train, y_train):
        # Добавляем столбец из единиц к матрице признаков
        X_with_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

        # Вычисляем веса с помощью матричного уравнения
        self.weights = np.linalg.pinv(
            X_with_bias.T @ X_with_bias + self.lambda_ * np.eye(X_with_bias.shape[1])) @ X_with_bias.T @ y_train

    def predict_proba(self, X_test):
        # Делаем предсказания на обучающих данных
        X_test_bias = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
        return X_test_bias @ self.weights

    def predict(self, X_test):
        # Делаем предсказания на обучающих данных
        y_pred = self.predict_proba(X_test)
        y_pred = np.where(y_pred > 0.6, 1, -1)
        return y_pred