from sklearn.metrics import f1_score

from utils.eda import *
from sklearn.base import BaseEstimator


class LinearRegression(BaseEstimator):
    def __init__(self, learning_rate=0.1, max_epoches=1000, size_batch=60, eps=0.0000001):
        params = {
            "learning_rate": learning_rate,
            "max_epoches": max_epoches,
            "size_batch": size_batch,
            "eps": eps
        }

        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

        super(LinearRegression, self).set_params(**params)
        self.batches = []
        self.w = None
        self.train_losses = []
        self.test_losses = []
        self.learning_rate = learning_rate
        self.max_epoches = max_epoches
        self.size_batch = size_batch

    def _stable_sigmoid(self, z: np.ndarray):
        """Sigmoid: функция преобразования предсказания модели в диапазон [0,1]"""
        z = np.sum(z)
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (np.exp(z) + 1)

    def _log_loss(self, y_pred: np.ndarray, y: np.ndarray):
        """Логистическая функция потерь"""
        y_pred = self._stable_sigmoid(y_pred)
        y_one_loss = y * np.log(y_pred + 1e-9)
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def _deriative_log_loss(self, X: np.ndarray, y_pred: np.ndarray, y: np.ndarray):
        """Производная лог лосса"""
        return np.dot(X.T, (y_pred - y)) / y_pred.shape[0]

    def _mse(self, y_pred: np.ndarray, y: np.ndarray):
        """MSE: функция отклонения таргета от предсказаний"""
        return np.divide(np.sum((y_pred - y) ** 2), len(y_pred))

    def _deriative_mse(self, X: np.ndarray, y_pred: np.ndarray, y: np.ndarray):
        """Градиента MSE"""
        return np.dot(np.divide(2 * (y - y_pred), len(X)), (-X))

    def stable_softmax(X: np.ndarray, w: np.ndarray):
        """Функция SoftMax"""
        z = np.dot(-X, w)
        z = z - np.max(z, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        softmax = numerator / denominator
        return softmax

    def cross_entropy(y_pred: np.ndarray, y: np.ndarray, epsilon=1e-9):
        """Кросс-энтропийная функция потерь в многоклассовой классификации"""
        n = y_pred.shape[0]
        ce = -np.sum(y * np.log(y_pred + epsilon)) / n
        return ce

    def gradient_softmax(self, X: np.ndarray, y_pred: np.ndarray, y: np.ndarray):
        """Градиент кросс-энтропийной функции потерь"""
        return np.array(1 / y_pred.shape[0] * np.dot(X.T, (y - y_pred)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучение линейной регрессии на градиентном спуске: модификация Adam"""
        epoches = 0
        n_objects, n_features = X.shape
        self.w = np.random.normal(size=n_features)
        y_pred = X @ self.w
        pred_w = np.random.normal(size=n_features)
        while epoches < self.max_epoches:
            epoches += 1
            indices = np.random.choice(X.shape[0], size=self.size_batch, replace=False)
            # вычисление батчей
            X_batch = X[indices]
            y_batch = y[indices]
            # вычисление предсказаний
            y_pred = X_batch @ self.w
            # вычисление градиента
            grad = self._deriative_log_loss(X_batch, y_pred, y_batch)
            # oбновление весов
            self.w -= self.learning_rate * grad
            # проверка на то, что веса действительно изменились
            current_loss = self._log_loss(pred_w, self.w)

    def predict_proba(self, X_test):
        """Предсказываем вероятности соотнесения к объекту"""
        return X_test @ self.w

    def predict(self, X_test):
        """Предсказываем классы"""
        y_pred = self.predict_proba(X_test)
        y_pred = np.where(y_pred > 0.55, 1, -1)
        return y_pred

    def fit_test(self, X_train, y_train, X_test, y_test, epochs):
        self.w = np.zeros(X_train.shape[1])
        self.train_losses = np.zeros(self.max_epoches)
        self.test_metrics = np.zeros(self.max_epoches)
        for epoch in range(self.max_epoches):
            y_pred = X_train @ self.w
            gradient = self._deriative_log_loss(X_train, y_pred, y_train)
            self.w -= self.learning_rate * gradient
            self.train_losses[epoch] = self._log_loss(y_train, self.predict(X_train))
            self.test_metrics[epoch] = f1_score(y_test, self.predict(X_test))
