import numpy as np
from tqdm import tqdm


class Regression_LOWESS:
    def __init__(self, method, use_weighing=True, f=0.2, learning_rate=0.1, n_iterations=1000):
        self.decision_method = method
        self.use_weighing = use_weighing
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.f = f
        self.lowess_weights = []

    @staticmethod
    def tricube_weight(distance) -> float:
        """Вычисление весов по трикубической функции."""
        weight = (1 - np.abs(distance) ** 3) ** 3
        weight[distance > 1] = 0
        return weight

    def gradient_descent(self, x, y, weights) -> (float, float):
        """Градиентный спуск для линейной регрессии."""
        m = len(y)
        beta_0 = 0  # Смещение
        beta_1 = 0  # Наклон

        for _ in range(self.n_iterations):
            predictions = beta_0 + beta_1 * x
            errors = predictions - y

            # Вычисляем градиенты, взвешенные по весам
            gradient_0 = (1 / m) * np.sum(weights * errors)
            gradient_1 = (1 / m) * np.sum(weights * errors * x)

            # Обновление параметров
            beta_0 -= self.learning_rate * gradient_0
            beta_1 -= self.learning_rate * gradient_1

        return beta_0, beta_1

    @staticmethod
    def analytical_decision(x, y, weights) -> (float, float):
        A = np.vstack([np.ones(len(x)), x]).T
        W = np.diag(weights)
        beta = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
        return beta[0], beta[1]

    def fit_predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Реализация LOWESS с помощью градиентного спуска."""
        n = len(y)
        smoothed = np.zeros(n)
        self.lowess_weights = [0] * n
        r = int(self.f * n)
        for i in tqdm(range(n), desc="Обучение lOWESS"):
            # Вычисляем расстояния от текущего x до всех остальных
            distances = np.abs(x - x[i])
            # Получаем индексы определенного количества соседей
            idx = np.argsort(distances)[:r]
            # Извлекаем x и y для соседей
            x_neighbors = x[idx]
            y_neighbors = y[idx]
            dists_neighbours = distances[idx]
            # Вычисляем веса каждого соседа для объекта
            weights = self.tricube_weight(dists_neighbours / dists_neighbours.max())
            # теперь чтобы достать это в knn, мне надо для каждого объекта положить в этот массив таргет
            # и веса для этого объекта
            self.lowess_weights[i] = (list(y_neighbors), list(weights))
            if self.decision_method == 'gradient_descent':
                # находим взвешенную линейную регрессию с помощью ГС
                beta_0, beta_1 = self.gradient_descent(x_neighbors, y_neighbors, weights)
            else:
                # находим взвешенную линейную регрессию с помощью аналитического метода
                beta_0, beta_1 = self.analytical_decision(x_neighbors, y_neighbors, weights)
            smoothed[i] = beta_0 + beta_1 * x[i]
        return smoothed
