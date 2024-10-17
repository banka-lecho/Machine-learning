import numpy as np
from sklearn.metrics import f1_score


class SVMClassifier:
    def __init__(self, c, learning_rate, epochs, sigma=1):
        self.c = c
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = []

    def loss(self, X, y):
        value = 0
        n = X.shape[0]
        for i in range(n):
            value += max(0, 1 - np.dot(X.iloc[i], self.w) * y.iloc[i])
        return value / n

    def gaussian_kernel(self, u, v):
        return np.exp(-(np.dot(u - v, u - v)) / (2 * self.sigma ** 2))

    def gaussian_kernel_gradient(self, X, y, w):
        value = np.zeros(X.shape[1])
        sigma2 = self.sigma * self.sigma
        for i in range(X.shape[0]):
            if y.iloc[i] * self.gaussian_kernel(X.iloc[i], w) < 1:
                diff = X.iloc[i] - w
                value -= y.iloc[i] / sigma2 * np.exp(-np.dot(diff, diff) / (2 * sigma2)) * diff
        return value

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            gradient = self.gaussian_kernel_gradient(X, y, self.w) + 1 / self.c * self.w
            self.w -= self.learning_rate * gradient

    def predict(self, X):
        return np.array([1 if self.gaussian_kernel(X.iloc[i], self.w) >= 1 else -1 for i in range(X.shape[0])])

    def fit_test(self, X_train, y_train, X_test, y_test):
        self.w = np.zeros(X_train.shape[1])
        self.train_losses = np.zeros(self.epochs)
        self.test_losses = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            gradient = self.gaussian_kernel_gradient(X_train, y_train, self.w) + 1 / self.c * self.w
            self.w -= self.learning_rate * gradient
            self.train_losses[epoch] = self.loss(X_train, y_train)
            self.test_losses[epoch] = f1_score(y_test, self.predict(X_test))
