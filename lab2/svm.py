from lab2.kernels import *
from sklearn.metrics import f1_score


class SVMClassifier:
    def __init__(self, c, learning_rate, epochs, kernel, r=0, n=2, sigma=1, alpha=1):
        dict_kernels = {'polynomial_kernel': polynomial_kernel, 'gaussian_kernel': gaussian_kernel,
                        'linear_kernel': linear_kernel}
        dict_gr_kernels = {'polynomial_kernel': polynomial_kernel_gradient, 'gaussian_kernel': gaussian_kernel_gradient,
                           'linear_kernel': linear_kernel_gradient}
        self.c = c
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = []
        self.kernel = dict_kernels[kernel]
        self.kernel_gradient = dict_gr_kernels[kernel]
        self.train_losses = np.zeros(self.epochs)
        self.test_losses = np.zeros(self.epochs)

    def loss(self, X, y):
        value = 0
        n = X.shape[0]
        for i in range(n):
            value += max(0, 1 - np.dot(X.iloc[i], self.w) * y.iloc[i])
        return value / n

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            gradient = self.kernel_gradient(X, y, self.w) + 1 / self.c * self.w
            self.w -= self.learning_rate * gradient

    def predict(self, X):
        return np.array([1 if self.kernel(X.iloc[i], self.w) >= 1 else -1 for i in range(X.shape[0])])

    def fit_test(self, X_train, y_train, X_test, y_test):
        self.w = np.zeros(X_train.shape[1])
        for epoch in range(self.epochs):
            gradient = self.kernel_gradient(X_train, y_train, self.w) + 1 / self.c * self.w
            self.w -= self.learning_rate * gradient
            self.train_losses[epoch] = self.loss(X_train, y_train)
            self.test_losses[epoch] = f1_score(y_test, self.predict(X_test))
