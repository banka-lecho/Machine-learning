import numpy as np


def linear_kernel(u, v):
    return np.dot(u, v)


def linear_kernel_gradient(X, y, w):
    value = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        if y.iloc[i] * linear_kernel(X.iloc[i], w) < 1:
            value -= y.iloc[i] * X.iloc[i]
    return value


def polynomial_kernel(u, v, r=0, n=2):
    return (np.dot(u, v) + r) ** n


def polynomial_kernel_gradient(X, y, w, r=0, n=2):
    value = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        if y.iloc[i] * polynomial_kernel(X.iloc[i], w, r=0, n=2) < 1:
            value -= y.iloc[i] * n * (np.dot(X.iloc[i], w) + r) ** (n - 1) * X.iloc[i]
    return value


def gaussian_kernel(u, v, sigma=1):
    return np.exp(-(np.dot(u - v, u - v)) / (2 * sigma ** 2))


def gaussian_kernel_gradient(X, y, w, sigma=1):
    value = np.zeros(X.shape[1])
    sigma2 = sigma * sigma
    for i in range(X.shape[0]):
        if y.iloc[i] * gaussian_kernel(X.iloc[i], w, sigma) < 1:
            diff = X.iloc[i] - w
            value -= y.iloc[i] / sigma2 * np.exp(-np.dot(diff, diff) / (2 * sigma2)) * diff
    return value
