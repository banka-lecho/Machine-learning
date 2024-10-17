"""Метрики"""
import numpy as np


def euclidean(x: np.array, x_neighbor: np.array) -> float:
    """ Euclidean distance between two objects """
    return float(np.sqrt(np.sum(np.square(x - x_neighbor))))


def cosine(x: np.array, x_neighbor: np.array) -> float:
    """ Cosine similarity between two objects """
    result = float(1 - np.dot(x, x_neighbor.T) / (np.linalg.norm(x) * np.linalg.norm(x_neighbor)))
    return result


def minkowski_distance(x: np.array, x_neighbor: np.array) -> float:
    """ Minkowski distance between two objects """

    def p_root(sum_diff, root):
        root_value = 1 / float(root)
        return (sum_diff ** root_value).astype(float)

    p_value = 1
    sum_diff = sum(pow(abs(a - b), p_value) for a, b in zip(x, x_neighbor))
    return float(p_root(sum_diff, p_value)[0])
