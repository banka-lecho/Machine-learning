import numpy as np

"""Ядра"""


def uniform(r: float) -> float:
    """ Uniform kernel """
    return 1 / 2


def triangular(r: float) -> float:
    """ Triangular kernel """
    return 1 - abs(r)


def quartic(r: float) -> float:
    """ Quartic kernel """
    return 15 / 16 * (1 - r ** 2) ** 2


def gaussian(r: float) -> float:
    """ Gaussian kernel """
    return np.exp(-0.5 * r ** 2) / np.sqrt(2 * np.pi)
