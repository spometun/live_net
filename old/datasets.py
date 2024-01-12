import numpy as np


def get_xor():
    x = np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return x, y


def get_xor0():
    x = np.array([[0.0, 0], [0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [1], [1], [0]])
    return x, y
