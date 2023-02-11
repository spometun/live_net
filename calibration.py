import math

import numpy as np
from typing import Sequence, List, Callable
import pytest
import scipy.special
from matplotlib import pyplot as plt
import bisect


def monotonic_function_from_binary_values(x: Sequence[bool | int]) -> np.ndarray:
    assert len(x) >= 1, "Invalid input"
    steps = []

    def ratio(pair):
        return pair[1] / (pair[0] + pair[1])

    for val in x:
        assert val == 0 or val == 1
        if val == 1:
            steps.append([0, 1])
        else:
            steps.append([1, 0])
        while True:
            if len(steps) == 1:
                break
            prev = ratio(steps[-2])
            cur = ratio(steps[-1])
            if cur > prev:
                break
            steps[-2][0] += steps[-1][0]
            steps[-2][1] += steps[-1][1]
            steps.pop()

    res = []
    for pair in steps:
        res += [ratio(pair)] * (pair[0] + pair[1])
    return np.array(res)


def test_one_zero():
    x = [1, 0]
    f = list(monotonic_function_from_binary_values(x))
    assert f == [0.5, 0.5]


def test_constant():
    assert monotonic_function_from_binary_values([False]) == [0.0]
    assert monotonic_function_from_binary_values([1]) == [1.0]
    assert (monotonic_function_from_binary_values([0, 0]) == [0.0, 0.0]).all()
    assert (monotonic_function_from_binary_values([1, 1]) == [1.0, 1.0]).all()


def test_1per3():
    x = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
    y = list(monotonic_function_from_binary_values(x))
    y_expected = [0, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1, 1]
    assert y == y_expected


def test_ascend():
    x = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
    y = list(monotonic_function_from_binary_values(x))
    y_expected = [0, 1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 3/4, 3/4, 3/4, 3/4]
    assert y == y_expected


def get_binary_entropy(x: Sequence[bool | int]) -> float:
    assert len(x) >= 1
    x = np.array(x)
    n0 = np.sum(x == 0)
    n1 = np.sum(x == 1)
    assert n0 + n1 == len(x)
    p0 = n0 / (n0 + n1)
    p1 = n1 / (n0 + n1)
    nats = - scipy.special.xlogy(p0, p0) - scipy.special.xlogy(p1, p1)
    bits = nats / math.log(2)
    return bits


def test_entropy():
    x = [1]
    y = get_binary_entropy(x)
    assert y == 0.0
    x = [0, 1]
    y = get_binary_entropy(x)
    assert y == 1
    x = [0, 0]
    y = get_binary_entropy(x)
    assert y == 0.0
    x = [0, 1, 0]
    y = get_binary_entropy(x)
    expected = 1 / 3 * math.log2(3) + 2 / 3 * math.log2(3 / 2)
    assert y == pytest.approx(expected)


def _sort_by_scores(y_true: np.ndarray[bool], scores: np.ndarray[float]) -> np.ndarray[bool]:
    assert y_true.ndim == 1 and y_true.shape == scores.shape
    y = [y for x, y in sorted(zip(-scores, y_true))]
    # was sorted by -scores to ensure descending order of y_true for equal scores
    y = y[::-1]
    return np.array(y)


def test_sort_by_scores():
    y_true = np.array([False, False, True, False, True])
    scores = np.array([0.5, 0.2, 0.2, 0.2, 0.3])
    y_sorted = _sort_by_scores(y_true, scores)
    assert list(y_sorted) == [True, False, False, True, False]

    y_true = np.array([1, 2, 0])
    scores = np.array([0.2, 0.2, 0.2])
    y_sorted = _sort_by_scores(y_true, scores)
    assert list(y_sorted) == [2, 1, 0]


def get_calibrated_scores(y_true: np.ndarray[bool], scores: np.ndarray[float]) -> np.ndarray[bool]:
    y_sorted = _sort_by_scores(y_true, scores)
    prob_calibrated = monotonic_function_from_binary_values(y_sorted)
    indices = np.argsort(scores)
    calib_scores = np.empty_like(scores)
    calib_scores[indices] = prob_calibrated
    return calib_scores


# input sorted_values must be sorted in ascending order and all different
# returns (index, weight) such as 0 <= weight <= 1 and
# sorted_values[index] * weight + sorted_values[index + 1] * (1 - weight) = x
def _get_linear_weights(sorted_values: List[float], x: float) -> (int, float):
    assert len(sorted_values) >= 2, "Invalid input"
    assert sorted_values[0] <= x <= sorted_values[-1], "Invalid input"
    i = bisect.bisect_left(sorted_values, x)
    if i == 0:
        return 0, 1.0
    assert sorted_values[i - 1] <= x <= sorted_values[i], "Internal error"
    assert sorted_values[i] != sorted_values[i - 1], "Invalid input"
    delta = sorted_values[i] - sorted_values[i - 1]
    if delta == 0.0:  # very rarely possible because of floating point nature
        return i - 1, 1.0
    weight = (sorted_values[i] - x) / delta
    assert 0 <= weight <= 1, "Internal error"
    return i - 1, weight


def test_get_linear_weights():
    values = [1.0, 5.0]
    x = 4.0
    ind, w = _get_linear_weights(values, x)
    assert ind == 0 and w == 0.25

    values = [2.0, 3.0, 5.0]
    x = 2.0
    ind, w = _get_linear_weights(values, x)
    assert ind == 0 and w == 1.0

    x = 3.0
    ind, w = _get_linear_weights(values, x)
    assert ind == 0 and w == 0.0

    x = 5.0
    ind, w = _get_linear_weights(values, x)
    assert ind == 1 and w == 0.0

    x = 2.5
    ind, w = _get_linear_weights(values, x)
    assert ind == 0 and w == 0.5

    x = 1.9
    with pytest.raises(Exception):
        _get_linear_weights(values, x)

    x = 5.1
    with pytest.raises(Exception):
        _get_linear_weights(values, x)


def _linear_interpolated_function_from_scores(scores_original: np.ndarray[float],
                                              scores_calibrated: np.ndarray[float]) -> Callable[[float], float]:
    assert scores_original.ndim == 1
    assert len(scores_original) == len(scores_calibrated) >= 1
    inds = np.argsort(scores_original)
    orig = list(scores_original[inds])
    calib = list(scores_calibrated[inds])
    assert 0 <= orig[0] and orig[-1] <= 1, "Invalid input"
    assert 0 <= calib[0] and calib[-1] <= 1, "Invalid input"
    if orig[0] != 0.0:
        orig.insert(0, 0.0)
        calib.insert(0, 0.0)
    if orig[-1] != 1.0:
        orig.append(1.0)
        calib.append(1.0)
    i = 1
    while i < len(orig):
        assert orig[i] >= orig[i - 1] and calib[i] >= calib[i - 1], "Invalid input. Not sorted"
        if orig[i] == orig[i - 1]:
            assert calib[i] == calib[i - 1], "Invalid input. orig to calib mapping must be injection"
            orig.pop(i)
            calib.pop(i)
        else:
            i = i + 1

    def func(x: float):
        index, weight = _get_linear_weights(orig, x)
        return calib[index] * weight + calib[index + 1] * (1 - weight)

    return func


def test_function_from_scores():
    original = np.array([0.2, 0.5, 0.1])
    calibrated = np.array([0.3, 0.5, 0.1])
    f = _linear_interpolated_function_from_scores(original, calibrated)
    for i in range(len(original)):
        assert f(original[i]) == calibrated[i]
    assert f(0) == 0.0
    assert f(1) == 1.0
    with pytest.raises(Exception):
        f(1.1)

    x = 0.3
    original = np.array([x])
    calibrated = np.array([x])
    f = _linear_interpolated_function_from_scores(original, calibrated)
    assert f(x) == x
    assert f(2 * x) == pytest.approx(2 * x)

    x = 0.3
    original = np.array([x])
    calibrated = np.array([2 * x])
    f = _linear_interpolated_function_from_scores(original, calibrated)
    assert f(x) == pytest.approx(2 * x)
    assert f(0) == 0
    assert f(1) == 1
    assert f(x / 2) == pytest.approx(x)

    original = np.array([0.2, 0.5, 0.5, 1, 1])
    calibrated = np.array([0.3, 0.5, 0.5, 1, 1])
    f = _linear_interpolated_function_from_scores(original, calibrated)
    assert f(0.2) == 0.3
    assert f(0.5) == 0.5
    assert f(0) == 0
    assert f(1) == 1


def test_calibrated_scores():
    y_true = np.array([0, 1, 0, 0, 1])
    scores = np.array([0.4, 0.4, 0.3, 0.3, 0.8])
    calib_scores = get_calibrated_scores(y_true, scores)
    assert list(calib_scores) == [0.5, 0.5, 0.0, 0.0, 1.0]


def get_calibration_function(y_true, scores):
    calib_scores = get_calibrated_scores(y_true, scores)
    func = _linear_interpolated_function_from_scores(scores, calib_scores)
    return func


def plot_calibrated_curve(scores: np.ndarray, calibrated_scores: np.ndarray,
                          use_scores_as_x_axis=False):
    inds = np.argsort(scores)
    if use_scores_as_x_axis:
        x = scores[inds]
        xlim = (0, 1)
    else:
        x = np.arange(len(scores))
        xlim = (0, len(x))
    plt.plot(x, calibrated_scores[inds], label="calibrated")
    plt.plot(x, scores[inds], "--", label="original")
    plt.xlabel("scores")
    plt.ylabel("calibrated scores")
    plt.title("calibration curve")
    plt.xlim(xlim)
    plt.ylim((0, 1))
    plt.legend()
    plt.show()


def test_plot_calibrated_curve():
    scores = np.array([0.1, 0.5, 0.8])
    calib_scores = np.array([0.2, 0.6, 0.7])
    plot_calibrated_curve(scores, calib_scores)


if __name__ == '__main__':
    assert [0] == [0.0]
    monotonic_function_from_binary_values([0])
