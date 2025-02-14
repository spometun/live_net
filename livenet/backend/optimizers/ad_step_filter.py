import math
import random

import numpy as np
import pytest

from ai_libs.simple_log import LOG

class UnitFilter:
    def __init__(self):
        pass

    def process_value(self, value: np.ndarray):
        assert np.isfinite(value).all(), f"Invalid input value {value}"
        return np.ones_like(value)


class AdStepFilter:
    # output is always in between alpha and 1 where alpha is a small value equal to 1 / (2 * window_length)
    # TODO: Support tensor/numpy array as an input
    def __init__(self, window_length: int):
        assert window_length >= 1, f"Invalid input window_length {window_length}"
        self.last_sign = None
        self.last_last_sign = None
        self.window_length = window_length
        self.q = (window_length - 1) / window_length
        self.qt = 1
        self.v = 0.0

    def process_value(self, value: np.ndarray):
        assert np.isfinite(value).all(), f"Invalid input value {value}"
        sign = np.sign(value).astype(value.dtype)
        if self.last_last_sign is None:
            self.last_last_sign = self.last_sign
            self.last_sign = sign
            return np.ones_like(sign)

        # v = 1.0 if sign * self.last_sign > 0 and self.last_last_sign * self.last_sign > 0 else 0.0
        v = np.logical_and(np.sign(sign * self.last_sign) == 1, np.sign(sign * self.last_last_sign) == 1)  # 'True' only if 3 signs equal '1' or '-1' in a row
        # v = 1.0 if sign * self.last_sign > 0 else 0.0
        # LOG(v)

        self.last_last_sign = self.last_sign
        self.last_sign = sign

        self.qt *= self.q
        self.v = self.q * self.v + (1 - self.q) * v
        result = self.v / (1 - self.qt)
        eps = 1e-6
        assert np.all(-eps <= result) and np.all(result <= 1 + eps)
        result = np.clip(result, 1 / (2 * self.window_length), 1)
        return result


def test_ad_step_filter_array():
    rng = random.Random(x=42)
    f = AdStepFilter(10)
    for i in range(20):
        v1 = rng.randint(0, 10) - 5
        v2 = rng.randint(0, 10) - 9
        v = np.array([v1, v2])

        # v = 20 * (i % 2) - 10
        r = f.process_value(v)
        LOG(i, f"{r[0]:.2g} {r[1]:.2g}", v)


def test_ad_step_filter2():
    rng = random.Random(x=42)
    f = AdStepFilter(10)
    for i in range(20):
        v = rng.randint(0, 10) - 5
        # v = 20 * (i % 2) - 10
        r = f.process_value(v)
        LOG(i, f"{r:.2g}", v)


def test_ad_step_filter():
    LOG()
    f = AdStepFilter(500)
    for i in range(3000):
        if i < 2500:
            v = 20 * (i % 2) - 10
        else:
            if i == 2500:
                assert r <= 0.001
            v = 100  # absolute value should not matter
        r = f.process_value(v)
    assert r == pytest.approx(1 - math.exp(-1), 0.01)  # relaxation time

    f = AdStepFilter(20)
    rng = random.Random(x=42)
    for i in range(100):
        v = 2 * rng.randint(0, 1) - 1
        r = f.process_value(v)
        LOG(i, r)
    assert r == pytest.approx(0.2725, 0.1)
