import numpy as np
import pytest
import tensorflow as tf


# calculates L-ord norm over last axis
# input must have rank = 2, output will have rank = 1
def stable_norm(tensor, ord):
    tf.debugging.assert_equal(tf.rank(tensor), 2)

    def _norm(x):
        ma = tf.reduce_max(tf.abs(x), axis=1, keepdims=True)
        # print(f"ma {ma.shape}")
        x1 = x / ma
        # print(f"x1 {x1.shape}")
        po = tf.pow(x1, ord)
        sum_ = tf.reduce_sum(po, axis=1, keepdims=True)
        # print(f"sum {sum_.shape}")
        res = tf.pow(sum_, 1 / ord) * ma
        # print(f"res {res.shape}")
        return tf.squeeze(res, axis=1)

    def _abs(x):
        return tf.reduce_sum(tf.abs(x), axis=1)

    non_zero = tf.math.count_nonzero(tensor, axis=1)
    non_zero_inds = tf.where(non_zero)
    zero_inds = tf.where(non_zero == 0)
    res = tf.zeros_like(non_zero, dtype=tensor.dtype)
    for_norm = tf.gather(tensor, tf.squeeze(non_zero_inds, 1))
    norms_general = _norm(for_norm)
    # for_abs = tf.gather(tensor, tf.squeeze(zero_inds, 1))
    # norms_abs = _abs(for_abs)
    # print(norms_general)
    # print(norms_abs)
    res = tf.tensor_scatter_nd_update(res, non_zero_inds, norms_general)
    return res


def test_stable_norm():
    print("hi")
    x = tf.constant([[1e-30, -1e-20], [0, -0.1], [0, 0], [3, 4]])
    y = stable_norm(x, 2)
    y = y.numpy()
    expected = np.array([1e-20, 0.1, 0, 5])
    assert np.allclose(y, expected)

