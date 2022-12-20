import tensorflow as tf
import keras
import keras.regularizers


# L1 regularizer with slightly different strengths for each weight,
# bo break symmetry
class RegularizerL1Assymetric(keras.regularizers.Regularizer):

    def __init__(self, strength):
        self.base_strength = tf.cast(strength, tf.float32)
        self.strength = None

    def __call__(self, x):
        if self.strength is None:
            with tf.init_scope():
                k = 0.2
                minval = (1 - k) * self.base_strength
                maxval = (1 + k) * self.base_strength
                seed = (42, 24)
                shape = x.shape
                self.strength = tf.random.stateless_uniform(seed=seed, shape=shape, minval=minval, maxval=maxval)
        return tf.reduce_sum(tf.multiply(self.strength, tf.abs(x)))
