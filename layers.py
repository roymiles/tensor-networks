import tensorflow as tf
from base import *


class Layer:
    """ Most generic abstract class """

    def __init__(self):
        pass

    def num_parameters(self):
        return 0

    def __call__(self):
        raise Exception("This is not how you are supposed to call this layer")


class WeightLayer(Layer):
    """ All layer types that contain weights """

    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    def num_parameters(self):
        # Override parent function in this case
        n = 1
        for size in self._shape:
            n *= size

        return n

    def get_shape(self):
        return self._shape


class ConvLayer(WeightLayer):
    def __init__(self, shape, strides=[1, 1, 1, 1], use_bias=True, padding="SAME"):
        super().__init__(shape)
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, input, kernel, bias=None):
        net = tf.nn.conv2d(input, kernel, strides=self._strides, padding=self._padding)

        if bias:
            net = tf.nn.bias_add(net, bias)

        return net


class FullyConnectedLayer(WeightLayer):
    def __init__(self, shape, use_bias=True):
        super().__init__(shape)
        self._use_bias = use_bias

    def using_bias(self):
        return self._use_bias

    def __call__(self, input, kernel, bias=None):
        net = tf.linalg.matmul(input, kernel)

        if bias:
            net = tf.nn.bias_add(net, bias)

        return net


class PoolingLayer(Layer):
    def __init__(self, ksize, strides=None):
        """ In this case shape is the receptive field size to average over """
        super().__init__()
        self._ksize = ksize

        if strides:
            self._strides = strides
        else:
            # Just shift by the pool size (default pooling operation)
            self._strides = ksize

    def get_ksize(self):
        return self._ksize

    def get_strides(self):
        return self._strides


class AveragePoolingLayer(PoolingLayer):
    def __init__(self, shape):
        super().__init__(shape)

    def __call__(self, input):
        return tf.nn.avg_pool(input, ksize=super(AveragePoolingLayer, self).get_ksize(),
                              strides=super(AveragePoolingLayer, self).get_strides(), padding="SAME")


class MaxPoolingLayer(PoolingLayer):
    def __init__(self, shape):
        super().__init__(shape)

    def __call__(self, input):
        return tf.nn.max_pool(input, ksize=super(MaxPoolingLayer, self).get_ksize(),
                              strides=super(MaxPoolingLayer, self).get_strides(), padding="SAME")


class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self._keep_prob = keep_prob

    def __call__(self, input):
        return tf.nn.dropout(input, self._keep_prob)


class BatchNormalisationLayer(Layer):
    def __init__(self, num_features, affine=True, variance_epsilon=0.001):
        """ If affine is False, the scale and offset parameters won't be used """
        super().__init__()
        self._num_features = num_features
        self._affine = affine
        self._variance_epsilon = variance_epsilon

    def is_affine(self):
        return self._affine

    def get_num_features(self):
        return self._num_features

    def get_variance_epsilon(self):
        return self._variance_epsilon

    def __call__(self, input, mean, variance, offset, scale):
        return tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=self._variance_epsilon)


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.relu(input)


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.softmax(input)


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.layers.flatten(input)

