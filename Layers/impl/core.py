import tensorflow as tf
from Layers.layer import ILayer
import Weights.impl.core
import Weights.impl.sandbox
import math
import numpy as np


class ConvLayer(ILayer):
    def __init__(self, shape, build_method=Weights.impl.core, strides=(1, 1), use_bias=True, padding="SAME",
                 kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None, ranks=None):
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]
        self._build_method = build_method

        self._shape = shape
        self._padding = padding
        self._use_bias = use_bias

        # Can't be bothered to make getters for these, just make them public
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # If using some form of tensor decomposition
        self.ranks = ranks

    def create_weights(self):
        return self._build_method.convolution

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, input, kernel, bias=None):
        net = tf.nn.conv2d(input, kernel, strides=self._strides, padding=self._padding)

        if bias:
            net = tf.nn.bias_add(net, bias)

        return net


class ConvLayerConstant(ILayer):
    """ Same as ConvLayer, but fixed, constant filter """
    def __init__(self, kernel, build_method=Weights.impl.core, strides=[1, 1, 1, 1], padding="SAME"):
        super().__init__()
        self._kernel = kernel
        self._build_method = build_method
        self._strides = strides
        self._padding = padding

    def create_weights(self):
        return self._build_method.convolution

    def get_kernel(self):
        return self._kernel

    def get_strides(self):
        return self._strides

    def __call__(self, input):
        return tf.nn.conv2d(input, self._kernel, strides=self._strides, padding=self._padding)


class DepthwiseConvLayer(ILayer):
    """ Depthwise convolution
        NOTE: Pointwise convolution uses standard conv layer """
    def __init__(self, shape, build_method=Weights.impl.core, strides=(1, 1), use_bias=True, padding="SAME",
                 kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None):
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]
        self._build_method = build_method

        self._shape = shape
        self._padding = padding
        self._use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def create_weights(self):
        return self._build_method.depthwise_convolution

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, input, weights):
        # Depthwise convolution
        net = tf.nn.depthwise_conv2d(input, weights.kernel, strides=self._strides, padding=self._padding)

        if weights.bias:
            net = tf.nn.bias_add(net, weights.bias)

        return net


class BiasLayerConstant(ILayer):
    """ Fixed, constant bias added to a layer """
    def __init__(self, bias):
        super().__init__()
        self._bias = bias

    def get_bias(self):
        return self._bias

    def __call__(self, input):
        return tf.nn.bias_add(input, self._bias)


class FullyConnectedLayer(ILayer):
    def __init__(self, shape, build_method=Weights.impl.core, use_bias=True,
                 kernel_initializer=tf.glorot_normal_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None):

        super().__init__()
        self._shape = shape
        self._build_method = build_method
        self._use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def create_weights(self):
        return self._build_method.fully_connected

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def __call__(self, input, weights):
        # net = tf.linalg.matmul(input, kernel)
        net = tf.tensordot(input, weights.kernel, axes=[1, 0])

        if weights.bias:
            net = tf.nn.bias_add(net, weights.bias)

        return net


# Alias name
Dense = FullyConnectedLayer


class PoolingLayer(ILayer):
    def __init__(self, pool_size=(2, 2)):
        """ In this case shape is the receptive field size to average over """
        px = pool_size[0]
        py = pool_size[1]
        self._ksize = [1, px, py, 1]
        self._strides = [1, px, py, 1]

    def get_ksize(self):
        return self._ksize

    def get_strides(self):
        return self._strides


class AveragePoolingLayer(PoolingLayer):
    def __init__(self, pool_size):
        super().__init__(pool_size)

    def __call__(self, input):
        return tf.nn.avg_pool(input, ksize=super(AveragePoolingLayer, self).get_ksize(),
                              strides=super(AveragePoolingLayer, self).get_strides(), padding="SAME")


class MaxPoolingLayer(PoolingLayer):
    def __init__(self, pool_size):
        super().__init__(pool_size)

    def __call__(self, input):
        return tf.nn.max_pool(input, ksize=super(MaxPoolingLayer, self).get_ksize(),
                              strides=super(MaxPoolingLayer, self).get_strides(), padding="SAME")


class GlobalAveragePooling:
    """ Pool over entire spatial dimensions"""
    def __init__(self, keep_dims=True):
        self._keep_dims = keep_dims

        super().__init__()

    def __call__(self, input):
        return tf.reduce_mean(input, [1, 2], keep_dims=self._keep_dims, name='global_pool')


class DropoutLayer(ILayer):
    def __init__(self, rate):
        # rate defines the fraction of input units to drop
        super().__init__()
        self._rate = rate

    def __call__(self, input):
        return tf.nn.dropout(input, rate=self._rate)


class BatchNormalisationLayer(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input, is_training):
        net = tf.layers.batch_normalization(input, training=is_training)
        return net


class NonLinearityLayer(ILayer):
    def __init__(self):
        super().__init__()


class ReLU(NonLinearityLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.relu(input)


class ReLU6(NonLinearityLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.relu6(input)


class HSwish(NonLinearityLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x * tf.nn.relu6(x + 3) / 6


class SoftMax(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.softmax(input)


class Flatten(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.layers.flatten(input)