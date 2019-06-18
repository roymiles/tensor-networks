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

    def __call__(self, input, kernel, bias=None):
        # Depthwise convolution
        net = tf.nn.depthwise_conv2d(input, kernel, strides=self._strides, padding=self._padding)

        if bias:
            net = tf.nn.bias_add(net, bias)

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
    def __init__(self, shape, build_method=Weights.impl.core, use_bias=True, kernel_initializer=tf.glorot_normal_initializer(),
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

    def __call__(self, input, kernel, bias=None):
        # net = tf.linalg.matmul(input, kernel)
        net = tf.tensordot(input, kernel, axes=[1, 0])

        if bias:
            net = tf.nn.bias_add(net, bias)

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
    def __init__(self, switch_list=[1.0], affine=True):
        """ Implements switchable batch normalisation
            If affine is False, the scale and offset parameters won't be used
            When affine=False the output of BatchNorm is equivalent to considering gamma=1 and beta=0 as constants. """
        super().__init__()
        self._affine = affine
        self._switch_list = switch_list

        self.switchable_fns = []

    def get_switches(self, input, is_training, switch_idx):
        pred_fn_pairs = []
        for sw_idx, sw in enumerate(self._switch_list):
            with tf.variable_scope(f"switch_{sw}"):
                pred_fn_pairs.append((tf.equal(switch_idx, sw_idx),
                                      lambda: tf.layers.batch_normalization(input, training=is_training)))

        return pred_fn_pairs

    def __call__(self, input, is_training, switch_idx, affine=True):

        #net = tf.case(self.get_switches(input, is_training, switch_idx),
        #              default=lambda: tf.layers.batch_normalization(input, training=is_training),
        #              exclusive=True)
        #return net

        # Independant batch normalisation (for each switch)
        # with tf.variable_scope(f"switch"):
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


""" Multilayer classes e.g. residual layers etc """


class MobileNetV2BottleNeck(ILayer):
    def __init__(self, k, t, c, build_method=Weights.impl.core, strides=(1, 1)):
        """
        See: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c

        :param k: Number of input channels
        :param t: Expansion factor
        :param c: Number of output channels
        :param strides: Depthwise stride
        """
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]
        self._build_method = build_method
        self._k = k
        self._t = t
        self._c = c

        super().__init__()

    def create_weights(self):
        return self._build_method.mobilenetv2_bottleneck

    def __call__(self, input, expansion_kernel, depthwise_kernel, projection_kernel, switch_idx=0):

        # Expansion layer
        net = tf.nn.conv2d(input, expansion_kernel, strides=[1, 1, 1, 1], padding="SAME")
        with tf.variable_scope(f"bn1_switch_{switch_idx}"):
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)

        # Depthwise layer
        net = tf.nn.depthwise_conv2d(net, depthwise_kernel, strides=self._strides, padding="SAME")
        with tf.variable_scope(f"bn2_switch_{switch_idx}"):
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)

        # Projection layer (linear)
        net = tf.nn.conv2d(net, projection_kernel, strides=[1, 1, 1, 1], padding="SAME")
        with tf.variable_scope(f"bn3_switch_{switch_idx}"):
            net = tf.layers.batch_normalization(net)

        # Only residual add when strides is 1
        is_residual = True if self._strides == [1, 1, 1, 1] else False
        is_conv_res = False if net.get_shape().as_list()[3] == input.get_shape().as_list()[3] else True

        if is_residual:

            if is_conv_res:
                # See: https://github.com/MG2033/MobileNet-V2/blob/master/layers.py
                # If not matching channels, place a 1x1 convolution to ensure match
                x = tf.layers.conv2d(input, net.get_shape().as_list()[3], (1, 1), use_bias=False,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(),
                                     name="fix-channel-mismatch")

                with tf.variable_scope(f"switch_{switch_idx}"):
                    x = tf.layers.batch_normalization(x)

                return net + x
            else:
                return net + input
        else:
            return net

    def get_k(self):
        return self._k

    def get_t(self):
        return self._t

    def get_c(self):
        return self._c

    def get_strides(self):
        return self._strides


class PointwiseDot(ILayer):
    """
        Potential replacement for pointwise convolution
    """
    def __init__(self, shape, build_method=Weights.impl.sandbox, use_bias=True,
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
        return self._build_method.pointwise_dot

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def __call__(self, input, c, g, n, bias1=None, bias2=None, bias3=None, is_training=True, switch_idx=0):
        # input : B x w x h x c
        # c     : c x r1
        # g     : r1 x r2
        # n     : r2 x n
        net = tf.tensordot(input, c, axes=[3, 0])
        if bias1:
            net = tf.nn.bias_add(net, bias1)

        with tf.variable_scope(f"bn1_switch_{switch_idx}"):
            net = tf.layers.batch_normalization(net, training=is_training)

        # B x w x h x r1
        net = tf.tensordot(net, g, axes=[3, 0])
        if bias2:
            net = tf.nn.bias_add(net, bias2)

        with tf.variable_scope(f"bn2_switch_{switch_idx}"):
            net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # B x w x h x r2
        net = tf.tensordot(net, n, axes=[3, 0])
        if bias3:
            net = tf.nn.bias_add(net, bias3)

        with tf.variable_scope(f"bn3_switch_{switch_idx}"):
            net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # B x w x h x n
        return net
