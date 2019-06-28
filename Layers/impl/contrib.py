import tensorflow as tf
from Layers.layer import ILayer
import Weights.impl.core
import Weights.impl.sandbox
import math
import numpy as np


class SwitchableBatchNormalisationLayer(ILayer):
    def __init__(self, switch_list=[1.0], affine=True):
        raise Exception("Switchable batch norm is not implemented yet")
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

        # TODO: Implement switchable batch norm, but maybe in contrib
        # net = tf.case(self.get_switches(input, is_training, switch_idx),
        #              default=lambda: tf.layers.batch_normalization(input, training=is_training),
        #              exclusive=True)
        # return net

        # Independant batch normalisation (for each switch)
        # with tf.variable_scope(f"switch"):
        net = tf.layers.batch_normalization(input, training=is_training)
        return net


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

    def __call__(self, input, weights, is_training):

        # Expansion layer
        net = tf.nn.conv2d(input, weights.expansion_kernel, strides=[1, 1, 1, 1], padding="SAME")
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # Depthwise layer
        net = tf.nn.depthwise_conv2d(net, weights.depthwise_kernel, strides=self._strides, padding="SAME")
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # Projection layer (linear)
        net = tf.nn.conv2d(net, weights.projection_kernel, strides=[1, 1, 1, 1], padding="SAME")
        net = tf.layers.batch_normalization(net, training=is_training)

        # Only residual add when strides is 1
        is_residual = True if self._strides == [1, 1, 1, 1] else False
        # is_conv_res = False if net.get_shape().as_list()[3] == input.get_shape().as_list()[3] else True

        if is_residual:

            if net.get_shape().as_list()[3] == input.get_shape().as_list()[3]:
                return net + input
            else:
                return net

            """if is_conv_res:
                # See: https://github.com/MG2033/MobileNet-V2/blob/master/layers.py
                # If not matching channels, place a 1x1 convolution to ensure match
                x = tf.layers.conv2d(input, net.get_shape().as_list()[3], (1, 1), use_bias=False,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(),
                                     name="fix-channel-mismatch")

                with tf.variable_scope(f"switch_{switch_idx}"):
                    x = tf.layers.batch_normalization(x)

                return net + x
            else:
                return net + input"""
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

# class OffsetConvolutionLayer(ILayer):
    """
        Factors 
    """


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

    def __call__(self, input, weights, is_training=True):
        # input : B x w x h x c
        # c     : c x r1
        # g     : r1 x r2
        # n     : r2 x n
        net = tf.tensordot(input, weights.c, axes=[3, 0])
        net = tf.layers.batch_normalization(net, training=is_training)
        # No ReLU here?

        # B x w x h x r1
        net = tf.tensordot(net, weights.g, axes=[3, 0])
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # B x w x h x r2
        net = tf.tensordot(net, weights.n, axes=[3, 0])
        if weights.bias:
            net = tf.nn.bias_add(net, weights.bias)

        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # B x w x h x n
        return net


class CustomBottleneck(ILayer):
    def __init__(self, shape, strides=(1, 1), use_bias=True, padding="SAME", partitions=[0.8, 0.8],
                 kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None, ranks=None):
        """
            Custom implementation for the depthwise separable layers

            The pointwise convolution is separated across the input channel dimensions
            Whereas the depthwise + standard convolution
        """
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]

        # The two partitions
        self.partitions = partitions

        self._shape = shape
        self._padding = padding
        self._use_bias = use_bias

        # Can't be bothered to make getters for these, just make them public (I trust myself not to touch them)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Rank for the core tensor G
        self.ranks = ranks

    def create_weights(self):
        # Only one way to create these weights for now
        return Weights.impl.sandbox.custom_bottleneck

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, input, weights, is_training):
        if weights.conv_kernel is not None:
            # Standard convolution
            size = weights.conv_kernel.get_shape().as_list()[2]  # W x H x C x N
            conv_out = tf.nn.conv2d(input[:, :, :, 0:size], weights.conv_kernel, strides=self._strides, padding=self._padding)
            offset = size
        else:
            offset = 0

        if weights.depthwise_kernel is not None:
            # Depthwise separable stage
            dw_out = tf.nn.depthwise_conv2d(input[:, :, :, offset:], weights.depthwise_kernel, strides=self._strides,
                                            padding=self._padding)

        if weights.conv_kernel is not None and weights.depthwise_kernel is None:
            net = conv_out
        elif weights.depthwise_kernel is not None and weights.conv_kernel is None:
            net = dw_out
        elif weights.conv_kernel is not None and weights.depthwise_kernel is not None:
            # Combine depthwise + standard results
            net = tf.concat([conv_out, dw_out], axis=3)
        else:
            raise Exception("No standard or depthwise kernels.")

        # BN + ReLU
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)

        # Pointwise
        if weights.pointwise_kernel is not None:
            pw_out = tf.nn.conv2d(net, weights.pointwise_kernel, strides=self._strides, padding=self._padding)

        # Factored pointwise
        if weights.factored_pointwise_kernel is not None:
            fpw_out = tf.nn.conv2d(net, weights.factored_pointwise_kernel, strides=self._strides, padding=self._padding)

        if weights.pointwise_kernel is not None and weights.factored_pointwise_kernel is None:
            net = pw_out
        elif weights.factored_pointwise_kernel is not None and weights.pointwise_kernel is None:
            net = fpw_out
        elif weights.pointwise_kernel is not None and weights.factored_pointwise_kernel is not None:
            # Combine pointwise results
            net = tf.concat([pw_out, fpw_out], axis=3)
        else:
            raise Exception("No pointwise or factored pointwise kernels.")

        if weights.bias:
            net = tf.nn.bias_add(net, weights.bias)

        return net


class DenseBlock(ILayer):
    def __init__(self, name, in_channels, num_layers, growth_rate, dropout_rate, build_method=Weights.impl.sandbox,
                 kernel_initializer=tf.glorot_normal_initializer(),  bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None):
        """

        :param name: Variable scope
        :param N: How many layers
        """

        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.build_method = build_method
        self.dropout_rate = dropout_rate

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def create_weights(self):
        return self.build_method.dense_block

    def add_layer(self, name, input, pointwise_kernel, conv_kernel, is_training):
        with tf.variable_scope(name):
            net = tf.layers.batch_normalization(input, training=is_training)
            net = tf.nn.relu(net)

            # 1x1 conv
            net = tf.nn.conv2d(net, pointwise_kernel, strides=[1, 1, 1, 1], padding="SAME")
            net = tf.layers.dropout(net, rate=self.dropout_rate, training=is_training)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)

            net = tf.nn.conv2d(net, conv_kernel, strides=[1, 1, 1, 1], padding="SAME")
            net = tf.layers.dropout(net, rate=self.dropout_rate, training=is_training)
            net = tf.concat([input, net], axis=3)
            return net

    def __call__(self, input, weights, is_training):
        """ weights is just the set of conv weights """
        with tf.variable_scope(self.name):
            net = input
            for i in range(self.num_layers):
                net = self.add_layer(f"composite_layer_{i}", net, weights.pointwise_kernel[i], weights.conv_kernel[i],
                                     is_training)
                # output channels = input channels + growth rate

        return net

