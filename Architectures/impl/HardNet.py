from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, BatchNormalisationLayer, ReLU, DropoutLayer
import tensorflow as tf


class HardNet(IArchitecture):
    # See: https://github.com/DagnyT/hardnet
    def __init__(self):
        network = [
            ConvLayer(shape=[3, 3, 1, 32], use_bias=False),
            BatchNormalisationLayer(32, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32], use_bias=False),
            BatchNormalisationLayer(32, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 64], use_bias=False, strides=[1, 2, 2, 1]),
            BatchNormalisationLayer(64, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64], use_bias=False),
            BatchNormalisationLayer(64, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 128], use_bias=False, strides=[1, 2, 2, 1]),
            BatchNormalisationLayer(128, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 128, 128], use_bias=False),
            BatchNormalisationLayer(128, affine=False),
            ReLU(),
            DropoutLayer(0.3),
            ConvLayer(shape=[8, 8, 128, 128], use_bias=False, padding="VALID"),
            BatchNormalisationLayer(128, affine=False)
        ]

        super().__init__(network)

    @staticmethod
    def input_norm(x, eps=1e-6):
        x_flatten = tf.layers.flatten(x)
        x_mu, x_std = tf.nn.moments(x_flatten, axes=[1])
        # Add extra dimension
        x_mu = tf.expand_dims(x_mu, axis=1)
        x_std = tf.expand_dims(x_std, axis=1)
        x_norm = (x_flatten - x_mu) / (x_std + eps)

        # From 1024 back to 32 x 32 x 1
        s = x.get_shape().as_list()
        return tf.reshape(x_norm, shape=(-1, s[1], s[2], s[3]))

    @staticmethod
    def start(input):
        """ Called before running network """
        input_norm = HardNet.input_norm(input)
        return input_norm  # Then fed into network

    @staticmethod
    def end(x_features):
        """ Input is output of network """
        x = tf.layers.flatten(x_features)
        x_norm = tf.math.l2_normalize(x, axis=1)
        return x_norm