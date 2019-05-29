from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, BatchNormalisationLayer, ReLU, DropoutLayer
import tensorflow as tf


conv_ranks = {
    0: [3, 1, 8],
    3: [3, 8, 8],
    6: [3, 8, 12],
    9: [3, 12, 12],
    12: [3, 12, 18],
    15: [3, 18, 18],
    19: [3, 18, 18]
}

fc_ranks = None


class HardNet(IArchitecture):
    # See: https://github.com/DagnyT/hardnet
    def __init__(self):
        wd = 1e-4  # Weight decay
        gain = 0.6
        init = tf.glorot_normal_initializer()  # tf.keras.initializers.Orthogonal(gain)
        reg = None  # tf.contrib.layers.l1_regularizer(wd)
        network = [
            # Turns out that: kernel_initializer=tf.keras.initializers.Orthogonal(gain)
            # Actually fucks everything up, fun times
            ConvLayer(shape=[3, 3, 1, 32], use_bias=False, kernel_initializer=init,
                      kernel_regularizer=reg),
            BatchNormalisationLayer(32, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32], use_bias=False, kernel_initializer=init,
                      kernel_regularizer=reg),
            BatchNormalisationLayer(32, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 64], use_bias=False, strides=[1, 2, 2, 1],
                      kernel_initializer=init,
                      kernel_regularizer=reg),
            BatchNormalisationLayer(64, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64], use_bias=False, kernel_initializer=init,
                      kernel_regularizer=reg),
            BatchNormalisationLayer(64, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 128], use_bias=False, strides=[1, 2, 2, 1],
                      kernel_initializer=init,
                      kernel_regularizer=reg),
            BatchNormalisationLayer(128, affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 128, 128], use_bias=False, kernel_initializer=init,
                      kernel_regularizer=reg),
            BatchNormalisationLayer(128, affine=False),
            ReLU(),
            DropoutLayer(0.1),  # They use 0.3 when lr = 10
            ConvLayer(shape=[8, 8, 128, 128], use_bias=False, padding="VALID",
                      kernel_initializer=init,
                      kernel_regularizer=reg),
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
