""" Defines some common architectures under the framework """
from Layers.core import *
import Layers.axel


class Architecture:
    def __init__(self, network):
        # Array of sequential layers
        self._network = network

    def num_parameters(self):
        """ Get the number of parameters for an entire architecture """
        n = 0
        for layer in self._network:
            n += layer.num_parameters()

        return n

    def num_layers(self):
        return len(self._network)

    def get_layer(self, layer_idx):
        return self._network[layer_idx]


class MobileNetV1(Architecture):
    """ This is the original MobileNet architecture for ImageNet """
    def __init__(self):
        num_classes = 10
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32], strides=[1, 2, 2, 1]),  # 224 x 224 x 3
            ConvLayer(shape=[3, 3, 32, 64]),  # 112 x 112 x 32
            ConvLayer(shape=[3, 3, 64, 128], strides=[1, 2, 2, 1]),  # 112 x 112 x 64
            ConvLayer(shape=[3, 3, 128, 128]),  # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 128, 256], strides=[1, 2, 2, 1]),  # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 256, 256]),  # 28 x 28 x 256
            ConvLayer(shape=[3, 3, 256, 512], strides=[1, 2, 2, 1]),  # 28 x 28 x 256

            # Repeated 5x
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512

            # Confused about stride here (from paper they are both = 2)
            ConvLayer(shape=[3, 3, 512, 1024], strides=[1, 2, 2, 1]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 1024, 1024]),  # 7 x 7 x 512

            # Global average pooling
            AveragePoolingLayer(shape=[1, 7, 7, 1]),

            # Finally a fully connected layer
            Flatten(),
            FullyConnectedLayer(shape=[1024, num_classes])
        ]

        super().__init__(network)


class AlexNet(Architecture):
    def __init__(self):
        network = [
            # TODO: Finish
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 96]),  # 227x227x3
            ConvLayer(shape=[3, 3, 96, 256]),
            ConvLayer(shape=[3, 3, 256, 384], strides=[1, 2, 2, 1]),
            ConvLayer(shape=[3, 3, 384, 384]),
            ConvLayer(shape=[3, 3, 384, 256], strides=[1, 2, 2, 1]),
            ConvLayer(shape=[3, 3, 256, 256]),
            ConvLayer(shape=[3, 3, 256, 512], strides=[1, 2, 2, 1]),

        ]

        super().__init__(network)


class CIFAR100Example(Architecture):
    # See: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
    def __init__(self):
        network = [
            ConvLayer(shape=[3, 3, 3, 32]),
            # BatchNormalisationLayer(32),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32]),
            # BatchNormalisationLayer(32),
            ReLU(),
            MaxPoolingLayer(shape=[1, 2, 2, 1]),
            DropoutLayer(0.25),

            ConvLayer(shape=[3, 3, 32, 64]),
            # BatchNormalisationLayer(64),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64]),
            # BatchNormalisationLayer(64),
            ReLU(),
            MaxPoolingLayer(shape=[1, 2, 2, 1]),
            DropoutLayer(0.25),

            Flatten(),
            FullyConnectedLayer(shape=[4096, 512]),
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, 100])
        ]

        super().__init__(network)


class KeyNet(Architecture):
    # Axel's Key.Net
    def __init__(self):
        network = [
            Layers.axel.GaussianSmoothLayer(),

            # Makes a parallel split
            Layers.axel.PyramidScaleSplitLayer(pyramid_levels=3, scaling_factor=1.2),
            Layers.axel.HandcraftedFeaturesLayer(),
            Layers.axel.ConvLayer(shape=(5, 5, 9, 8), strides=[1, 1, 1, 1], use_bias=False),
            Layers.axel.ConvLayer(shape=(5, 5, 8, 8), strides=[1, 1, 1, 1], use_bias=False),
            Layers.axel.ConvLayer(shape=(5, 5, 8, 8), strides=[1, 1, 1, 1], use_bias=False),
            # Combines the parallel split
            Layers.axel.PyramidScaleCombineLayer(),

            # Back to standard layers
            BatchNormalisationLayer(24, affine=False),
            ConvLayer(shape=[5, 5, 24, 1])
        ]

        super().__init__(network)

        MSIP_sizes = [8, 16, 24, 32, 40]
        self.create_kernels(MSIP_sizes, "KeyNet")

    def create_kernels(self, MSIP_sizes, name_scope):
        # create_kernels
        self.kernels = {}

        # Grid Indexes for MSIP
        for ksize in MSIP_sizes:
            from KeyNet.model.keynet_architecture import ones_multiple_channels, grid_indexes, linear_upsample_weights
            ones_kernel = ones_multiple_channels(ksize, 1)
            indexes_kernel = grid_indexes(ksize)
            upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)

            self.ones_kernel = tf.constant(ones_kernel, name=name_scope +'_Ones_kernel_'+str(ksize))
            self.kernels['ones_kernel_'+str(ksize)] = self.ones_kernel

            self.upsample_filter_np = tf.constant(upsample_filter_np, name=name_scope+'_upsample_filter_np_'+str(ksize))
            self.kernels['upsample_filter_np_'+str(ksize)] = self.upsample_filter_np

            self.indexes_kernel = tf.constant(indexes_kernel, name=name_scope +'_indexes_kernel_'+str(ksize))
            self.kernels['indexes_kernel_'+str(ksize)] = self.indexes_kernel

    def get_kernels(self):
        return self.kernels


class HardNet(Architecture):
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
        #x = tf.expand_dims(x, axis=2)
        x_norm = tf.math.l2_normalize(x, axis=1)
        return x_norm
