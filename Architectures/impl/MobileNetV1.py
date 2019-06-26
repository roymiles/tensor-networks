from Architectures.architectures import IArchitecture
from Layers.impl.core import *
import Weights.impl.sandbox


class MobileNetV1(IArchitecture):
    """ This is the original MobileNet architecture for ImageNet
        Of course, the convolutional layers are replaced with depthwise separable layers """

    def DepthSepConv(self, shape, stride, depth, depth_multiplier=1):
        """
        Depthwise convolution followed by a pointwise convolution (with BN and ReLU in-between)

        :param shape:
        :param stride:
        :param depth:
        :param depth_multiplier:
        :return:
        """

        # Depth is pretty much shape[3] (if included)
        w = shape[0]
        h = shape[1]
        c = shape[2]
        # depth_multiplier = 1

        # Paper says to not regularise the depthwise filters but lots of other implementations do...?
        if self._method == "standard":
            sequential = [
                DepthwiseConvLayer(shape=[w, h, c, depth_multiplier], strides=(stride, stride), use_bias=False,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                BatchNormalisationLayer(self._switch_list),
                ReLU(),
                # Pointwise
                ConvLayer(shape=[1, 1, c * depth_multiplier, depth], use_bias=False,
                          kernel_initializer=tf.keras.initializers.he_normal(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                # PointwiseDot(shape=[c * depth_multiplier, 128, 128, depth]),
                # Not managed to integrate moving average decay
                BatchNormalisationLayer(self._switch_list),
                ReLU()
            ]
        elif self._method == "factored-pw-kernel":
            sequential = [
                DepthwiseConvLayer(shape=[w, h, c, depth_multiplier], strides=(stride, stride), use_bias=False,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                BatchNormalisationLayer(self._switch_list),
                ReLU(),
                # Using core factors for the pointwise kernel
                ConvLayer(shape=[1, 1, c * depth_multiplier, depth], use_bias=False,
                          build_method=Weights.impl.sandbox, ranks=[1, self._ranks[0], self._ranks[1]],
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                # PointwiseDot(shape=[c * depth_multiplier, 128, 128, depth]),
                # Not managed to integrate moving average decay
                BatchNormalisationLayer(self._switch_list),
                ReLU()
            ]
        elif self._method == "custom-bottleneck":
            # Custom bottleneck
            sequential = [
                CustomBottleneck(shape=[w, h, c, depth], use_bias=False, strides=(stride, stride),
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay),
                                 partitions=self._partitions, ranks=[1, self._ranks[0], self._ranks[1]]),
                BatchNormalisationLayer(self._switch_list),
                ReLU()
            ]
        else:
            raise Exception("Unknown method for MobileNetV1 architecture")

        return sequential

    def __init__(self, args, ds_args):
        """

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """
        self._switch_list = args.switch_list
        self._weight_decay = args.weight_decay
        self._method = args.method
        self._ranks = args.ranks
        self._partitions = args.partitions
        network = [
            ConvLayer(shape=[3, 3, ds_args.num_channels, 32], strides=(2, 2),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
            BatchNormalisationLayer(self._switch_list),
            ReLU(),

            *self.DepthSepConv(shape=[3, 3, 32], stride=1, depth=64),
            *self.DepthSepConv(shape=[3, 3, 64], stride=2, depth=128),
            *self.DepthSepConv(shape=[3, 3, 128], stride=1, depth=128),
            *self.DepthSepConv(shape=[3, 3, 128], stride=2, depth=256),
            *self.DepthSepConv(shape=[3, 3, 256], stride=1, depth=256),
            *self.DepthSepConv(shape=[3, 3, 256], stride=2, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=2, depth=1024),
            *self.DepthSepConv(shape=[3, 3, 1024], stride=1, depth=1024),

            # ConvLayer(shape=[1, 1, 1024, 1024], use_bias=False),
            GlobalAveragePooling(keep_dims=False),
            Flatten(),
            FullyConnectedLayer(shape=[1024, ds_args.num_classes],
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay))
        ]

        super().__init__(network)
