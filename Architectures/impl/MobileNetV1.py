from Architectures.architectures import IArchitecture
from Layers.impl.core import *
from Layers.impl.contrib import PartitionedDepthwiseSeparableLayer
import Weights.impl.sandbox


class MobileNetV1(IArchitecture):
    """ This is the original MobileNet architecture for ImageNet
        Of course, the convolutional layers are replaced with depthwise separable layers """

    def DepthSepConv(self, shape, stride, depth, depth_multiplier=1, partitions=None):
        """
        Depthwise convolution followed by a pointwise convolution (with BN and ReLU in-between)

        :param shape:
        :param stride:
        :param depth:
        :param depth_multiplier:
        :param partitions: [dw, std, fact, std]
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
                BatchNormalisationLayer(),
                ReLU(),
                # Pointwise
                ConvLayer(shape=[1, 1, c * depth_multiplier, depth], use_bias=False,
                          kernel_initializer=tf.keras.initializers.he_normal(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                # PointwiseDot(shape=[c * depth_multiplier, 128, 128, depth]),
                # Not managed to integrate moving average decay
                BatchNormalisationLayer(),
                ReLU()
            ]
        elif self._method == "factored-pw-kernel":
            sequential = [
                DepthwiseConvLayer(shape=[w, h, c, depth_multiplier], strides=(stride, stride), use_bias=False,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                BatchNormalisationLayer(),
                ReLU(),
                # Using core factors for the pointwise kernel
                ConvLayer(shape=[1, 1, c * depth_multiplier, depth], use_bias=False,
                          build_method=Weights.impl.sandbox, ranks=[1, self._ranks[0], self._ranks[1]],
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
                # PointwiseDot(shape=[c * depth_multiplier, 128, 128, depth]),
                # Not managed to integrate moving average decay
                BatchNormalisationLayer(),
                ReLU()
            ]
        elif self._method == "custom-bottleneck":
            # Custom bottleneck
            if partitions:
                p = partitions
            else:
                p = self._partitions

            sequential = [
                PartitionedDepthwiseSeparableLayer(shape=[w, h, c, depth], use_bias=False, strides=(stride, stride),
                                                   partitions=p,
                                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay),
                                                   # partitions=self._partitions,
                                                   ranks=[1, self._ranks[0], self._ranks[1]]),
                BatchNormalisationLayer(),
                HSwish()
            ]
        else:
            raise Exception("Unknown method for MobileNetV1 architecture")

        return sequential

    def __init__(self, args, ds_args):
        """
        Initialise MobileNetV1 architecture

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """
        self._weight_decay = args.weight_decay
        self._method = args.method
        self._ranks = args.ranks
        self._partitions = args.partitions
        network = [
            ConvLayer(shape=[3, 3, ds_args.num_channels, 32], strides=(2, 2),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
            BatchNormalisationLayer(),
            HSwish(),

            # Higher partition = less compression (% standard)
            *self.DepthSepConv(shape=[3, 3, 32], stride=1, depth=32, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 64], stride=1, depth=32, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 96], stride=1, depth=32, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 128], stride=1, depth=32, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 156], stride=1, depth=32, partitions=[0, 0]),

            *self.DepthSepConv(shape=[3, 3, 48], stride=2, depth=64, partitions=[0, 0]),


            *self.DepthSepConv(shape=[3, 3, 64], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 80], stride=2, depth=96, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 96], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 112], stride=2, depth=124, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 124], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 140], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 156], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 172], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 188], stride=1, depth=16, partitions=[0, 0]),
            *self.DepthSepConv(shape=[3, 3, 204], stride=2, depth=256, partitions=[0, 0]),
            # *self.DepthSepConv(shape=[3, 3, 2048], stride=1, depth=512),

            #*self.DepthSepConv(shape=[3, 3, 32], stride=1, depth=64),
            #*self.DepthSepConv(shape=[3, 3, 64], stride=2, depth=128),
            #*self.DepthSepConv(shape=[3, 3, 128], stride=1, depth=128),
            #*self.DepthSepConv(shape=[3, 3, 128], stride=2, depth=256),
            #*self.DepthSepConv(shape=[3, 3, 256], stride=1, depth=256),
            #*self.DepthSepConv(shape=[3, 3, 256], stride=2, depth=512),
            #*self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            #*self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            #*self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            #*self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            #*self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            #*self.DepthSepConv(shape=[3, 3, 512], stride=2, depth=1024),
            #*self.DepthSepConv(shape=[3, 3, 1024], stride=1, depth=1024),

            # ConvLayer(shape=[1, 1, 1024, 1024], use_bias=False),
            GlobalAveragePooling(keep_dims=False),
            Flatten(),
            FullyConnectedLayer(shape=[256, ds_args.num_classes],
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay),
                                use_bias=True)
        ]

        super().__init__(network)
