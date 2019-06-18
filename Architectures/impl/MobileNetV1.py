from Architectures.architectures import IArchitecture
from Layers.impl.core import *
import Weights.impl.sandbox


class MobileNetV1(IArchitecture):
    """ This is the original MobileNet architecture for ImageNet
        Of course, the convolutional layers are replaced with depthwise separable layers """

    def DepthSepConv(self, shape, stride, depth, depth_multiplier=1):
        # Depth is pretty much shape[3] (if included)
        w = shape[0]
        h = shape[1]
        c = shape[2]
        # depth_multiplier = 1

        # By default, don't regularise depthwise filters
        sequential = [
            DepthwiseConvLayer(shape=[w, h, c, depth_multiplier], strides=(stride, stride), use_bias=False),
            BatchNormalisationLayer(self._switch_list),
            ReLU(),
            # Pointwise
            ConvLayer(shape=[1, 1, c * depth_multiplier, depth], use_bias=False,
                      build_method=Weights.impl.sandbox, ranks=[1, 96, 96],
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
            #PointwiseDot(shape=[c * depth_multiplier, 128, 128, depth]),
            # Not managed to integrate moving average decay
            BatchNormalisationLayer(self._switch_list),
            ReLU()
        ]

        return sequential

    def __init__(self, num_classes, channels, switch_list=[1.0], weight_decay=5e-4):
        self._switch_list = switch_list
        self._weight_decay = weight_decay
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, channels, 32], strides=(2, 2),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay)),
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
            FullyConnectedLayer(shape=[1024, num_classes],
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._weight_decay))
        ]

        super().__init__(network)
