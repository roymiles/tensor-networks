from Architectures.architectures import IArchitecture
from Layers.impl.core import *

# These hyperparameters control the compression
# of the convolutional and fully connected weights
conv_ranks = {
}
fc_ranks = {
}


"""
    THIS IMPLEMENTATION IS MESSY AND DEPRECATED, REMOVE SOON
"""

class MobileNetV2(IArchitecture):
    """ See: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenetv2.py"""

    @staticmethod
    def linear_bottleneck(in_channels, out_channels, stride, t=6):
        sequential = [
            ConvLayer(shape=[1, 1, in_channels, in_channels * t]),
            BatchNormalisationLayer(in_channels * t),
            ReLU6(),

            # DepthwiseConvLayer accepts multiplier as 3rd dimension
            DepthwiseConvLayer(shape=(3, 3, in_channels * t, 1), strides=[1, stride, stride, 1],
                               padding="SAME"),
            BatchNormalisationLayer(in_channels * t),
            ReLU6(),

            ConvLayer(shape=[1, 1, in_channels * t, out_channels]),
            BatchNormalisationLayer(out_channels)
        ]

        return sequential

    @staticmethod
    def make_stage(repeat, in_channels, out_channels, stride, t):
        layers = list()
        layers.append(MobileNetV2.linear_bottleneck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.extend(MobileNetV2.linear_bottleneck(out_channels, out_channels, 1, t))
            repeat -= 1

        return layers[0]

    @staticmethod
    def expanded_conv(stride, num_outputs):
        return 1

    def __init__(self, num_classes, depth_multiplier):
        self._num_classes = num_classes
        self._depth_multiplier = depth_multiplier

        network = [
            ConvLayer(shape=[3, 3, 3, 32], strides=[2, 2]),
            *MobileNetV2.expanded_conv(stride=2, num_outputs=24),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=24),
            *MobileNetV2.expanded_conv(stride=2, num_outputs=32),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=32),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=32),
            *MobileNetV2.expanded_conv(stride=2, num_outputs=64),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=64),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=64),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=64),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=96),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=96),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=96),
            *MobileNetV2.expanded_conv(stride=2, num_outputs=160),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=160),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=160),
            *MobileNetV2.expanded_conv(stride=1, num_outputs=320),
            ConvLayer(shape=[1, 1, 3, 1280]),

            BatchNormalisationLayer(32),
            ReLU6(),

            # Linear bottleneck stages
            *MobileNetV2.linear_bottleneck(32, 16, 1, 1),
            *MobileNetV2.make_stage(2, 16, 24, 2, 6),
            *MobileNetV2.make_stage(3, 24, 32, 2, 6),
            *MobileNetV2.make_stage(4, 32, 64, 2, 6),
            *MobileNetV2.make_stage(3, 64, 96, 1, 6),
            *MobileNetV2.make_stage(3, 96, 160, 1, 6),
            *MobileNetV2.linear_bottleneck(160, 320, 1, 6),

            # Final
            ConvLayer(shape=[1, 1, 320, 1280]),
            BatchNormalisationLayer(1280),
            ReLU6(),

            AveragePoolingLayer(pool_size=(4, 4)),
            ConvLayer(shape=[1, 1, 1280, num_classes])
        ]

        super().__init__(network)

    def print(self):
        print(self.get_network())
