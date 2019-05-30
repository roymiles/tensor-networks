from Architectures.architectures import IArchitecture
from Layers.impl.core import *

# These hyperparameters control the compression
# of the convolutional and fully connected weights
conv_ranks = {
}
fc_ranks = {
}


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

    def __init__(self, num_classes):
        network = [
            # Pre
            ConvLayer(shape=[3, 3,  3, 3], strides=[2, 2], use_bias=False),
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
