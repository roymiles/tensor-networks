from Architectures.architectures import IArchitecture
from Layers.impl.core import *
from math import floor


class DenseNet(IArchitecture):
    @staticmethod
    def transition_layer(name, in_channel):
        """
        Change feature-map sizes via convolution and pooling

        :param name: Scope name
        :param in_channel: Number of input channels
        """
        reduction = 0.5
        out_channels = math.floor(in_channel * reduction)
        with tf.variable_scope(name):
            network = [
                BatchNormalisationLayer(),
                ReLU(),
                ConvLayer(shape=[1, 1, in_channel, out_channels], use_bias=False),
                ReLU(),
                AveragePoolingLayer(pool_size=(2, 2))
            ]

        return network

    def __init__(self, args, ds_args):
        """
        Instantiate the DenseNet architecture

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """
        N = int((args.depth - 4) / 3)
        growth_rate = 12
        if args.dataset_name == 'CIFAR10' or args.dataset_name == 'CIFAR100':
            network = [
                ConvLayer(shape=(3, 3, ds_args.num_channels, 32), use_bias=False),
                DenseBlock("DenseBlock1", N, growth_rate),
                # Hard coded in channels makes everything much easier, else DenseBlock and TransitionLayer
                # would need to be merged into a single class layer, which is ugly.
                *self.transition_layer("TransitionLayer1", in_channel=176),

                DenseBlock("DenseBlock2", N, growth_rate),
                *self.transition_layer("TransitionLayer2", in_channel=232),

                DenseBlock("DenseBlock3", N, growth_rate),
                *self.transition_layer("TransitionLayer3", in_channel=260),

                BatchNormalisationLayer(),
                ReLU(),
                GlobalAveragePooling(keep_dims=False),
                FullyConnectedLayer(shape=(130, args.num_classes))
            ]
        elif args.dataset_name == 'ImageNet2012':
            if args.depth == 121:
                stages = [6, 12, 24, 16]
            elif args.depth == 169:
                stages = [6, 12, 32, 32]
            elif args.depth == 201:
                stages = [6, 12, 48, 32]
            elif args.depth == 161:
                stages = [6, 12, 36, 24]
            else:
                stages = [args.d1, args.d2, args.d3, args.d4]

            network = [
                ConvLayer(shape=(3, 3, ds_args.num_channels, 32), use_bias=False),
                BatchNormalisationLayer(),
                ReLU(),
                MaxPoolingLayer(pool_size=(2, 2)),

                # Dense - Block 1 and transition(56x56)
                DenseBlock("DenseBlock1", growth_rate=stages[0]),
                *self.transition_layer("TransitionLayer1", in_channels=1),

                # Dense-Block 2 and transition (28x28)
                DenseBlock("DenseBlock2", growth_rate=stages[1]),
                *self.transition_layer("TransitionLayer2", in_channels=1),

                # Dense-Block 3 and transition (14x14)
                DenseBlock("DenseBlock3", growth_rate=stages[2]),
                *self.transition_layer("TransitionLayer3", in_channels=1),

                # Dense-Block 4 and transition (7x7)
                DenseBlock("DenseBlock4", growth_rate=stages[3]),
                *self.transition_layer("TransitionLayer4", in_channels=1),

                BatchNormalisationLayer(),
                ReLU(),
                GlobalAveragePooling(keep_dims=False),
                FullyConnectedLayer(shape=(130, args.num_classes))
            ]

        super().__init__(network)