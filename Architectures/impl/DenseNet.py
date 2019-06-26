from Architectures.architectures import IArchitecture
from Layers.impl.core import *
import Layers.impl.contrib as contrib
from math import floor


class DenseNet(IArchitecture):
    def transition_layer(self, name, in_channel):
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
                ConvLayer(shape=[1, 1, in_channel, out_channels], use_bias=False,
                          build_method=Weights.impl.sandbox, ranks=[1, self.args.ranks[0], self.args.ranks[1]],
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.weight_decay)),
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

        self.args = args
        self.ds_args = ds_args

        if args.dataset_name == 'CIFAR10' or args.dataset_name == 'CIFAR100':
            network = [
                # Initial convolution layer
                ConvLayer(shape=(3, 3, ds_args.num_channels, 32), use_bias=False),
                contrib.DenseBlock("DenseBlock1", in_channels=32, N=N, growth_rate=growth_rate,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                # Hard coded in channels makes everything much easier, else DenseBlock and TransitionLayer
                # would need to be merged into a single class layer, which is ugly.
                *self.transition_layer("TransitionLayer1", in_channel=176),

                contrib.DenseBlock("DenseBlock2", in_channels=88, N=N, growth_rate=growth_rate,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer2", in_channel=232),

                contrib.DenseBlock("DenseBlock3", in_channels=116, N=N, growth_rate=growth_rate,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer3", in_channel=260),

                BatchNormalisationLayer(),
                ReLU(),
                GlobalAveragePooling(keep_dims=False),
                FullyConnectedLayer(shape=(130, ds_args.num_classes))
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
                # Initial convolution layer
                ConvLayer(shape=(3, 3, ds_args.num_channels, 32), use_bias=False),
                BatchNormalisationLayer(),
                ReLU(),
                MaxPoolingLayer(pool_size=(2, 2)),

                # Dense - Block 1 and transition(56x56)
                contrib.DenseBlock("DenseBlock1", in_channels=1, growth_rate=stages[0],
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer1", in_channels=1),

                # Dense-Block 2 and transition (28x28)
                contrib.DenseBlock("DenseBlock2", in_channels=1, growth_rate=stages[1],
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer2", in_channels=1),

                # Dense-Block 3 and transition (14x14)
                contrib.DenseBlock("DenseBlock3", in_channels=1, growth_rate=stages[2],
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer3", in_channels=1),

                # Dense-Block 4 and transition (7x7)
                contrib.DenseBlock("DenseBlock4", in_channels=1, growth_rate=stages[3],
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer4", in_channels=1),

                BatchNormalisationLayer(),
                ReLU(),
                GlobalAveragePooling(keep_dims=False),
                FullyConnectedLayer(shape=(130, ds_args.num_classes))
            ]

        super().__init__(network)