from Architectures.architectures import IArchitecture
from Layers.impl.core import *
import Layers.impl.contrib as contrib
from math import floor


class DenseNet(IArchitecture):
    def transition_layer(self, name, in_channels, reduction=0.5):
        """
        Change feature-map sizes via convolution and pooling

        :param name: Scope name
        :param in_channels: Number of input channels
        """
        out_channels = math.floor(in_channels * reduction)
        with tf.variable_scope(name):

            if self.args.build_method == "standard":
                network = [
                    BatchNormalisationLayer(),
                    ReLU(),
                    ConvLayer(shape=[1, 1, in_channels, out_channels], use_bias=False,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.weight_decay)),
                    ReLU(),
                    AveragePoolingLayer(pool_size=(2, 2))
                ]
            elif self.args.build_method == "non-standard":
                network = [
                    BatchNormalisationLayer(),
                    ReLU(),
                    ConvLayer(shape=[1, 1, in_channels, out_channels], use_bias=False,
                              build_method=Weights.impl.sandbox, ranks=[1, self.args.ranks[0], self.args.ranks[1]],
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.weight_decay)),
                    ReLU(),
                    AveragePoolingLayer(pool_size=(2, 2))
                ]
            else:
                raise Exception("Unknown build method")

        return network

    def __init__(self, args, ds_args):
        """
        Instantiate the DenseNet architecture

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """
        N = int((args.depth - 4) / 3)
        # if bottleneck, N = N / 2
        growth_rate = 12
        reduction = 0.5

        self.args = args
        self.ds_args = ds_args

        if args.build_method == "standard":
            build_method = Weights.impl.core
        elif args.build_method == "sandbox":
            build_method = Weights.impl.sandbox
        else:
            raise Exception("Unknown build method")

        if args.dataset_name == 'CIFAR10' or args.dataset_name == 'CIFAR100':
            network = [
                # Initial convolution layer
                ConvLayer(shape=(3, 3, ds_args.num_channels, 32), use_bias=False),
                contrib.DenseBlock("DenseBlock1", in_channels=32, num_layers=N, growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer1", in_channels=176),

                contrib.DenseBlock("DenseBlock2", in_channels=88, num_layers=N, growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer2", in_channels=232),

                contrib.DenseBlock("DenseBlock3", in_channels=116, num_layers=N, growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer3", in_channels=260),

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

            # Pre-calculate all the input channel dims
            in_channels11 = 32 + (stages[0] * growth_rate)
            in_channels12 = math.floor(in_channels11 * reduction)

            in_channels21 = in_channels12 + (stages[1] * growth_rate)
            in_channels22 = math.floor(in_channels21 * reduction)

            in_channels31 = in_channels22 + (stages[2] * growth_rate)
            in_channels32 = math.floor(in_channels31 * reduction)

            in_channels41 = in_channels32 + (stages[3] * growth_rate)
            in_channels42 = math.floor(in_channels41 * reduction)

            network = [
                # Initial convolution layer
                ConvLayer(shape=(3, 3, ds_args.num_channels, 32), use_bias=False),
                BatchNormalisationLayer(),
                ReLU(),
                MaxPoolingLayer(pool_size=(2, 2)),
                # NOTE: These hardcoded parameters only work for depth = 169

                # Dense - Block 1 and transition(56x56)
                contrib.DenseBlock("DenseBlock1", in_channels=32, num_layers=stages[0], growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer1", in_channels=in_channels11),

                # Dense-Block 2 and transition (28x28)
                contrib.DenseBlock("DenseBlock2", in_channels=in_channels12, num_layers=stages[1],
                                   growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer2", in_channels=in_channels21),

                # Dense-Block 3 and transition (14x14)
                contrib.DenseBlock("DenseBlock3", in_channels=in_channels22, num_layers=stages[2],
                                   growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer3", in_channels=in_channels31),

                # Dense-Block 4 and transition (7x7)
                contrib.DenseBlock("DenseBlock4", in_channels=in_channels32, num_layers=stages[3],
                                   growth_rate=growth_rate,
                                   build_method=build_method,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
                *self.transition_layer("TransitionLayer4", in_channels=in_channels41),

                BatchNormalisationLayer(),
                ReLU(),
                GlobalAveragePooling(keep_dims=False),
                FullyConnectedLayer(shape=(in_channels42, ds_args.num_classes))
            ]

        super().__init__(network)
