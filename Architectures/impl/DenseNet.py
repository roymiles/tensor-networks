from Architectures.architectures import IArchitecture
from Layers.impl.core import *
from Layers.impl.contrib import DenseBlock
from math import floor

# NOTE: LOOK AT KERAS IMPLEMENTATION FOR THIS ARCHITECTURE


class DenseNet(IArchitecture):
    def transition_layer(self, name, in_channels):
        """
        Change feature-map sizes via convolution and pooling

        :param name: Scope name
        :param in_channels: Number of input channels
        """
        # keep_prob = 1 - self.args.dropout_rate
        # out_channels = math.floor(in_channels * self.args.reduction)
        with tf.variable_scope(name):

            """
            if self.args.build_method == "standard":
                network = [
                    BatchNormalisationLayer(),
                    ReLU(),
                    ConvLayer(shape=[1, 1, in_channels, out_channels], use_bias=False,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.weight_decay)),
                    DropoutLayer(rate=self.args.dropout_rate),
                    AveragePoolingLayer(pool_size=(2, 2))
                ]
            elif self.args.build_method == "sandbox":
                network = [
                    BatchNormalisationLayer(),
                    ReLU(),
                    ConvLayer(shape=[1, 1, in_channels, out_channels], use_bias=False,
                              # build_method=Weights.impl.sandbox, ranks=[1, in_channels//4, out_channels//4],
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.weight_decay)),
                    DropoutLayer(rate=self.args.dropout_rate),
                    ReLU(),
                    AveragePoolingLayer(pool_size=(2, 2))
                ]
            else:
                raise Exception("Unknown build method")
            """

            # From CondenseNet, reduce by args.reduction (aka 2)
            network = [AveragePoolingLayer(pool_size=(2, 2))]

        return network

    def __init__(self, args, ds_args):
        """
        Instantiate the DenseNet architecture

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """

        self.args = args
        self.ds_args = ds_args

        if args.build_method == "standard":
            build_method = Weights.impl.core
        elif args.build_method == "sandbox":
            build_method = Weights.impl.sandbox
        else:
            raise Exception("Unknown build method")

        """if args.depth == 121:
            stages = [6, 12, 24, 16]
        elif args.depth == 169:
            stages = [6, 12, 32, 32]
        elif args.depth == 201:
            stages = [6, 12, 48, 32]
        elif args.depth == 161:
            stages = [6, 12, 36, 24]
        else:"""

        stages = args.stages

        network = [
            # Initial convolution layer
            ConvLayer(shape=(7, 7, ds_args.num_channels, 64), use_bias=False, padding="SAME",
                      build_method=Weights.impl.core, ranks=None),
            BatchNormalisationLayer(),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2))
        ]

        in_channels = 64
        for i, (stage, growth_rate) in enumerate(zip(stages, args.growth_rates)):
            dense_block = DenseBlock(f"DenseBlock{i}", in_channels=in_channels, num_layers=stage,
                                     dropout_rate=args.dropout_rate,
                                     growth_rate=growth_rate,
                                     bottleneck=args.bottleneck,
                                     build_method=build_method,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay))

            in_channels = in_channels + (stage * growth_rate)
            transition = self.transition_layer(f"TransitionLayer{i}", in_channels=in_channels)
            # in_channels = math.floor(in_channels * args.reduction)

            # Add them to the sequential network
            network.extend([dense_block, *transition])

        network.extend([
            BatchNormalisationLayer(),
            ReLU(),

            # Max or Average
            GlobalAveragePooling(keep_dims=False),

            # Top
            FullyConnectedLayer(shape=(in_channels, ds_args.num_classes),
                                build_method=Weights.impl.sandbox, ranks=[256, 128])
        ])

        print(network)

        super().__init__(network)
