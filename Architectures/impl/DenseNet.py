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
        out_channels = math.floor(in_channels * self.args.reduction)
        with tf.variable_scope(name):

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

        return network

    def __init__(self, args, ds_args):
        """
        Instantiate the DenseNet architecture

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """
        # if bottleneck, N = N / 2

        self.args = args
        self.ds_args = ds_args

        if args.build_method == "standard":
            build_method = Weights.impl.core
        elif args.build_method == "sandbox":
            build_method = Weights.impl.sandbox
        else:
            raise Exception("Unknown build method")

        if args.depth == 121:
            stages = [6, 12, 24, 16]
        elif args.depth == 169:
            stages = [6, 12, 32, 32]
        elif args.depth == 201:
            stages = [6, 12, 48, 32]
        elif args.depth == 161:
            stages = [6, 12, 36, 24]
        else:
            stages = args.stages

        # Pre-calculate all the input channel dims
        in_channels0 = 64

        in_channels11 = in_channels0 + (stages[0] * args.growth_rate)
        in_channels12 = math.floor(in_channels11 * args.reduction)

        in_channels21 = in_channels12 + (stages[1] * args.growth_rate)
        in_channels22 = math.floor(in_channels21 * args.reduction)

        in_channels31 = in_channels22 + (stages[2] * args.growth_rate)
        in_channels32 = math.floor(in_channels31 * args.reduction)

        in_channels41 = in_channels32 + (stages[3] * args.growth_rate)
        in_channels42 = math.floor(in_channels41 * args.reduction)

        network = [
            # Initial convolution layer
            ConvLayer(shape=(7, 7, ds_args.num_channels, 64), use_bias=False, padding="SAME"),
            BatchNormalisationLayer(),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),

            # Dense - Block 1 and transition(56x56)
            DenseBlock("DenseBlock1", in_channels=in_channels0, num_layers=stages[0],
                       dropout_rate=args.dropout_rate,
                       growth_rate=args.growth_rate,
                       build_method=build_method,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
            *self.transition_layer("TransitionLayer1", in_channels=in_channels11),

            # Dense-Block 2 and transition (28x28)
            DenseBlock("DenseBlock2", in_channels=in_channels12, num_layers=stages[1],
                       dropout_rate=args.dropout_rate,
                       growth_rate=args.growth_rate,
                       build_method=build_method,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
            *self.transition_layer("TransitionLayer2", in_channels=in_channels21),

            # Dense-Block 3 and transition (14x14)
            DenseBlock("DenseBlock3", in_channels=in_channels22, num_layers=stages[2],
                       dropout_rate=args.dropout_rate,
                       growth_rate=args.growth_rate,
                       build_method=build_method,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
            *self.transition_layer("TransitionLayer3", in_channels=in_channels31),

            # Dense-Block 4 and transition (7x7)
            DenseBlock("DenseBlock4", in_channels=in_channels32, num_layers=stages[3],
                       dropout_rate=args.dropout_rate,
                       growth_rate=args.growth_rate,
                       build_method=build_method,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)),
            *self.transition_layer("TransitionLayer4", in_channels=in_channels41),

            BatchNormalisationLayer(),
            ReLU(),

            # Max or Average
            GlobalAveragePooling(keep_dims=False),

            # Top
            FullyConnectedLayer(shape=(in_channels42, ds_args.num_classes))
        ]

        super().__init__(network)
