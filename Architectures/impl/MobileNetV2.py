from Architectures.architectures import IArchitecture
from Layers.impl.core import *
from Layers.impl.contrib import MobileNetV2BottleNeck


class MobileNetV2(IArchitecture):
    """ See: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c """

    def __init__(self, args, ds_args):
        """
        Initialise MobileNetV1 architecture

        :param args: Model training parameters
        :param ds_args: Dataset parameters
        """

        if args.build_method == "standard":
            build_method = Weights.impl.core
            ranks = None
        elif args.build_method == "sandbox":
            build_method = Weights.impl.sandbox
            ranks = args.ranks

        network = [
            # in: 224 x 224 x 3
            ConvLayer(shape=[3, 3, ds_args.num_channels, 32], strides=(2, 2), use_bias=False),
            BatchNormalisationLayer(),
            ReLU(),

            MobileNetV2BottleNeck(in_channels=32, expansion=1, filters=16, strides=(1, 1), build_method=build_method, ranks=ranks),

            MobileNetV2BottleNeck(in_channels=16, expansion=6, filters=24, strides=(2, 2), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=24, expansion=6, filters=24, strides=(1, 1), build_method=build_method, ranks=ranks),

            MobileNetV2BottleNeck(in_channels=24, expansion=6, filters=32, strides=(2, 2), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=32, expansion=6, filters=32, strides=(1, 1), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=32, expansion=6, filters=32, strides=(1, 1), build_method=build_method, ranks=ranks),

            MobileNetV2BottleNeck(in_channels=32, expansion=6, filters=64, strides=(2, 2), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=64, expansion=6, filters=64, strides=(1, 1), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=64, expansion=6, filters=64, strides=(1, 1), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=64, expansion=6, filters=64, strides=(1, 1), build_method=build_method, ranks=ranks),
            DropoutLayer(rate=0.25),

            MobileNetV2BottleNeck(in_channels=64, expansion=6, filters=96, strides=(1, 1), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=96, expansion=6, filters=96, strides=(1, 1), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=96, expansion=6, filters=96, strides=(1, 1), build_method=build_method, ranks=ranks),
            DropoutLayer(rate=0.25),

            MobileNetV2BottleNeck(in_channels=96, expansion=6, filters=160, strides=(2, 2), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=160, expansion=6, filters=160, strides=(1, 1), build_method=build_method, ranks=ranks),
            MobileNetV2BottleNeck(in_channels=160, expansion=6, filters=160, strides=(1, 1), build_method=build_method, ranks=ranks),
            DropoutLayer(rate=0.25),

            MobileNetV2BottleNeck(in_channels=160, expansion=1, filters=320, strides=(1, 1), build_method=build_method, ranks=ranks),
            DropoutLayer(rate=0.25),

            ConvLayer(shape=[1, 1, 320, 1280], strides=(1, 1), use_bias=False),
            BatchNormalisationLayer(),
            ReLU(),

            # Classification part
            GlobalAveragePooling(keep_dims=False),
            FullyConnectedLayer(shape=[1280, ds_args.num_classes], use_bias=True),
        ]

        super().__init__(network)
