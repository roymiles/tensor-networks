from Architectures.architectures import IArchitecture
from Layers.impl.core import *

learning_rate = tf.placeholder(tf.float64, shape=[])
training_params = {
    "initial_learning_rate": 0.1,
    "learning_rate": learning_rate,
    "batch_size": 96,
    "optimizer": tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.9),
}


class MobileNetV2(IArchitecture):
    """ See: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c """

    def __init__(self, num_classes):
        network = [
            # Fudge dataset normalisation
            BatchNormalisationLayer(3),

            # in: 224 x 224 x 3
            ConvLayer(shape=[3, 3, 3, 32], strides=(2, 2)),
            BatchNormalisationLayer(num_features=32),
            DropoutLayer(rate=0.2),
            ReLU6(),

            MobileNetV2BottleNeck(k=32, t=1, c=16, strides=(1, 1)),

            MobileNetV2BottleNeck(k=16, t=6, c=24, strides=(2, 2)),
            MobileNetV2BottleNeck(k=24, t=6, c=24, strides=(1, 1)),

            MobileNetV2BottleNeck(k=24, t=6, c=32, strides=(2, 2)),
            MobileNetV2BottleNeck(k=32, t=6, c=32, strides=(1, 1)),
            MobileNetV2BottleNeck(k=32, t=6, c=32, strides=(1, 1)),

            MobileNetV2BottleNeck(k=32, t=6, c=64, strides=(2, 2)),
            MobileNetV2BottleNeck(k=64, t=6, c=64, strides=(1, 1)),
            MobileNetV2BottleNeck(k=64, t=6, c=64, strides=(1, 1)),
            MobileNetV2BottleNeck(k=64, t=6, c=64, strides=(1, 1)),

            MobileNetV2BottleNeck(k=64, t=6, c=96, strides=(1, 1)),
            MobileNetV2BottleNeck(k=96, t=6, c=96, strides=(1, 1)),
            MobileNetV2BottleNeck(k=96, t=6, c=96, strides=(1, 1)),

            MobileNetV2BottleNeck(k=96, t=6, c=160, strides=(2, 2)),
            MobileNetV2BottleNeck(k=160, t=6, c=160, strides=(1, 1)),
            MobileNetV2BottleNeck(k=160, t=6, c=160, strides=(1, 1)),

            MobileNetV2BottleNeck(k=160, t=1, c=320, strides=(1, 1)),

            ConvLayer(shape=[1, 1, 320, 1280], strides=(1, 1)),
            BatchNormalisationLayer(num_features=1280),
            ReLU6(),

            # Classification part
            GlobalAveragePooling(keep_dims=True),
            DropoutLayer(rate=0.2),
            ConvLayer(shape=[1, 1, 1280, num_classes]),
            GlobalAveragePooling(keep_dims=False)

            # FullyConnectedLayer(shape=[1280, num_classes])

            # GlobalAveragePooling(keep_dims=True),
            # ConvLayer(shape=[1, 1, 1280, num_classes]),

            # Remove spatial dims, so output is ? x 10
            # GlobalAveragePooling(keep_dims=False)
        ]

        super().__init__(network)
