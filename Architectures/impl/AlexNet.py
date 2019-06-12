from Architectures.architectures import IArchitecture
from Layers.impl.core import *


class AlexNet(IArchitecture):
    def __init__(self, num_classes):
        network = [
            ConvLayer(shape=[3, 3, 3, 96]),
            MaxPoolingLayer(pool_size=(3, 3)),
            ReLU(),

            ConvLayer(shape=[3, 3, 96, 256]),
            ReLU(),
            MaxPoolingLayer(pool_size=(3, 3)),

            ConvLayer(shape=[3, 3, 256, 384], strides=(2, 2)),
            ReLU(),
            ConvLayer(shape=[3, 3, 384, 384]),
            ReLU(),
            ConvLayer(shape=[3, 3, 384, 256], strides=(2, 2)),
            ReLU(),

            MaxPoolingLayer(pool_size=(3, 3)),

            Flatten(),
            Dense(shape=(256, 4096)),
            ReLU(),
            Dense(shape=(4096, 4096)),
            ReLU(),
            Dense(shape=(4096, num_classes)),
        ]

        super().__init__(network)
