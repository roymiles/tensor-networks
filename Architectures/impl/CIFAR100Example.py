from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, ReLU, MaxPoolingLayer, DropoutLayer, Flatten, FullyConnectedLayer


class CIFAR100Example(IArchitecture):
    # See: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
    def __init__(self):
        network = [
            ConvLayer(shape=[3, 3, 3, 32]),
            # BatchNormalisationLayer(32),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32]),
            # BatchNormalisationLayer(32),
            ReLU(),
            MaxPoolingLayer(shape=[1, 2, 2, 1]),
            DropoutLayer(0.25),

            ConvLayer(shape=[3, 3, 32, 64]),
            # BatchNormalisationLayer(64),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64]),
            # BatchNormalisationLayer(64),
            ReLU(),
            MaxPoolingLayer(shape=[1, 2, 2, 1]),
            DropoutLayer(0.25),

            Flatten(),
            FullyConnectedLayer(shape=[4096, 512]),
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, 100])
        ]

        super().__init__(network)
