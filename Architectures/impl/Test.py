from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, ReLU, MaxPoolingLayer, DropoutLayer, Flatten, FullyConnectedLayer, \
    BatchNormalisationLayer


class Test(IArchitecture):
    def __init__(self, num_classes=100):
        network = [
            ConvLayer(shape=[3, 3, 3, 32], padding='VALID'),
            BatchNormalisationLayer(32),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),
            ConvLayer(shape=[3, 3, 32, 64], padding='VALID'),
            BatchNormalisationLayer(64),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),

            # 16:20
            Flatten(),
            FullyConnectedLayer(shape=[3136, 512]),  # 4096
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, num_classes])
        ]

        super().__init__(network)
