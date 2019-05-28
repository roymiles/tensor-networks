from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, ReLU, MaxPoolingLayer, DropoutLayer, Flatten, FullyConnectedLayer, \
    BatchNormalisationLayer


# These hyperparameters control the compression
# of the convolutional and fully connected weights
conv_ranks = {
    0: [6, 8, 16],
    2: [6, 16, 16],
    6: [6, 16, 32],
    8: [6, 32, 32]
}
fc_ranks = {
    13: [128, 128],
    16: [128, 64]
}


class CIFARExample(IArchitecture):
    # See: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
    def __init__(self, num_classes=100):
        # Change num_classes 10/100 depending on cifar10 or cifar100
        network = [
            # NOTE: Don't add batch normalisation layers, this messes up on 12 epochs for cifar10

            # 0:5
            ConvLayer(shape=[3, 3, 3, 32]),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32]),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),
            DropoutLayer(0.25),

            # 6:11
            ConvLayer(shape=[3, 3, 32, 64]),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64]),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),
            DropoutLayer(0.25),

            # 12:16
            Flatten(),
            FullyConnectedLayer(shape=[4096, 512]),
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, num_classes])
        ]

        super().__init__(network)
