from Architectures.architectures import IArchitecture
from Layers.impl.core import Flatten, Dense, DropoutLayer, ReLU


class MNISTExample(IArchitecture):
    # See: https://www.tensorflow.org/tutorials
    def __init__(self):
        network = [
            Flatten(),
            Dense(shape=(784, 512)),
            ReLU(),
            DropoutLayer(0.2),
            Dense(shape=(512, 10))
        ]

        super().__init__(network)
