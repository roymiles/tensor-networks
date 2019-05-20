from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer


class AlexNet(IArchitecture):
    def __init__(self):
        network = [
            # TODO: Finish
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 96]),  # 227x227x3
            ConvLayer(shape=[3, 3, 96, 256]),
            ConvLayer(shape=[3, 3, 256, 384], strides=[1, 2, 2, 1]),
            ConvLayer(shape=[3, 3, 384, 384]),
            ConvLayer(shape=[3, 3, 384, 256], strides=[1, 2, 2, 1]),
            ConvLayer(shape=[3, 3, 256, 256]),
            ConvLayer(shape=[3, 3, 256, 512], strides=[1, 2, 2, 1]),

        ]

        super().__init__(network)