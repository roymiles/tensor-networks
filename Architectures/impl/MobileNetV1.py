from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, AveragePoolingLayer, Flatten, FullyConnectedLayer


class MobileNetV1(IArchitecture):
    """ This is the original MobileNet architecture for ImageNet """
    def __init__(self):
        num_classes = 10
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32], strides=[1, 2, 2, 1]),  # 224 x 224 x 3
            ConvLayer(shape=[3, 3, 32, 64]),  # 112 x 112 x 32
            ConvLayer(shape=[3, 3, 64, 128], strides=[1, 2, 2, 1]),  # 112 x 112 x 64
            ConvLayer(shape=[3, 3, 128, 128]),  # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 128, 256], strides=[1, 2, 2, 1]),  # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 256, 256]),  # 28 x 28 x 256
            ConvLayer(shape=[3, 3, 256, 512], strides=[1, 2, 2, 1]),  # 28 x 28 x 256

            # Repeated 5x
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512

            # Confused about stride here (from paper they are both = 2)
            ConvLayer(shape=[3, 3, 512, 1024], strides=[1, 2, 2, 1]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 1024, 1024]),  # 7 x 7 x 512

            # Global average pooling
            AveragePoolingLayer(shape=[1, 7, 7, 1]),

            # Finally a fully connected layer
            Flatten(),
            FullyConnectedLayer(shape=[1024, num_classes])
        ]

        super().__init__(network)
