from Architectures.architectures import IArchitecture
from Layers.impl.core import ConvLayer, AveragePoolingLayer, Flatten, FullyConnectedLayer

# These hyperparameters control the compression
# of the convolutional and fully connected weights
conv_ranks = {
    0: [6, 8, 16],
    1: [6, 16, 32],
    2: [6, 32, 64],
    3: [6, 64, 64],
    4: [6, 64, 128],
    5: [6, 128, 128],
    6: [6, 128, 256],

    7: [6, 256, 256],
    8: [6, 256, 256],
    9: [6, 256, 256],
    10: [6, 256, 256],
    11: [6, 256, 256],

    12: [6, 256, 512],
    13: [6, 512, 512]
}
fc_ranks = {
    16: [512, 256]
}


class MobileNetV1(IArchitecture):
    """ This is the original MobileNet architecture for ImageNet
        Of course, the convolutional layers are replaced with depthwise separable layers """
    def __init__(self, num_classes):
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
            AveragePoolingLayer(pool_size=[7, 7]),

            # Finally a fully connected layer
            Flatten(),
            FullyConnectedLayer(shape=[1024, num_classes])
        ]

        super().__init__(network)
