from Architectures.architectures import IArchitecture
from Layers.impl.core import *

# These hyperparameters control the compression
# of the convolutional and fully connected weights
conv_ranks = {
}
fc_ranks = {
}


class TransitionLayer(IArchitecture):
    def __init__(self, k0):
        """
        Change feature-map sizes via convolution and pooling

        :param k0: Number of input channels
        """
        network = [
            BatchNormalisationLayer(k0),
            ReLU(),
            ConvLayer(shape=[1, 1, k0, 1]),
            AveragePoolingLayer(pool_size=(2, 2))
        ]

        super().__init__(network)


class DenseBlock(IArchitecture):
    def __init__(self, n, k0, k):
        """

        :param n: Number of 1x1 then 3x3 layers
        :param k0: Input channels
        :param k: Growth rate (is also the input channel dim k0)
        """

        c = k0
        print("Input dim {}".format(c))
        network = []
        for _ in range(n):
            print("c = {}, k = {}".format(c, k))
            _net = [
                BatchNormalisationLayer(c),
                ReLU(),
                ConvLayer(shape=[1, 1, c, k])
            ]

            network = network + _net

            # Growth rate
            c += k

        # Output channels
        self._c = c

        super().__init__(network)

    def get_output_dim(self):
        return self._c

