import tensorflow as tf
import numpy as np
from layers import *
from tensor_network import TensorNetV1, LayerTypes


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
class Architecture:

    # Array of sequential layers
    network = []

    def num_parameters(self):
        """ Get the number of parameters for an entire architecture """
        n = 0
        for layer in self.network:
            n += layer.num_parameters()

        return n


class MobileNetV1(Architecture):

    def __init__(self):
        Architecture.network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32], strides=2),    # 224 x 224 x 3
            ConvLayer(shape=[3, 3, 32, 64], strides=1),   # 112 x 112 x 32
            ConvLayer(shape=[3, 3, 64, 128], strides=2),  # 112 x 112 x 64
            ConvLayer(shape=[3, 3, 128, 128], strides=1), # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 128, 256], strides=2), # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 256, 256], strides=1), # 28 x 28 x 256
            ConvLayer(shape=[3, 3, 256, 512], strides=2), # 28 x 28 x 256
            
            # Repeated 5x
            ConvLayer(shape=[3, 3, 512, 512], strides=1),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512], strides=1),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512], strides=1),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512], strides=1),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512], strides=1),  # 14 x 14 x 512
            
            # Confused about stride here
            ConvLayer(shape=[3, 3, 512, 1024], strides=2),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 1024, 1024], strides=1),  # 7 x 7 x 512

            # Gloval average pooling
            AveragePoolingLayer(shape=[7, 7]),

            # Finally a fully connected layer (1000 classes)
            FullyConnectedLayer(shape=[1024, 1000])
        ]
                                    
        


class AlexNet(Architecture):
    def __init__(self):
        # TODO: Update this
        """
        Architecture.network_config = [
            [[3, 3, 3, 96],            LayerTypes.CONV, 1],
            [[3, 3, 96, 256],          LayerTypes.CONV, 1],
            [[3, 3, 256, 384],         LayerTypes.CONV, 1],
            [[3, 3, 384, 384],         LayerTypes.CONV, 1],
            [[3, 3, 384, 256],         LayerTypes.CONV, 1],
            
            [[9216, 4096], LayerTypes.FC],
            [[4096, 4096], LayerTypes.FC],
            [[4096, 1000], LayerTypes.FC]
        ]
        """


conv_ranks = [5, 5, 5, 5]
fc_ranks = [3, 3, 3]

model_config = MobileNetV1()
print("Number of parameters before = {}".format(model.num_parameters()))

model = TensorNetV1(model_config=model_config,
                    conv_ranks=conv_ranks, 
                    fc_ranks=fc_ranks)

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(x_train, y_train, batch_size=32, epochs=5)

"""
x = tf.random.uniform(shape=(8, 32, 32, 3))  # Batch size of 8

e.get_info()
a = e.layer(x, layer_idx=0, name="first_layer")

print("Number of parameters = {}".format(e.num_parameters()))
"""

