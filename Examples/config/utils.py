import tensorflow as tf
import json
import os
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
from Architectures.impl.CIFARExample import CIFARExample
from Architectures.impl.MNISTExample import MNISTExample


def load_config(name):
    """ Load training configuration """
    with open(f"{os.getcwd()}/config/{name}") as json_file:
        return json.load(json_file)


def get_architecture(tc):
    """
        Get the architecture based on the arguments

        :param tc: Dictionary of arguments, including name of architecture, (num_classes?) etc
        :return: The architecture instance
    """

    name = tc['name']
    if name == "mobilenetv2":
        return MobileNetV2(num_classes=tc['num_classes'], channels=tc['num_channels'])
    elif name == "mobilenetv1":
        return MobileNetV1(num_classes=tc['num_classes'], channels=tc['num_channels'])
    elif name == "cifar_example":
        return CIFARExample()
    elif name == "mnist_example":
        return MNISTExample()
    else:
        raise Exception("Unknown architecture")


def get_optimizer(tc):
    """
        Get the optimizer based on the arguments

        :param tc: Dictionary of arguments, including name of optimizer, learning rate etc
        :return: Optimizer, just call .minimize on this object
    """
    name = tc['name']
    if name == "adam":
        return tf.train.AdamOptimizer(learning_rate=tc['learning_rate'])
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate=tc['learning_rate'])
    else:
        raise Exception("Unknown optimizer")