import tensorflow as tf
import json
import os
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
from Architectures.impl.CIFARExample import CIFARExample


def load_config(name):
    """ Load training configuration """
    with open(f"{os.getcwd()}/config/{name}") as json_file:
        return json.load(json_file)


def get_architecture(name):
    """ Get the architecture from the name in the config """
    if name == "mobilenetv2":
        return MobileNetV2
    elif name == "mobilenetv1":
        return MobileNetV1
    elif name == "cifar_example":
        return CIFARExample
    else:
        raise Exception("Unknown architecture")


def get_optimizer(name):
    """ Get the optimizer from the name """
    if name == "adam":
        return tf.train.AdamOptimizer
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer
    else:
        raise Exception("Unknown optimizer")