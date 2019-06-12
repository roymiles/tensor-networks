import tensorflow as tf
import json
import os
from collections import namedtuple
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
from Architectures.impl.CIFARExample import CIFARExample
from Architectures.impl.MNISTExample import MNISTExample


def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())


# Convert json file to object
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)


def load_config(name, as_obj=True):
    """ Load training configuration """
    with open(f"{os.getcwd()}/config/{name}") as json_file:
        if as_obj:
            # As object
            d = json_file.read()
            x = json2obj(d)
            return x
        else:
            # As dictionary
            return json.load(json_file)


def get_architecture(args):
    """
        Get the architecture based on the arguments

        :param args: Object containing arguments as members, including name of architecture, (num_classes?) etc
        :return: The architecture instance
    """

    print(args)
    name = args.architecture
    if name == "mobilenetv2":
        return MobileNetV2(num_classes=args.num_classes, channels=args.num_channels)
    elif name == "mobilenetv1":
        return MobileNetV1(num_classes=args.num_classes, channels=args.num_channels)
    elif name == "cifar_example":
        return CIFARExample()
    elif name == "mnist_example":
        return MNISTExample()
    else:
        raise Exception("Unknown architecture")


def get_optimizer(args):
    """
        Get the optimizer based on the arguments

        :param tc: Object containing arguments as members, including name of optimizer, learning rate etc
        :return: Optimizer, just call .minimize on this object
    """

    with tf.variable_scope("input"):
        learning_rate = tf.placeholder(tf.float64, shape=[])

    name = args.optimizer
    if name == "adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate), learning_rate
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate), learning_rate
    else:
        raise Exception("Unknown optimizer")
