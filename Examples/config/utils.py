import tensorflow as tf
import json
import os
from collections import namedtuple
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
from Architectures.impl.CIFARExample import CIFARExample
from Architectures.impl.MNISTExample import MNISTExample
from Architectures.impl.AlexNet import AlexNet


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


def get_architecture(args, ds_args):
    """
        Get the architecture based on the arguments

        :param args: Model/Training arguments
        :param ds_args: Dataset arguments
        :return: The architecture instance
    """

    name = args.architecture
    if name == "MobileNetV1":
        return MobileNetV1(num_classes=ds_args.num_classes, channels=ds_args.num_channels)
    elif name == "MobileNetV2":
        return MobileNetV2(num_classes=ds_args.num_classes, channels=ds_args.num_channels)
    elif name == "CIFARExample":
        return CIFARExample(num_classes=ds_args.num_classes)
    elif name == "MNISTExample":
        return MNISTExample()
    elif name == "AlexNet":
        return AlexNet(num_classes=ds_args.num_classes)
    else:
        raise Exception("Unknown architecture")


def get_optimizer(args):
    """
        Get the optimizer based on the arguments

        :param args: Object containing arguments as members, including name of optimizer, learning rate etc
        :return: Optimizer, just call .minimize on this object
    """

    with tf.variable_scope("input"):
        learning_rate = tf.placeholder(tf.float64, shape=[])

    name = args.optimizer
    if name == "Adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate), learning_rate
    elif name == "RMSProp":
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate), learning_rate
    elif name == "Momentum":
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum,
                                          use_nesterov=args.use_nesterov), learning_rate
    else:
        raise Exception("Unknown optimizer")


def get_data_augmentation_fn(ds_args):
    return lambda x: tf.image.resize_image_with_crop_or_pad(tf.image.random_flip_left_right(x),
                                                            target_width=ds_args.img_width,
                                                            target_height=ds_args.img_height)
