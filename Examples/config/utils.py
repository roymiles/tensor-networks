import tensorflow as tf
import json
import os
import yaml
import io
from collections import namedtuple
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
from Architectures.impl.CIFARExample import CIFARExample
from Architectures.impl.MNISTExample import MNISTExample
from Architectures.impl.AlexNet import AlexNet


def _dict_object_hook(d): return namedtuple('X', d.keys())(*d.values())


# Convert json file to object
def json2obj(data): return json.loads(data, object_hook=_dict_object_hook)


_JSON = 0
_YAML = 1


def load_config(name, mode=_JSON, as_obj=True):
    """ Load training configuration """
    if mode == _JSON:
        """ Load .json format """
        with open(f"{os.getcwd()}/config/{name}") as json_file:
            if as_obj:
                # As object
                d = json_file.read()
                x = json2obj(d)
                return x
            else:
                # As dictionary
                return json.load(json_file)
    elif mode == _YAML:
        """ Load .yaml format """
        with open(f"{os.getcwd()}/config/{name}", 'r') as yaml_file:
            d = yaml_file.safe_load()
            if as_obj:
                # As object
                x = _dict_object_hook(d)
                return x
            else:
                # As dictionary
                return d
    else:
        raise Exception("Unknown file type")


def get_architecture(args, ds_args):
    """
        Get the architecture based on the arguments

        :param args: Model/Training arguments
        :param ds_args: Dataset arguments
        :return: The architecture instance
    """

    name = args.architecture
    if name == "MobileNetV1":
        return MobileNetV1(num_classes=ds_args.num_classes, channels=ds_args.num_channels,
                           switch_list=args.switch_list, method=args.method)
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


def preprocess_images_fn(ds_args):
    """ Data augmentation """
    return lambda x: tf.image.resize_image_with_crop_or_pad(tf.image.random_flip_left_right(x),
                                                            target_width=ds_args.img_width,
                                                            target_height=ds_args.img_height)


def is_epoch_decay(epoch, args):
    """ Can decay depending on list or every n epochs """
    if epoch == 0:
        return False

    if hasattr(args, 'num_epochs_decay'):
        if epoch % args.num_epochs_decay == 0:
            return True
    elif hasattr(args, 'epoch_decay_boundaries'):
        if epoch in args.epoch_decay_boundaries:
            return True

    return False
