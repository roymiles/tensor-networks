import json
import os
from Architectures.impl.MobileNetV2 import MobileNetV2
from Networks.impl.sandbox import TuckerNet
from Networks.impl.standard import StandardNetwork


def load_config(name):
    """ Load training configuration """
    with open(f"{os.getcwd()}/config/{name}") as json_file:
        return json.load(json_file)


def get_architecture(name):
    """ Get the architecture from the name in the config """
    if name == "mobilenetv2":
        return MobileNetV2
    else:
        raise Exception("Unknown architecture")
