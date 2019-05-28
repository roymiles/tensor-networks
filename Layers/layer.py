from enum import IntEnum
from abc import abstractmethod


class LayerTypes(IntEnum):
    CONV = 1
    DW_CONV = 2
    FC = 2
    BN = 3


class ILayer:
    """ Most generic abstract class """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
