from enum import IntEnum
from abc import abstractmethod


class LayerTypes(IntEnum):
    CONV = 1
    DW_CONV = 2
    FC = 2
    BN = 3


"""
    Why? Having wrappers for the layers allows us to add extra behaviour to the networks. 
    New layers can then inherit from the same common interface.
    
"""


class ILayer:
    """ Most generic abstract class """

    def __init__(self):
        pass

    def __call__(self):
        pass
