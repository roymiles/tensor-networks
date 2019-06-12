""""
    Why? Having wrappers for the layers allows us to add extra behaviour to the networks. 
    New layers can then inherit from the same common interface.
    
"""


class ILayer:
    """ Most generic abstract class """

    def __init__(self):
        pass

    def __call__(self):
        pass
