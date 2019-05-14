""" Convert a network between types """

from Networks.impl.standard_network import StandardNetwork
from Networks.impl.tucker_network import TuckerNet
from Networks.network import LayerTypes
import tensorly


def standard_to_tucker(standardNetwork):
    """ Accepts a standard CNN and converts it to Tucker format

        Params:
            standardNetwork: ...

        Returns
            tuckerNetwork: Same architecture as standardNetwork"""

    assert isinstance(standardNetwork, StandardNetwork)

    # All the weights across all layers
    weights = standardNetwork.get_weights()

    # Instantiate and initialise the network
    tuckerNetwork = TuckerNet()
    tuckerNetwork.architecture = standardNetwork.architecture
    tuckerNetwork.num_layers = standardNetwork.num_layers()

    # Loop through all the layers and parse the weights
    for i in range(standardNetwork.num_layers()):
        layer_weights = weights.get_layer_weights(layer_idx=i)

        if layer_weights["__type__"] == LayerTypes.CONV:
            """ Decompose the single weight tensor into factors """
            core, factors = tensorly.tucker(layer_weights["kernel"])

        elif layer_weights["__type__"] == LayerTypes.FC:

        elif layer_weights["__type__"] == LayerTypes.BN:

        else:
            raise Exception("Unknown layer type for conversion")

    return tuckerNetwork


def tucker_to_standard(tuckerNetwork):

    assert isinstance(tuckerNetwork, TuckerNet)

