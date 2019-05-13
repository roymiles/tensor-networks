""" Convert a network between types """

from Networks.standard_network import StandardNetwork
from Networks.tucker_network import TuckerNet
from Networks.network import LayerTypes
import tensorflow as tf
import tensorly


def standard_to_tucker(standardNetwork):
    """ Accepts a standard CNN and converts it to Tucker format """
    assert isinstance(standardNetwork, StandardNetwork)

    # All the weights across all layers
    weights = standardNetwork.get_weights()

    # Initiate the network
    tuckerNetwork = TuckerNet
    tuckerNetwork.architecture = standardNetwork.architecture
    tuckerNetwork.num_layers = standardNetwork.num_layers()

    # Loop through all the layers
    for i in range(standardNetwork.num_layers()):
        layer_weights = weights.get_layer_weights(layer_idx=i)

        if layer_weights["__type__"] == LayerTypes.CONV:
            core, factors = tensorly.tucker(weights)
            # TODO: Finish this..

