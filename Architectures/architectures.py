class IArchitecture:
    """ All architecture must inherit from this class """
    def __init__(self, network):
        # Array of sequential layers
        self._network = network

    def num_parameters(self):
        """ Get the number of parameters for an entire architecture """
        n = 0
        for layer in self._network:
            n += layer.num_parameters()

        return n

    def num_layers(self):
        return len(self._network)

    def get_layer(self, layer_idx):
        return self._network[layer_idx]