class Layer():
    """ Most generic abstract class """
    def __init__(self):
        pass

    def num_parameters(self):
        return 0
        
class WeightLayer(Layer):
    """ All layer types that contain weights """
    def __init__(self):
        pass
        
    def num_parameters(self):
        # Overide parent function in this case
        n = 0
        for size in self.shape
            n += size
            
        return n

class ConvLayer(WeightLayer):
    def __init__(self, shape, strides):
        WeightLayer.shape = shape
        self.strides = strides
        
        
class FullyConnectedLayer(WeightLayer):
    def __init__(self, shape):
        WeightLayer.shape = shape
        
        
class AveragePoolingLayer(Layer):
    def __init__(self, shape):
        """ In this case shape is the receptive field size to average over """
        self.shape = shape