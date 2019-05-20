# TensorNetworks

A wrapper framework for building Tensor networks in Tensorflow.
These framework leverages NetworkX [1] for a simple interface for creating graphs.

For example, to create a 3-way tensor that is contracted with another 2-way tensor across an auxilliary index **r**
```

tensor_network = Graph("my_tensor_network")

# Two exposed edges for A and one exposed for B
tensor_network.add_node("A", name="A", shape=[22, 9])
tensor_network.add_node("B", name="B", shape=[5])
tensor_network.add_edge("A", "B", length=10)

```



[1] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008