""" Generic tensor network. Core tensors can be added to the graph and the result can be compiled
    to generate the Tensorflow variables.

    We use dummy nodes to represent dangling (open indices) """
import tensorflow as tf
import networkx as nx
from base import *
from base import random_string
from Weights.utils import sine2D
import numpy as np


class Graph:
    # Intermediate representation is using NetworkX
    _graph = None

    # Once compiled, can no longer add edges (as this will change dimensions of node tensors)
    _is_compiled = False

    # Tensorflow naming scope
    _name = None

    # List of nodes for the output kernel
    # If left as None, the output kernel is not reshaped after all the nodes are contracted
    _output_shape = None

    def __init__(self, name):
        self._graph = nx.Graph()
        self._name = name

    def add_node(self, u_of_edge, shape, names, initializer=tf.glorot_normal_initializer(), regularizer=None,
                 shared=None, collections=None):
        """ Creates a node with dangling edges defined by shape
            Internally, these creates dummy nodes on these dangling edges

            :param u_of_edge: Name of the node e.g. "A"
            :param shape: Dimensions of the exposed indices
            :param names: Name of the open indices e.g. "W" for width
            :param initializer: Initialization strategy
            :param regularizer: If a regularization term, for example L2 norm, weight decay
            :param shared: (boolean) If the weight is shared across layers
            :param collections: Used if you want to group tensorflow variables
            :param shared: When creating the tensorflow variable, ignore the Graph name scope
        """

        if self._is_compiled:
            raise Exception("Unable to add more edge/nodes once the graph is compiled")

        assert len(shape) == len(names), "Must have a name for each open index"

        if not self._graph.has_node(u_of_edge):
            # TODO: How can we integrate shared property (share weights across layers)
            self._graph.add_node(u_of_edge, dummy_node=False, initializer=initializer, regularizer=regularizer,
                                 shared=shared, collections=collections)  # Make it possible to share (shared=shared)

        # Create a dummy node for each of the exposed indices
        dummy_node_names = []
        for i in range(len(shape)):
            dummy_node_names.append(random_string())
            self._graph.add_node(dummy_node_names[i], dummy_node=True, initializer=None, regularizer=None,
                                 shared=None, collections=None)

        # Now connect to the dummy nodes
        for i in range(len(shape)):
            self._graph.add_edge(u_of_edge, dummy_node_names[i], weight=shape[i], name=names[i])

        # So can chain operations
        return self

    def add_edge(self, u_of_edge, v_of_edge, length, name, initializer=tf.glorot_normal_initializer(),
                 regularizer=None, shared=False, collections=None, handcrafted=False):
        """
        Adds an edge between two tensors. If these tensors do not exist, it will create them

        :param u_of_edge:
        :param v_of_edge: Names of the two nodes e.g. "A", "B"
        :param length: Size/length of the edge/dimension
        :param name: Name of the auxilliary index, typically r1, r2 etc

        NOTE: if v_of_edge does not exist, the following arguments are used for its creation
        :param initializer: Initialization strategy
        :param regularizer: If a regularization term, for example L2 norm, weight decay
        :param shared: (boolean) If the weight is shared across layers
        :param collections: Used if you want to group tensorflow variables
        :param handcrafted: If using tf.Constant, non-learnable
        """

        if self._is_compiled:
            raise Exception("Unable to add more edge/nodes once the graph is compiled")

        # Check if the nodes exist. If they do not, add them.
        # NOTE: Assumes nodes that do not exist are NOT dummy nodes
        if not self._graph.has_node(u_of_edge):
            self._graph.add_node(u_of_edge, dummy_node=False, initializer=initializer,
                                 regularizer=regularizer, shared=None, collections=collections)

        if not self._graph.has_node(v_of_edge):
            # Can specify if v is shared or part of a collection
            self._graph.add_node(v_of_edge, dummy_node=False, initializer=initializer,
                                 regularizer=regularizer, shared=shared, collections=collections)
                                 # handcrafted=handcrafted)

        self._graph.add_edge(u_of_edge, v_of_edge, weight=length, name=name)

        return self

    def compile(self):
        """ Create the tf.Variables with the dimensions outlined in the graph """

        # Loop through all the _nodes and make appropriate Tensors
        for node in self._graph.nodes():
            # e.g. [('A', 'B'), ('A', 'C')]
            connected_edges = self._graph.edges(node)
            dims = []  # Will build the shape of the tensor
            edge_names = []
            for e in connected_edges:
                edge_data = self._graph.get_edge_data(e[0], e[1])
                dims.append(edge_data["weight"])
                edge_names.append(edge_data["name"])

            if not self._graph.nodes[node]['dummy_node']:
                # Dummy nodes do not have an associated tf.Variable

                # If sharing, don't include graph name in scope (shared across graphs)
                scope_name = ""
                if not self._graph.nodes[node]['shared']:
                    scope_name += "{}/".format(self._name)

                init = self._graph.nodes[node]['initializer']
                reg = self._graph.nodes[node]['regularizer']
                collections = self._graph.nodes[node]['collections']
                with tf.variable_scope("tfvar", reuse=tf.AUTO_REUSE):

                    # TODO: Note this crashes if node is not 2D!
                    if "handcrafted" in self._graph.nodes[node]:  # DOES NOT CHECK IF == TRUE
                        c = np.array([sine2D(dims[1], dims[2])])  # 1 x w x h
                        self._graph.nodes[node]["tfvar"] = tf.constant(value=c,
                                                                       name="{}{}_handcrafted".format(scope_name, node),
                                                                       dtype=tf.float32)
                        # self._graph.nodes[node]["tfvar"] = tf.constant(np.random.normal(loc=0.0,
                        #                                                                scale=1.0, size=dims)
                        #                                               .astype(np.float32))

                    else:
                        self._graph.nodes[node]["tfvar"] = tf.get_variable("{}{}_trainable".format(scope_name, node),
                                                                           shape=dims,
                                                                           initializer=init,
                                                                           regularizer=reg,
                                                                           collections=collections,
                                                                           trainable=True)

            self._graph.nodes[node]["edge_names"] = edge_names

        self._is_compiled = True

    def set_output_shape(self, shape):
        """ Set the output shape defined by the order of output edges for after graph compilation """
        if not self._is_compiled:
            raise Exception("Can only set the output shape after the graph is compiled")

        self._output_shape = shape

    @staticmethod
    def debug(g, title="debug"):
        """ Just temporary for debugging """

        print("-------{}-------".format(title))
        edges = g.edges()
        for e in edges:
            print("Edge {} -> {}".format(e, g.get_edge_data(e[0], e[1])))

        nodes = list(g.nodes(data=True))
        for n in nodes:
            print("Node {}".format(n))
            
    def create_summaries(self):
        """ Create Tensorboard summaries for all the nodes """
        with tf.variable_scope(self._name):
            node_names = self._graph.nodes.keys()
            for node_name in node_names:
                node = self._graph.nodes[node_name]
                if not node["dummy_node"]:
                    tfvar = self._graph.nodes[node_name]["tfvar"]
                    shape = tfvar.get_shape().as_list()
                    tf.summary.histogram(f"{node_name}", tfvar, collections=['train'])

                    # If 2D tensor, add as image summary
                    if len(shape) == 2:
                        n = tf.reshape(tfvar, shape=(1, shape[0], shape[1], 1))
                        tf.summary.image(f"{node_name}", n, collections=['train'])

    @staticmethod
    def number_of_nodes(g):
        """ Returns the number of nodes that are stored as tensors. aka not open dummy nodes """
        i = 0
        attr_dict = nx.get_node_attributes(g, "dummy_node")
        for is_dummy in attr_dict.values():
            if not is_dummy:
                i += 1

        return i

    def combine(self, switch=1.0):
        """ Combine all the nodes into a single tensor - which is likely then used for convolution or something
            returns: tf.Variable with dimensions the same as all exposed edges

            :param switch: Value in range (0, 1] to control compression.
                           Effectively only uses a percentage of each factor, so s x W

            :returns W: Single weight tensor with all the exposed dimensions """

        # TODO: Make more efficient merging strategy

        assert self._is_compiled == 1, "Must be compiled before can combine the core factors"
        g = self._graph.copy()

        # If single node (aka bias, batch norm params, or standard networks) just return that tfvar
        if Graph.number_of_nodes(g) == 1:
            # Get first non-dummy node
            for key in list(g.nodes.keys()):
                if not g.nodes[key]["dummy_node"]:
                    return g.nodes[key]["tfvar"]

        # Calculate the reshape by looping through all edges and matching
        # to the reshape index names (supplied in function argument)
        s = []
        if self._output_shape:  # List of desired index order
            for index_name in self._output_shape:
                for _, _, a in self._graph.edges(data=True):
                    if a['name'] == index_name:
                        s.append(a['weight'])

        # We start with the first tensor and loop through all factors and merge
        # NOTE: Could reorder prior to make this more efficient
        updated_graph = False
        while Graph.number_of_nodes(g) > 1:
            # Keep contracting nodes until we get one node left (excluding dummy nodes)
            keys = list(g.nodes.keys())
            nkeys = len(keys)
            for i in range(nkeys):
                n1 = keys[i]

                if g.nodes[n1]["dummy_node"]:
                    continue

                for j in range(nkeys):
                    n2 = keys[j]

                    if g.nodes[n2]["dummy_node"]:
                        # Don't contract dummy nodes (open indices)
                        continue

                    if n1 == n2:
                        # Don't bother trying to contract the node with itself...
                        continue

                    # Shows the edges being merged (good debugging line)
                    # print("{} : {}".format(n1, n2))

                    node_data = g.nodes(data=True)

                    n1_data = node_data[n1]
                    n2_data = node_data[n2]

                    # Attempt to contract the two nodes
                    if Graph.nodes_connected(g, n1, n2):
                        g, n3 = self.contract_nodes(g, n1, n2)

                        # The edges were contracted and put into n2
                        # ** Update tfvar and edge_names **

                        # Shared (contraction) index
                        aux_ind = list(set(n1_data["edge_names"]).intersection(n2_data["edge_names"]))
                        # All the indices
                        all_ind = n1_data["edge_names"] + n2_data["edge_names"]
                        # Remove auxiliary indices
                        open_ind = [e for e in all_ind if e not in aux_ind]

                        g.nodes[n3]["edge_names"] = open_ind
                        g.nodes[n3]["tfvar"] = self.combine_factors(n1_data, n2_data, switch)
                        g.nodes[n3]["dummy_node"] = False

                        updated_graph = True

                    if updated_graph:
                        break

                if updated_graph:
                    break

            # Reset (rescan over graph)
            updated_graph = False

        # Uncomment if want to see the nodes and edges in the graph
        # Graph.debug(g, "debug2")

        # Reshape to desired
        tfvar = g.nodes[n3]["tfvar"]

        if self._output_shape and s:
            tfvar = tf.reshape(tfvar, s)

        return tfvar

    @staticmethod
    def nodes_connected(g, u, v):
        """ True/False depending on if u is connected to v in graph g """
        return u in g.neighbors(v)

    @staticmethod
    def contract_nodes(g, n1, n2):
        """ Yes, this is because NetworkX didn't actually remove nodes after contraction.
            It merely replaced the graph with some weird {'contraction': {'A': {'tfvar':] ... } thing

            This function takes in a graph, and two nodes. It attempts to contract the two nodes
            and replaces the second node with the contracted node

            NOTE: This does not update the attributes, only updates the edges/nodes
            NOTE: This does not update self._graph

            :param
                n1: Node 1
                n2: Node 2

            :return
                Node: New graph with contracted node (n1 + n2) """

        if not Graph.nodes_connected(g, n1, n2):
            raise Exception("Unable to contract the two nodes (not connected). Perform this check a-priori.")

        assert not g.nodes[n1]["dummy_node"] or not g.nodes[n2]["dummy_node"], "Cannot contract dummy nodes"

        # Just for a sanity check
        num_nodes_before = g.number_of_nodes()

        n1_edges = list(g.edges(n1))
        n2_edges = list(g.edges(n2))

        n3 = str(n1 + "_" + n2)  # New node name just concatenates names
        g.add_node(node_for_adding=n3)
        open_edges = []

        # Loop through n1 and n2 to find open edges and add them to this new node
        for edge in n1_edges:
            if edge[0] == n1 and edge[1] != n2:
                open_edges.append((n3, edge[1]))
            elif edge[1] == n1 and edge[0] != n2:
                open_edges.append((edge[0], n3))

        for edge in n2_edges:
            if edge[0] == n2 and edge[1] != n1:
                open_edges.append((n3, edge[1]))
            elif edge[1] == n2 and edge[0] != n1:
                open_edges.append((edge[0], n3))

        # Removes nodes and all connected edges
        g.remove_node(n1)
        g.remove_node(n2)

        # Add the edges to the new node
        for edge in open_edges:
            g.add_edge(edge[0], edge[1])

        assert g.number_of_nodes() == num_nodes_before - 1, "Node contraction should reduce the number of nodes by one"

        return g, n3

    def get_graph(self):
        """ Get the underlying NetworkX graph """
        return self._graph

    def get_node(self, node):
        """
        Return a single node, typically for a Tensorflow summary

        :param node: Name of the node
        :return: Corresponding tf.Tensor
        """
        return self._graph.nodes[node]["tfvar"]

    @staticmethod
    def multiway_tensor_slice(tensor, axis, widths):
        """
        Slice tensor along multiple axis to specified widths
        Effectively a switch like in Slimmable networks

        :param tensor: The tf.Tensor
        :param axis: List of indexes that are sliced
        :param widths: List of widths for each axis
        :return: Sliced tf.Tensor
        """

        assert len(axis) == len(widths), "Must specify the width for each sliced axis"

        # Extract the sliced factors across all shared indices (as determined by switch)
        n = 0  # Keep track on what dim_size we are on
        slice_ind = ()
        for i, d in enumerate(tensor.get_shape()):
            # Affectively trying to do :, :, :, 0:L, ...
            # Where 0:L is a shared index and L is the dim_size (after compression)
            if i in axis:
                # Compressing along this axis
                slice_ind += (slice(0, widths[n], 1),)
                n += 1
            else:
                # Not compressing, keep entire width
                slice_ind += (slice(None),)

        # Perform the multi-way slice
        return tensor[slice_ind]

    def combine_factors(self, u_data, v_data, switch=1):
        """ Attempts to combine two tensors using tensor contraction over shared indices

            :param
                u_data, v_data: Node data
                                e.g. {'tfvar': <tf.Variable 'A:0' shape=(213, 122) dtype=float32_ref>,
                                      'edge_names': ['r1', 'r2']}
                switch: Value in range (0, 1]. Only use a fraction of the factors across the contracted dimension
            :return
                Combined weight or None if no shared indices !!! Where does it return None?? !!!"""

        if not self._is_compiled:
            raise Exception("Graph must be compiled before you can combine factors.")

        assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

        # Get index/key of shared (auxiliary indices)
        u_axis = []
        v_axis = []
        for i, ud in enumerate(u_data["edge_names"]):
            for j, vd in enumerate(v_data["edge_names"]):
                # If they have the same edge name (at edge index i, j)
                if ud == vd:
                    u_axis.append(i)
                    v_axis.append(j)

        # The dimension size for all the shared indices should be equal
        u_shape = u_data['tfvar'].get_shape().as_list()
        v_shape = v_data['tfvar'].get_shape().as_list()
        # The widths for the auxilliary indices
        widths = []
        for i, j in zip(u_axis, v_axis):
            assert u_shape[i] == v_shape[j], \
                f"Dimension size mismatch for contraction {u_shape[i]} and {v_shape[i]}"

            # Compress the factors using switch, effectively only use a percentage across this index
            # The size of the dimension after compression (just multiply by switch)
            d = tf.dtypes.cast(switch * u_shape[i], dtype=tf.int32)
            # print("{} -> {}".format(u_shape[i], d))
            # assert d > 0, "Compressing a bit too much on this index"
            widths.append(d)

        # Extract appropriate width slices along the shared axis
        u = Graph.multiway_tensor_slice(u_data['tfvar'], u_axis, widths)
        v = Graph.multiway_tensor_slice(v_data['tfvar'], v_axis, widths)

        c = tf.tensordot(u, v, axes=[u_axis, v_axis])
        return c  # The dimensions are the open edges

    def num_parameters(self):
        """ Return the total number of parameters across all (non-dummy) nodes """
        # TODO: This function should take in a switch parameters.
        assert self._is_compiled, "Must be compiled first"

        num_params = 0
        nodes = list(self._graph.nodes.keys())
        for node in nodes:

            # Dummy nodes are not stored as tensors, these are "exposed" indices
            if self._graph.nodes[node]["dummy_node"]:
                continue

            num_params += tfvar_size(self._graph.nodes[node]["tfvar"])

        return num_params


if __name__ == "__main__":
    """ Test examples """
    g = Graph("MyGraph")
    g.add_node("A", shape=[22, 9], names=["a1", "a2"])
    g.add_node("B", shape=[5], names=["b1"])
    g.add_edge("A", "B", length=10, name="r1")
    g.compile()
    g.debug(g.get_graph(), "debug1")
    tfvar = g.combine()
    print("Compressed size {}".format(g.num_parameters()))
    print("Merged size {}".format(tfvar_size(tfvar)))
    print("Combined shape {}".format(tfvar))
