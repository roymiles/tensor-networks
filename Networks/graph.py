""" Generic tensor network. Core tensors can be added to the graph and the result can be compiled
    to generate the Tensorflow variables.

    We use dummy nodes to represent dangling (open indices) """
import tensorflow as tf
import networkx as nx
from base import *
import numpy as np
from base import random_string


class Graph:
    # Intermediate representation is using NetworkX
    _graph = None

    # Once compiled, can no longer add edges (as this will change dimensions of node tensors)
    _is_compiled = False

    # Tensorflow naming scope
    _name = None

    def __init__(self, name):
        self._graph = nx.Graph()
        self._name = name

    def add_node(self, u_of_edge, shape, names):
        """ Creates a node with dangling edges defined by shape
            Internally, these creates dummy nodes on these dangling edges

            :param u_of_edge: Name of the node e.g. "A"
                   shape: Dimensions of the exposed indices
                   name: Name of the open indices e.g. "W" for width """

        if self._is_compiled:
            raise Exception("Unable to add more edge/nodes once the graph is compiled")

        assert len(shape) == len(names), "Must have a name for each open index"

        if not self._graph.has_node(u_of_edge):
            self._graph.add_node(u_of_edge, dummy_node=False)

        # Create a dummy node for each of the exposed indices
        dummy_node_names = []
        for i in range(len(shape)):
            dummy_node_names.append(random_string())
            self._graph.add_node(dummy_node_names[i], dummy_node=True)

        # Now connect to the dummy nodes
        for i in range(len(shape)):
            self._graph.add_edge(u_of_edge, dummy_node_names[i], weight=shape[i], name=names[i])

    def add_edge(self, u_of_edge, v_of_edge, length, name):
        """
        Adds an edge between two tensors. If these tensors do not exist, it will create them

        :param u_of_edge:
               v_of_edge: Names of the two nodes e.g. "A", "B"
               length: Size/length of the edge/dimension
               name: Name of the auxilliary index, typically r1, r2 etc
        """

        if self._is_compiled:
            raise Exception("Unable to add more edge/nodes once the graph is compiled")

        # Check if the nodes exist. If they do not, add them.
        if not self._graph.has_node(u_of_edge):
            # Dummy is always v_of_edge
            self._graph.add_node(u_of_edge, dummy_node=False)

        if not self._graph.has_node(v_of_edge):
            self._graph.add_node(v_of_edge, dummy_node=False)

        self._graph.add_edge(u_of_edge, v_of_edge, weight=length, name=name)

    def compile(self):
        """ Create the tf.Variables with the dimensions outlined in the graph """

        # Loop through all the _nodes and make appropriate Tensors
        with tf.variable_scope(self._name):
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
                    self._graph.nodes[node]["tfvar"] = tf.get_variable(node, shape=dims, initializer=initializer)

                self._graph.nodes[node]["edge_names"] = edge_names

        self._is_compiled = True

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

    @staticmethod
    def number_of_nodes(g):
        """ Returns the number of nodes that are stored as tensors. aka not open dummy nodes """
        i = 0
        attr_dict = nx.get_node_attributes(g, "dummy_node")
        for is_dummy in attr_dict.values():
            if not is_dummy:
                i += 1

        return i

    def combine(self):
        """ Combine all the nodes into a single tensor - which is likely then used for convolution or something
            returns: tf.Variable with dimensions the same as all exposed edges

            :returns W: Single weight tensor with all the exposed dimensions """

        # TODO: Make more efficient merging strategy

        assert self._is_compiled == 1, "Must be compiled before can combine the core factors"

        # We start with the first tensor and loop through all factors and merge
        # NOTE: Could reorder prior to make this more efficient
        g = self._graph.copy()
        updated_graph = False
        while Graph.number_of_nodes(g) > 1:
            # Keep contracting nodes until we get one
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

                    print("{} : {}".format(n1, n2))

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
                        g.nodes[n3]["tfvar"] = self.combine_factors(n1_data, n2_data)
                        g.nodes[n3]["dummy_node"] = False

                        updated_graph = True

                    if updated_graph:
                        break

                if updated_graph:
                    break

            # Reset (rescan over graph)
            updated_graph = False

        Graph.debug(g, "debug2")

        return g.nodes[n3]["tfvar"]

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

    def combine_factors(self, u_data, v_data):
        """ Attempts to combine two tensors using tensor contraction over shared indices

            :param
                u_data, v_data: Node data
                                e.g. {'tfvar': <tf.Variable 'A:0' shape=(213, 122) dtype=float32_ref>,
                                      'edge_names': ['r1', 'r2']}
            :return
                Combined weight or None if no shared indices """

        if not self._is_compiled:
            raise Exception("Graph must be compiled before you can combine factors.")

        # Get index/key of shared (auxiliary indices)
        u_ind = []
        v_ind = []
        for i, ud in enumerate(u_data["edge_names"]):
            for j, vd in enumerate(v_data["edge_names"]):
                if ud == vd:
                    u_ind.append(i)
                    v_ind.append(j)

        assert len(u_ind) == len(v_ind), "Something has gone horribly wrong"

        c = tf.tensordot(u_data['tfvar'], v_data['tfvar'], axes=[u_ind, v_ind])
        return c  # The dimensions are the open edges

    def num_parameters(self):
        """ Return the total number of parameters across all (non-dummy) nodes """
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
