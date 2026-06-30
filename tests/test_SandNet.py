#!/usr/bin/python
# -*- coding: utf-8 -*-

import SandNet
import networkx as nx
import numpy as np
import pytest
from collections import Counter

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


# Testing network retrieval


def test_get_network():
    '''
    Tests the correct retrieval of the network structure of the sandpile model

    GIVEN: a SandNet.Model object with a certain network structure
    WHEN: I get the network structure of the sandpile model
    THEN: I obtain a network equal to the one given as input

    Note: the graph given as input for the initialization of the SandNet.Model object is different at any repetition of the test,
    but this is not an issue since we are only testing the equivalence between the input and output networks
    '''
    G = nx.fast_gnp_random_graph(1000, 0.4) #a random graph generator from networkx
    model = SandNet.Model(G)
    network = model.get_network()
    assert(nx.is_isomorphic(G, network))
    #True if the two graphs are isomorphic. We cannot verify the perfect equality, since some attributes (threshold and grains) have
    #been added to the network during model initialization, so the two networks will not be exactly equal.


# Testing initialization of SandNet.Model


def test_default_square_lattice_model():
    '''
    Tests the initialization of a default classical square lattice sandpile model

    GIVEN: I am initializing a SandNet.Model object
    WHEN: I don't give any input parameter
    THEN: I obtain a sandpile model on a square lattice network with 25 nodes (square of side 5, as in the default N parameter).
    So we have four nodes with degree 2 (vertices), twelve nodes with degree 3 (sides) and nine nodes with degree 4 (internal nodes)
    '''
    model = SandNet.Model()

    #check we have 25 nodes
    assert(len(model.network.nodes) == 25)

    #check we only have nodes with the right degrees
    all_degrees = dict(model.network.degree()).values() #degrees of all nodes in the network
    degree_counts = Counter(all_degrees) #counts how many times a degree value appears in the network
    assert(degree_counts[2] == 4)
    assert(degree_counts[3] == 12)
    assert(degree_counts[4] == 9)


def test_zero_grains():
    '''
    Tests the correct initialization of the number of grains for each node of the model network to 0

    GIVEN: I am initializing a SandNet.Model object
    WHEN: I give arbitrary input parameters
    THEN: every node has an initial number of grains equal to 0

    Note: in the test I check one random node of the network. The node checked could change at any
    repetition of the test, but this does not matter since all nodes must have 0 grains
    '''

    G=nx.Graph()
    G.add_nodes_from(range(10))
    model = SandNet.Model(G, threshold_rule = "fixed", threshold = 5)
    assert(model.network.nodes[np.random.randint(0, 10)]["grains"] == 0)


def test_random_grains():
    '''
    Tests the correct initialization of the number of grains for each node of the model network to a random number
    lower than the node threshold

    GIVEN: a sandpile model on a network of four nodes, with a central node connected to all the others and a threshold
    equal to the degree
    WHEN: I set a random initial number of grains
    THEN: every node which is not the central one has an initial number of grains equal to 0
    (because the threshold is 1 for those nodes)
    '''
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    model = SandNet.Model(G, threshold_rule= "degree", initial_grains="random")

    assert(model.network.nodes[0]["grains"] == 0)
    assert(model.network.nodes[2]["grains"] == 0)
    assert(model.network.nodes[3]["grains"] == 0)


def test_wrong_initial_grains_input():
    '''
    Tests the raise of ValueError when the input rule for initial grains is not in the list of possible values

    GIVEN: I am initializing a SandNet.Model object
    WHEN: the input value for the initial_grains argument is not a standard one
    THEN: the code raises a ValueError
    '''
    with pytest.raises(ValueError):
        SandNet.Model(initial_grains="not_standard_string")


def test_fixed_threshold():
    '''
    Tests the setting of a fixed threshold in the network of the model

    GIVEN: a model network with ten nodes and without links (links are useless in this case)
    WHEN: I set a threshold of 5 for each node
    THEN: any node has a threshold of 5

    Note: in the test I check one random node of the network. The node checked could change at any
    repetition of the test, but this does not matter since all nodes must have the same threshold
    '''
    
    G=nx.Graph()
    G.add_nodes_from(range(10))
    model = SandNet.Model(G, threshold_rule = "fixed", threshold = 5)
    assert(model.network.nodes[np.random.randint(0, 10)]["threshold"] == 5)


def test_fixed_threshold_lower_than_one():
    '''
    Tests the raise of ValueError when the input fixed threshold is lower than 1

    GIVEN: I am initializing a SandNet.Model object
    WHEN: the input value for the fixed threshold is lower than one
    THEN: the code raises a ValueError
    '''
    with pytest.raises(ValueError):
        SandNet.Model(threshold_rule = 'fixed', threshold = 0)
                

def test_degree_threshold():
    '''
    Tests the setting of a threshold equal to the degree of each node in the network of the model

    GIVEN: a model network with three nodes in a 1-dimensional chain (the first one is linked with the second and the second with the third)
    WHEN: I set a threshold equal to the degree
    THEN: the central node has a threshold of 2, while the two side nodes have a threshold of 1
    '''

    G=nx.Graph()
    G.add_nodes_from(range(3))
    G.add_edges_from([(0, 1), (1, 2)])
    model = SandNet.Model(G, threshold_rule = "degree")
    assert(model.network.nodes[1]["threshold"]==2) #central node
    assert(model.network.nodes[0]["threshold"]==1) #one of the side nodes


def test_incorrect_rule_input():
    '''
    Tests the raise of ValueError when the input threshold rule is not in the list of possible values

    GIVEN: I am initializing a SandNet.Model object
    WHEN: the input value for the threshold_rule argument is not a standard one
    THEN: the code raises a ValueError
    '''
    with pytest.raises(ValueError):
        SandNet.Model(threshold_rule="not_standard_string")


def test_index_setting():
    '''
    Tests the correct setting of indexes to the network nodes

    GIVEN: I am initializing a SandNet.Model object
    WHEN: I give as input a network with N nodes
    THEN: the node indexes range from 0 to N-1
    '''
    model = SandNet.Model(N=6)
    indexes = list(dict(model.network.nodes(data="index", default=0)).values()) #gives a list of all the index values

    #compare using collections.Counter to have comparison independent from order
    assert(Counter(indexes) == Counter(range(36)))

#Testing selection node by index

def test_generic_index_selection():
    '''
    Tests the correct extraction of a network node starting from the index

    GIVEN: a sandpile model on a network of 3 nodes with heterogeneous types
    WHEN: I extract the nodes from their indexes
    THEN: I retrieve the correct node values
    '''
    G = nx.Graph()

    #add nodes with heterogeneous types
    G.add_node(1)
    G.add_node("string")
    G.add_node((6, "tuple"))

    model = SandNet.Model(G)
    #indexes are assigned following node order, so node 1 corresponds to index 0,
    #node "string" to index 1 and so on

    assert(model.select_node_by_index(0) == 1)
    assert(model.select_node_by_index(1) == "string")
    assert(model.select_node_by_index(2) == (6, "tuple"))


def test_wrong_index_input():
    '''
    Tests the raise of error when the input index is higher than the number of nodes in the network

    GIVEN: a classical sandpile model with a square grid network 5x5
    WHEN: the input value for the index does not correspond to any node
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    with pytest.raises(ValueError):
        model.select_node_by_index(25)


#Testing selection node by degree


def test_generic_degree_selection():
    '''
    Tests the correct selection of a network node starting from the degree

    GIVEN: a network of 4 nodes with one central node of degree 3
    WHEN: I select a node with degree 3
    THEN: I obtain the index of the central node of the network (the only one with degree 3)
    '''
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    model = SandNet.Model(network=G, threshold_rule='degree')
    correct_index = model.network.nodes[1]["index"] #get the index of the node of degree 3 for the test

    retrieved_index = model.select_nodes_by_degree(3)
    assert(retrieved_index==[correct_index]) #select_node_by_degree returns a list, not an integer


def test_wrong_degree_input():
    '''
    Tests the raise of error when no node in the network has the input degree and the parameter raises is True

    GIVEN: a classical sandpile model with a square grid network
    WHEN: the input value for the degree does not match the degree of any node and the parameter raises is True
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    with pytest.raises(ValueError):
        model.select_nodes_by_degree(1)


def test_degree_empty_list():
    '''
    Tests the return of an empty list when no node in the network has the input degree and the parameter raises is False

    GIVEN: a classical sandpile model with a square grid network
    WHEN: the input value for the degree does not match the degree of any node and the parameter raises is False
    THEN: the code returns an empty list
    '''
    model=SandNet.Model()
    nodes = model.select_nodes_by_degree(1, raises=False)
    assert(not nodes)


#Testing node counting


def test_correct_node_count():
    '''
    Tests the correct count of the nodes in the network

    GIVEN: a 5x5 grid network
    WHEN: I want to obtain the number of nodes in the network
    THEN: I obtain 25 nodes
    '''
    model = SandNet.Model()
    assert(model.get_number_of_nodes() == 25)


#Testing node features retrieval


def test_correct_node_degree_retrieval():
    '''
    Tests the correct retrieval of the degree of a node in the sandpile model network

    GIVEN: sandpile model on a 3x3 grid network
    WHEN: I want to obtain the degree of nodes
    THEN: I obtain degree 4 for the central node, degree 3 for sides and degree 2 for vertexes
    '''
    model = SandNet.Model(N = 3)
    assert(model.get_node_degree(0) == 2) #index 0 is the (0, 0) node of the grid, a vertex
    assert(model.get_node_degree(1) == 3) #index 1 is the (0, 1) node of the grid, a side
    assert(model.get_node_degree(4) == 4) #index 4 is the (1, 1) node of the grid, the center


def test_incorrect_input_index_get_degree():
    '''
    Tests the raise of a ValueError when using the get_node_degree method with an input index not associated to
    any node

    GIVEN: a 5x5 grid network
    WHEN: I want to obtain the degree of node with index 25 (not existing, indexes go from 0 to 24)
    THEN: the code raises a ValueError
    '''
    with(pytest.raises(ValueError)):
       SandNet.Model().get_node_degree(25)


def test_correct_node_grains_retrieval():
    '''
    Tests the correct retrieval of the number of grains of one or more nodes

    GIVEN: a sandpile model on a network with 3 nodes and 0 grains on each node
    WHEN: I want to obtain the number of grains of one random node
    THEN: I obtain 0 grains

    Note: in the test I check one random node of the network. The node checked could change at any
    repetition of the test, but this does not matter since all nodes must have 0 grains
    '''
    G = nx.Graph()
    G.add_nodes_from(range(3))
    model = SandNet.Model(G, initial_grains='zero')
    assert(model.get_node_grains(np.random.randint(0, 3)) == 0)


def test_correct_node_threshold_retrieval():
    '''
    Tests the correct retrieval of the threshold of one or more nodes

    GIVEN: a sandpile model on a 3x3 grid network with threshold equal to the degree
    WHEN: I want to obtain the threshold of three nodes: the center, one side and one vertex
    THEN: I obtain threshold of 4 for the center, 3 for the side and 2 for the vertex
    '''
    model = SandNet.Model(N = 3, threshold_rule='degree')
    thresholds = model.get_node_threshold([4, 1, 0]) #4 is the center, 1 is a side and 0 is a vertex
    assert(thresholds[0] == 4)
    assert(thresholds[1] == 3)
    assert(thresholds[2] == 2)


#Testing detection and creation of network boundaries


def test_standard_detection_of_boundaries():
    '''
    Tests the correct detection of the boundaries of a 2d square lattice network

    GIVEN: a sandpile model on a 3x3 grid network
    WHEN: I want to obtain the indexes of the boundary nodes of the network
    THEN: I obtain all the indexes from 0 to 8 except 4 (which is the central node)
    '''
    model = SandNet.Model(N = 3)
    assert(model.find_boundaries() == [i for i in range(9) if i != 4])


def test_detection_with_no_boundaries():
    '''
    Tests the correct detection of no boundaries in the sandpile model network when the threshold is equal
    to the degree of the nodes

    GIVEN: a sandpile model on a Barabasi-Albert network with 100 nodes and 3 starting links for each new node,
    with a threshold equal to the degree of each node
    (for more details on Barabasi-Albert networks see the documentation for the barabasi_albert_graph function in the Networkx package)
    WHEN: I want to obtain the indexes of the boundary nodes of the network
    THEN: I obtain an empty list
    '''
    G = nx.barabasi_albert_graph(100, 3)
    model = SandNet.Model(G, threshold_rule='degree')
    assert(model.find_boundaries() == [])


def test_standard_creation_of_boundaries():
    '''
    Tests the correct creation of boundaries on the lowest degree nodes of a network

    GIVEN: a sandpile model with a network of 5 nodes, with three nodes fully connected and the other two nodes
    connected one to the other
    WHEN: I create two boundaries in the network
    THEN: the threshold of the two nodes of degree 1 is increased by 1 while the threshold of the other nodes remains
    the same
    '''
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4)])
    model = SandNet.Model(G, threshold=2)
    model.add_boundaries(n_boundaries=2)

    assert(model.network.nodes[0]["threshold"] == 2)
    assert(model.network.nodes[1]["threshold"] == 2)
    assert(model.network.nodes[2]["threshold"] == 2)
    assert(model.network.nodes[3]["threshold"] == 3)
    assert(model.network.nodes[4]["threshold"] == 3)


# Test threshold and number of grains modification


def test_change_threshold():
    '''
    Tests the correct change of the threshold of some nodes

    GIVEN: a sandpile model on a 3x3 grid network
    WHEN: I want to change the threshold of two nodes to 5
    THEN: the threshold of those nodes becomes 5
    '''
    model = SandNet.Model(N = 3)
    model.change_threshold([0, 4], 5) #correspond to nodes (0, 0) and (1, 1)
    assert(model.network.nodes[(0, 0)]["threshold"] == 5)
    assert(model.network.nodes[(1, 1)]["threshold"] == 5)


def test_change_number_of_grains():
    '''
    Tests the correct change of the number of grains on one node

    GIVEN: a sandpile model on a 3x3 grid network, with 0 initial grains on each node
    WHEN: I want to change the number of grains on the central node to 3
    THEN: the number of grains of that node becomes 3
    '''
    model = SandNet.Model(N = 3)
    model.change_grains(index = 4, grains = 3) #4 is the index of the central node
    assert(model.network.nodes[(1, 1)]["grains"] == 3)


def test_wrong_new_number_of_grains():
    '''
    Tests the raise of a ValueError when trying to change the number of grains on one node to a number higher or equal than the threshold
    of that node

    GIVEN: a sandpile model on a 3x3 grid network, with 0 initial grains on each node
    WHEN: I want to change the number of grains on the central node to 4
    THEN: the code raises a ValueError
    '''
    model = SandNet.Model(N = 3)
    with pytest.raises(ValueError):
        model.change_grains(index = 4, grains = 4)


# Testing node removal


def test_node_removal():
    '''
    Tests the correct removal of nodes from a network

    GIVEN: a sandpile model on a 3x3 grid network
    WHEN: I want to remove all nodes with degree equal to 2 (so the vertexes)
    THEN: we have five remaining nodes in the network, one with degree 4 (the central one) and the others with
    degree 1 (since also the edges of the vertexes are removed together with the nodes)
    '''
    model = SandNet.Model(N = 3)
    model.remove_nodes_by_index(model.select_nodes_by_degree(2))

    assert(len(model.network.nodes) == 5)
    assert(len(model.select_nodes_by_degree(4)) == 1)
    assert(len(model.select_nodes_by_degree(1)) == 4)


def test_index_assignment_after_node_removal():
    '''
    Tests the correct re-assignment of indexes to network nodes after the removal of some nodes

    GIVEN: a sandpile model on a network with 3 nodes
    WHEN: the node with the index 1 is removed
    THEN: the node with initial index equal to 2 has now an index equal to 1
    '''
    G = nx.Graph()
    G.add_nodes_from(range(3))
    model = SandNet.Model(G)
    model.remove_nodes_by_index([1])

    assert(model.network.nodes[2]["index"] == 1)


# Testing count of total number of grains


def test_total_number_of_grains():
    '''
    Tests the correct retrieval of the total number of grains in a sandpile model network

    GIVEN: a sandpile model on a 3 x 3 square grid, with two grains on a node and one grain on all the others
    WHEN: I get the total number of grains in the system
    THEN: I obtain 10
    '''
    model = SandNet.Model(N = 3, initial_grains='zero')
    model.change_grains(index = 3, grains = 2)
    for index in [0, 1, 2, 4, 5, 6, 7, 8]:
        model.change_grains(index = index, grains = 1)
    
    assert(model.get_total_grains() == 10)