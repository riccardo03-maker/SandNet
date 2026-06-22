#!/usr/bin/python
# -*- coding: utf-8 -*-

import SandNet
import networkx as nx
import numpy as np
import pytest
from collections import Counter

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

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
    assert(retrieved_index==correct_index)


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


#Testing sandpile evolution and avalanches


def test_incorrect_evolution_mode():
    '''
    Tests the raise of a ValueError when the evolution mode is not a standard one

    GIVEN: a classical sandpile model with a square grid network 5x5
    WHEN: the evolution mode is not 'fixed' or 'random'
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    with pytest.raises(ValueError):
        model.evolve(steps = 50, evolve_mode = 'not_standard', position = 3)


def test_incorrect_position_index():
    '''
    Tests the raise of a ValueError when the position at which new grains should be added does not exist

    GIVEN: a classical sandpile model with a square grid network 5x5
    WHEN: the input value for the index does not correspond to any node
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    with pytest.raises(ValueError):
        model.evolve(steps = 50, evolve_mode = 'fixed', position = 25)


def test_zero_degree_node_avalanche():
    '''
    Tests the raise of an Exception when the node from which an avalanche starts has no neighbours

    GIVEN: a classical sandpile model on a 1-node network
    WHEN: during the evolution, the single node reaches its threshold and starts toppling, but it has no neighbours
    to which grains can be sent
    THEN: the code raises an Exception
    '''
    G = nx.Graph()
    G.add_node(1)
    model = SandNet.Model(G)
    with pytest.raises(Exception):
        model.evolve(50, 'fixed')


def test_infinite_avalanche_loop():
    '''
    Tests the raise of a RecursionError when an avalanche of infinite size occurs

    GIVEN: a sandpile model on a network with three nodes, each one connected to the other two, with a threshold
    given by the node degree (2 for each node)
    WHEN: the system evolves for 4 steps, adding grains always on the same node.
    At the third step each node will have one grain, so at the fourth step an avalanche of infinite size starts
    THEN: the code raises a RecursionError 
    '''
    G = nx.Graph()
    G.add_nodes_from(range(3))
    G.add_edges_from([(0,1), (1, 2), (0, 2)])
    model = SandNet.Model(G, threshold_rule='degree')

    with pytest.raises(RecursionError):
        model.evolve(4, evolve_mode= 'fixed', position=1)


def test_classical_model_evolution():
    '''
    Tests the correct behaviour of the evolution on a 3x3 grid

    GIVEN: a 3x3 grid network with a threshold of 4 for each node
    WHEN: I evolve for 16 steps, adding all grains to the central node
    THEN: I have two grains in each vertex, one grain at each side and 0 grains in the center
    '''
    model = SandNet.Model(N=3)
    model.evolve(16, evolve_mode='fixed', position=model.network.nodes[(1, 1)]["index"])
    #nodes in a 2d square grid created with the grid_2d_graph function are named using tuples of 2 integers
    #in this case (1,1) represents the center of the grid
    #see networkx documentation for better explanation

    assert(model.network.nodes[(1, 1)]["grains"] == 0) #center
    assert(model.network.nodes[(0, 2)]["grains"] == 2) #vertex
    assert(model.network.nodes[(1, 2)]["grains"] == 1) #side


def test_threshold_lower_than_neighbours():
    '''
    Tests the raise of an Exception when one node topples and it has a threshold lower than the number of neighbours,
    so not all the neighbours can receive a grain

    GIVEN: a sandpile model on a network of four nodes, with one central node connected to all the others, and 
    a fixed threshold height of 2
    WHEN: I evolve for 2 steps, adding all grains to the central node
    THEN: at the second step the node topples, and since the number of neighbours is higher than the threshold,
    the code raises an Exception
    '''
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1), (1, 2), (1, 3)])
    model = SandNet.Model(G, threshold_rule='fixed', threshold=2)

    with(pytest.raises(Exception)):
        model.evolve(2, evolve_mode='fixed', position = 1)


def test_threshold_higher_than_neighbours():
    '''
    Tests the behaviour of the code when one node topples and it has a threshold higher than the number of neighbours,
    so some grains are lost (only one grain is given to each neighbour)

    GIVEN: a sandpile model on a network of four nodes, with one central node connected to all the others, and 
    a fixed threshold height of 2
    WHEN: I evolve for 2 steps, adding all grains to one of the non-central nodes
    THEN: at the second step the node topples, and since the number of neighbours is lower than the threshold,
    the total number of grains in the system after the avalanche is 1
    '''
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1), (1, 2), (1, 3)])
    model = SandNet.Model(G, threshold_rule='fixed', threshold=2)

    model.evolve(2, evolve_mode='fixed', position = 2)
    assert(model.network.nodes[0]["grains"] + model.network.nodes[1]["grains"] + model.network.nodes[2]["grains"] 
           + model.network.nodes[3]["grains"] == 1)


def test_dynamics_on_large_network():
    '''
    Tests the correct evolution of the sandpile model on a large 2d grid network for a large number of steps. Since the
    threshold is fixed to 4 and so some nodes have a threshold lower than their degree, grains can be lost by nodes at the borders,
    so the system should never remain trapped in an infinite size avalanche

    GIVEN: a sandpile model on a 100 x 100 2d grid network, with threshold fixed to 4
    WHEN: I evolve for 20000 steps, adding grains in random positions
    THEN: the code must never raise an error

    Note: In this test we just want to verify that the code reaches its end, without testing the results, since we
    don't have any property that can be tested properly for large networks
    '''
    model = SandNet.Model(N = 100)
    model.evolve(20000)
    assert True


#Avalanche size calculation


def test_avalanche_size_calculation():
    '''
    Tests the correct calculation of avalanche size history

    GIVEN: a 3x3 grid network with a threshold of 4 for each node
    WHEN: I evolve for 16 steps, adding all grains to the central node
    THEN: I have an avalanche of size 1 at step 4, 8 and 12, an avalanche of size 6 at step 16, and avalanches
    of size 0 for all the other steps
    '''
    model = SandNet.Model(N=3)
    model.evolve(16, evolve_mode='fixed', position=model.network.nodes[(1, 1)]["index"])
    #nodes in a 2d square grid created with the grid_2d_graph function are named using tuples of 2 integers
    #in this case (1,1) represents the center of the grid
    #see networkx documentation for better explanation

    assert(model.avalanche_sizes_collector[3] == 1) #avalanche_size_collector[N-1] corresponds to step N
    assert(model.avalanche_sizes_collector[7] == 1)
    assert(model.avalanche_sizes_collector[11] == 1)
    assert(model.avalanche_sizes_collector[4] == 0)
    assert(model.avalanche_sizes_collector[15] == 6)


def test_more_complex_avalanche_size():
    '''
    Tests the correct calculation of avalanche size for a more complex avalanche with respect to previous tests.
    The calculation of the correct value of avalanche size has been done directly by hand

    GIVEN: a 5x5 grid network with a threshold of 4 for each node, with 3 grains in each node
    WHEN: I add one grain on the central node
    THEN: I have an avalanche of size 35
    '''
    model = SandNet.Model()
    for index in range(25):
        model.network.nodes[model.select_node_by_index(index)]["grains"] = 3
        #set manually the initial condition because we want to test just one avalanche
    
    central_index = model.network.nodes[(2, 2)]["index"]
    #the index of the central node, that can be passed as an argument to the evolve method

    model.evolve(1, evolve_mode='fixed', position = central_index)

    assert(model.avalanche_sizes_collector[0] == 35)


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


#Testing node degree retrieval


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


def test_incorrect_input_index():
    '''
    Tests the raise of a ValueError when using the get_node_degree method with an input index not associated to
    any node

    GIVEN: a 5x5 grid network
    WHEN: I want to obtain the degree of node with index 25 (not existing, indexes go from 0 to 24)
    THEN: the code raises a ValueError
    '''
    with(pytest.raises(ValueError)):
       SandNet.Model().get_node_degree(25) 


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


def test_loop_without_boundaries():
    '''
    TODO
    '''
    assert True


def test_avoid_loop_with_boundaries():
    '''
    TODO
    '''
    assert True