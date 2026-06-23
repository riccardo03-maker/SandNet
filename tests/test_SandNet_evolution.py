#!/usr/bin/python
# -*- coding: utf-8 -*-

import SandNet
import networkx as nx
import pytest

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

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


def test_wrong_lose_probability_input():
    '''
    Tests the raise of a ValueError when the lose probability for the evolve function is not a probability (not between 0 and 1)

    GIVEN: a sandpile model on an arbitrary network
    WHEN: I evolve the model giving a wrong lose probability
    THEN: the code raises a ValueError
    '''
    with pytest.raises(ValueError):
        SandNet.Model().evolve(10, lose_probability=2)


def test_no_infinite_loop_with_lose_probability():
    '''
    Tests that the system does not remain trapped into an avalanche of infinite size when the threshold is equal to the degree
    of each node, but we have a non-zero probability to lose grains during a toppling

    GIVEN: a sandpile model on a 50 x 50 square grid, with threshold equal to the degree of each node and initial
    random number of grains
    WHEN: I evolve the model for 5000, adding grains in random positions and with a probability of 0.0001 to lose a grain
    during a toppling
    THEN: the code runs without remaining trapped in an avalanche of infinite size
    '''
    model = SandNet.Model(N = 50, threshold_rule="degree", initial_grains="random")
    model.evolve(5000, lose_probability=0.0001)
    assert True


def test_infinite_loop_without_lose_probability():
    '''
    Tests that in the same situation of the previous test but without losing grains during a toppling the system
    remains trapped in an avalanche of infinite size

    GIVEN: a sandpile model on a 50 x 50 square grid, with threshold equal to the degree of each node and initial
    random number of grains
    WHEN: I evolve the model for 5000, adding grains in random positions
    THEN: the code raises a RecursionError
    '''
    with pytest.raises(RecursionError):
        SandNet.Model(N = 50, threshold_rule='degree', initial_grains = "random").evolve(5000)


def test_loop_without_boundaries():
    '''
    Tests that in a sandpile model on a network without boundaries (which means a threshold equal to the degree of
    each node) the system remains trapped in a loop of infinite size after some steps of evolution
    
    GIVEN: a sandpile model on a fully connected network of 10 nodes, with a threshold equal to the degree of each node
    WHEN: I evolve for 1000 steps
    THEN: the code raises a RecursionError (means that it is trapped in an infinite size avalanche)
    '''
    G = nx.complete_graph(10) #returns a fully connected graph with 10 nodes
    with (pytest.raises(RecursionError)):
        SandNet.Model(G, threshold_rule='degree').evolve(1000)


def test_avoid_loop_with_boundaries():
    '''
    Tests that in the same situation of the previous test but with boundaries in the network the system never remains trapped
    in a loop of infinite size
    
    GIVEN: a sandpile model on a fully connected network of 10 nodes, with a threshold equal to the degree of each node
    WHEN: I create one boundary
    THEN: the system can evolve for 1000 steps without remaining trapped in an avalanche of infinite size
    '''
    G = nx.complete_graph(10)
    model = SandNet.Model(G, threshold_rule='degree')
    model.add_boundaries(index_list = [1])
    model.evolve(1000)

    assert True #we just want to test that the code reaches the end
    

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
