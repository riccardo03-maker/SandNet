#!/usr/bin/python
# -*- coding: utf-8 -*-

import SandNet
import pytest
import networkx as nx
from collections import Counter

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

# Testing same exceptions of the Model.evolve function

def test_incorrect_evolution_mode():
    '''
    Tests the raise of a ValueError when the evolution mode is not a standard one

    GIVEN: a multiplex sandpile model with two square grid networks 5x5
    WHEN: the evolution mode is not 'fixed' or 'random'
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    multiplex = SandNet.Multiplex([model, model])
    with pytest.raises(ValueError):
        multiplex.evolve_together(steps = 50, evolve_mode = 'not_standard', position = 3)


def test_incorrect_position_index():
    '''
    Tests the raise of a ValueError when the position at which new grains should be added does not exist

    GIVEN: a multiplex sandpile model with two square grid networks 5x5
    WHEN: the input value for the position index does not correspond to any node
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    multiplex = SandNet.Multiplex([model, model])
    with pytest.raises(ValueError):
        multiplex.evolve_together(steps = 50, evolve_mode = 'fixed', position = 25)


def test_incorrect_layer():
    '''
    Tests the raise of a ValueError when the input layer in which new grains should be added does not exist

    GIVEN: a multiplex sandpile model with three square grid networks 5x5
    WHEN: the input value for the layer does not correspond to any layer of the multiplex network
    THEN: the code raises a ValueError
    '''
    model=SandNet.Model()
    multiplex = SandNet.Multiplex([model, model, model])
    with pytest.raises(ValueError):
        multiplex.evolve_together(steps = 50, layer = 5)


def test_zero_degree_node_avalanche():
    '''
    Tests the raise of an Exception when the node from which an avalanche starts has no neighbours

    GIVEN: a multiplex sandpile model on two 1-node networks
    WHEN: during the evolution, the single node in one of the two layers reaches its threshold and starts toppling, but it has no 
    neighbours to which grains can be sent
    THEN: the code raises an Exception
    '''
    G = nx.Graph()
    G.add_node(1)
    model = SandNet.Model(G)
    multiplex = SandNet.Multiplex([model, model])
    with pytest.raises(Exception):
        multiplex.evolve_together(50, 'fixed')


def test_wrong_lose_probability_input():
    '''
    Tests the raise of a ValueError when the lose probability for the evolve_together function is not a probability
    (not between 0 and 1)

    GIVEN: a sandpile model on an arbitrary multiplex network
    WHEN: I evolve the model giving a wrong lose probability
    THEN: the code raises a ValueError
    '''
    model_1 = SandNet.Model()
    model_2 = SandNet.Model()

    with pytest.raises(ValueError):
        SandNet.Multiplex([model_1, model_2]).evolve_together(10, lose_probability=2)


# Testing evolution with a single layer


def test_equivalence_simplex_and_multiplex_evolution():
    '''
    Tests that evolving a sandpile model on a single-layer multiplex network with the Multiplex.evolve_together function is equivalent to evolve
    a sandpile model on a simplex network with the Model.evolve function. In this way all the tests that are valid for the Model.evolve
    function are also valid for the Multiplex.evolve_together function.

    GIVEN: a sandpile model on a 5 x 5 square grid
    WHEN: I evolve the same model using the Model.evolve function, and the Multiplex.evolve_together function (for a Multiplex instance
    with only one model stored), adding grains always on the central node. The parameter 'together' of the Multiplex.evolve_together function
    is set to True.
    THEN: the two final networks obtained are exactly identical, and also avalanche sizes and avalanche areas are identical
    '''
    model = SandNet.Model(initial_grains='random', seed = 42)
    multiplex = SandNet.Multiplex([SandNet.Model(initial_grains='random', seed = 42)])
    #since the random seed is fixed, the two models created have the same random initial grains

    model.evolve(5000, evolve_mode='fixed', position = 12, avalanche_matrix=True)
    multiplex.evolve_together(5000, together = True, evolve_mode = 'fixed', position = 12, avalanche_matrix=True)

    assert(nx.utils.graphs_equal(model.network, multiplex.get_model(index = 0).network))
    assert(model.avalanche_sizes_collector == multiplex.get_model(index = 0).avalanche_sizes_collector)
    assert(model.avalanche_areas_collector == multiplex.get_model(index = 0).avalanche_areas_collector)


def test_equivalence_simplex_and_multiplex_evolution_without_propagation():
    '''
    This test is exactly identical to the previous one, except for the fact that when using the evolve_together method of the Multiplex
    class, the 'together' parameter is set to False (while in the previous test it was set to True). This means that a toppling in one
    layer of the multiplex network does not influence the other layers. Since this test and the previous one use multiplex networks 
    with only one layer, setting the 'together' parameter to True or False should not make any difference

    GIVEN: a sandpile model on a 5 x 5 square grid
    WHEN: I evolve the same model using the Model.evolve function, and the Multiplex.evolve_together function (for a Multiplex instance
    with only one model stored), adding grains always on the central node. The parameter 'together' of the Multiplex.evolve_together function
    is set to False
    THEN: the two final networks obtained are exactly identical, and also avalanche sizes and avalanche areas are identical
    '''
    model = SandNet.Model(initial_grains='random', seed = 42)
    multiplex = SandNet.Multiplex([SandNet.Model(initial_grains='random', seed = 42)])

    model.evolve(5000, evolve_mode='fixed', position = 12, avalanche_matrix=True)
    multiplex.evolve_together(5000, together = False, evolve_mode = 'fixed', position = 12, avalanche_matrix=True)

    assert(nx.utils.graphs_equal(model.network, multiplex.get_model(index = 0).network))
    assert(model.avalanche_sizes_collector == multiplex.get_model(index = 0).avalanche_sizes_collector)
    assert(model.avalanche_areas_collector == multiplex.get_model(index = 0).avalanche_areas_collector)


# Testing multiplex evolution


def test_propagation_of_the_avalanche():
    '''
    Tests the correct propagation of the avalanche from one layer of the multiplex network to the other when the 'together' parameter
    is True

    GIVEN: a sandpile model on a double-layer multiplex network, with a 3 x 3 grid for each layer. Both of them have 3 grains on the central
    node and 0 grains on the other nodes. The threshold is 4 for all nodes.
    WHEN: I evolve for one step, adding a grain on the central node of one of the two layers, with 'together' = True
    THEN: the central node of both layers topples, so in the end the central nodes of the two layers will have both 0 grains
    '''
    model_1 = SandNet.Model(N = 3, initial_grains='zero')
    model_1.change_grains(index = 4, grains = 3)
    model_2 = SandNet.Model(N = 3, initial_grains='zero')
    model_2.change_grains(index = 4, grains = 3)
    multiplex = SandNet.Multiplex([model_1, model_2], ["first", "second"])
    multiplex.evolve_together(1, together = True, evolve_mode='fixed', position = 4)

    assert(multiplex.get_model(name = "first").network.nodes[(1, 1)]["grains"] == 0)
    assert(multiplex.get_model(name = "second").network.nodes[(1, 1)]["grains"] == 0)


def test_isolation_of_the_avalanche():
    '''
    Tests that the avalanche does not propagate from one layer of the multiplex network to the other when the 'together' parameter
    is False

    GIVEN: a sandpile model on a double-layer multiplex network, with a 3 x 3 grid for each layer. Both of them have 3 grains on the central
    node and 0 grains on the other nodes. The treshold is 4 for all nodes.
    WHEN: I evolve for one step, adding a grain on the central node of one of the two layers, with 'together' = False
    THEN: the toppling occurs only in one of the two layers, so the sum of the grains on the central nodes of the two layers is 3 (in
    one layer the central node has 3 grains and in the other layer it has 0 grains)
    '''
    model_1 = SandNet.Model(N = 3, initial_grains='zero')
    model_1.change_grains(index = 4, grains = 3)
    model_2 = SandNet.Model(N = 3, initial_grains='zero')
    model_2.change_grains(index = 4, grains = 3)
    multiplex = SandNet.Multiplex([model_1, model_2], ["first", "second"])
    multiplex.evolve_together(1, together = False, evolve_mode='fixed', position = 4)

    grains_first_layer = multiplex.get_model(name = "first").network.nodes[(1, 1)]["grains"]
    grains_second_layer = multiplex.get_model(name = "second").network.nodes[(1, 1)]["grains"]

    assert(grains_first_layer + grains_second_layer == 3)


def test_toppling_with_grains_lower_than_threshold():
    '''
    Tests that, if in a sandpile model on a multiplex network an avalanche can propagate from one layer to the others, than a node can topple
    even if it has a number of grains lower than its threshold

    GIVEN: a sandpile model on a double-layer multiplex network, with a 3 x 3 grid for each layer. Both of them have 3 grains on the central
    node and 0 grains on the other nodes. The treshold is 4 for all nodes, except for the threshold of the central node of the second layer
    which is set to 1000.
    WHEN: I evolve for 100 steps, adding all grains on the central node of both layers
    THEN: even if the threshold of the central node of the second layer is too high to allow an avalanche, we have some non-zero avalanche
    sizes in that layer, due to the propagation of the avalanches starting from the first layer

    Note: this test can fail in the extremely unlikely case that all 100 grains are added to the second layer. Even if this is
    almost impossible, we set the seed for random number generation to be sure that this will never happen.
    '''
    model_1 = SandNet.Model(N = 3, initial_grains='zero')
    model_1.change_grains(index = 4, grains = 3)
    model_2 = SandNet.Model(N = 3, initial_grains='zero')
    model_2.change_grains(index = 4, grains = 3)
    model_2.change_threshold(indexes = [4], threshold = 1000)
    multiplex = SandNet.Multiplex([model_1, model_2], ["first", "second"], seed = 42)

    multiplex.evolve_together(100, together = True, evolve_mode='fixed', position = 4)
    size_counter = Counter(multiplex.get_model(name = "second").get_avalanche_sizes())

    assert(size_counter[0] < 100) #not all avalanche sizes are 0
    assert(size_counter[1] >= 1) #at least one avalanche of size 1 happened


def test_toppling_with_zero_grains():
    '''
    Tests that if a node topples while it has 0 grains, the network remains unchanged.

    This is a test for the _avalanche method of the Model class. However, it has been put here among multiplex evolution tests because
    this situation can never happen while evolving a sandpile model on a simplex network, since a threshold of 0 is not allowed. On the contrary,
    in a multiplex network it can happen that a node topples while it has 0 grains, if this toppling starts because the corresponding node
    on another layer has toppled. This test verifies that the _avalanche method of the Model class can correctly handle these kind of situation.

    GIVEN: a sandpile model on a 3 x 3 grid network, with a fixed threshold of 4. The central node has 0 grains, while all the other
    nodes have 3 grains
    WHEN: I force the central node to topple using the _avalanche_together method of the Multiplex class
    THEN: the number of grains on each node remains unchanged, and the avalanche size and avalanche area corresponding to this avalanche are
    both 0
    '''
    model = SandNet.Model(N = 3, initial_grains= 'zero')

    for index in [0, 1, 2, 3, 5, 6, 7, 8]:
        model.change_grains(index = index, grains = 3)
    #change the threshold of all nodes except for the central one
    
    model._avalanche(model.select_node_by_index(4), lose_probability = None, step = 0, avalanche_matrix = False)

    assert(model.network.nodes[(1, 1)]["grains"] == 0)
    assert(model.network.nodes[(0, 1)]["grains"] == 3)
    assert(model.avalanche_size == 0)
    assert(model.avalanche_area == 0)


def test_generic_case_of_multiplex_evolution():
    '''
    Tests the correct evolution of the sandpile model over a multiplex network in a generic case. The calculation of the correct evolution
    of the system has been done by hand

    GIVEN: a sandpile model over a multiplex network of 9 nodes with three layers. The first layer is a 3 x 3 grid, with a fixed threshold
    of 4 and 3 grains for each node. The second layer is a network with a central node (of index 4, the same of the central node of the
    first layer), and all the other nodes connected only to the central node, except for node 8 which has no links. The central node has
    a threshold of 8 and 7 grains, while all the others have a threshold of 1 and 0 grains. The third layer is a 3 x 3 grid with zero grains
    on each node
    WHEN: I evolve for one step, putting one grain on the central node of the first layer
    THEN: in the first layer all nodes topple once, except for the central node which topples twice, so the avalanche size is 10 and the
    avalanche area is 9. There is a total of 16 grains left in the system. In the second layer all nodes topple twice, except for node 8 
    which never topples, so the avalanche size is 16 and the avalanche area is 8. There are 7 grains in the system, all on the central node.
    In the third layer nothing happens, so avalanche size and area are 0, as well as the grains on each node
    '''
    model_1 = SandNet.Model(N = 3)
    for index in range(9):
        model_1.change_grains(index = index, grains = 3)
    
    G = nx.Graph()
    G.add_nodes_from(range(9))
    G.add_edges_from([(0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (6, 4), (7, 4)])
    model_2 = SandNet.Model(G, threshold = 1)
    model_2.change_threshold(indexes = [4], threshold = 8)
    model_2.change_grains(index = 4, grains = 7)

    model_3 = SandNet.Model(N = 3)

    multiplex = SandNet.Multiplex([model_1, model_2, model_3], ["first", "second", "third"])
    multiplex.evolve_together(1, together = True, evolve_mode='fixed', position = 4, layer = 0, avalanche_matrix=True)

    assert(model_1.avalanche_sizes_collector[0] == 10)
    assert(model_1.avalanche_areas_collector[0] == 9)
    assert(model_1.avalanche_matrix[0, 4] == 2)
    assert(model_1.avalanche_matrix[0, 0] == 1)
    assert(model_1.get_total_grains() == 16)

    assert(model_2.avalanche_sizes_collector[0] == 16)
    assert(model_2.avalanche_areas_collector[0] == 8)
    assert(model_2.avalanche_matrix[0, 4] == 2)
    assert(model_2.avalanche_matrix[0, 0] == 2)
    assert(model_2.avalanche_matrix[0, 8] == 0)
    assert(model_2.get_total_grains() == 7)
    assert(model_2.network.nodes[model_2.select_node_by_index(4)]["grains"] == 7)
    
    assert(model_3.avalanche_sizes_collector[0] == 0)
    assert(model_3.avalanche_areas_collector[0] == 0)
    assert(model_3.get_total_grains() == 0)



