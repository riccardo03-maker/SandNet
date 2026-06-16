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
    Tests the raise of error when the input threshold rule is not in the list of possible values

    GIVEN: I am initializing a SandNet.Model object
    WHEN: the input value for the threshold_rule argument is not a standard one
    THEN: the code raises a ValueError
    '''
    with pytest.raises(ValueError):
        model = SandNet.Model(threshold_rule="not_standard_string")


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
