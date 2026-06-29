#!/usr/bin/python
# -*- coding: utf-8 -*-

import SandNet
import pytest
import networkx as nx

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


# Testing initialization of multiplex network model

def test_networks_with_different_length():
    '''
    Tests the raise of a ValueError when the sandpile models provided as input for the multiplex model initialization
    do not have the same number of nodes

    GIVEN: I am initializing a sandpile model on a multiplex network
    WHEN: I give as input three models, the first two on a 2 x 2 grid network and the third on a 3 x 3 grid network
    THEN: the code raises a ValueError
    '''
    model_1 = SandNet.Model(N = 2)
    model_2 = SandNet.Model(N = 2)
    model_3 = SandNet.Model(N = 3)
    with pytest.raises(ValueError):
        SandNet.Multiplex([model_1, model_2, model_3])


def test_longer_list_of_names():
    '''
    Tests the raise of a ValueError when the sandpile models provided as input for the multiplex model initialization
    are less than the number of names provided

    GIVEN: I am initializing a sandpile model on a multiplex network
    WHEN: I give as input three models and four names
    THEN: the code raises a ValueError
    '''
    model = SandNet.Model()
    with pytest.raises(ValueError):
        SandNet.Multiplex([model, model, model], names = ['a', 'b', 'c', 'd'])


def test_multiplex_initialization():
    '''
    Tests the correct initialization of a sandpile model on a multiplex network

    GIVEN: I am initializing a sandpile model on a multiplex network
    WHEN: I give as input five identical instances of class Model
    THEN: the attribute all_models of the multiplex model initialized has length 5
    '''
    model = SandNet.Model()
    multiplex = SandNet.Multiplex([model, model, model, model, model])
    assert(len(multiplex.all_models) == 5)


def test_correct_name_initialization():
    '''
    Tests the correct initialization of the names of sandpile models stored in an instance of the Multiplex class

    GIVEN: I am initializing a sandpile model on a multiplex network
    WHEN: I give as input three models and two names
    THEN: the second name of the list corresponds to index 1 (which is the index of the second model in the list)
    '''
    model = SandNet.Model()
    multiplex = SandNet.Multiplex([model, model, model], ["a", "b"])
    assert(multiplex.model_names["b"] == 1)


# Test retrieve and change of current model


def test_correct_get_model():
    '''
    Tests the correct retrieve of the current model stored in an instance of the Multiplex class

    GIVEN: a sandpile model on a multiplex network, with three models stored. Each model is built on a network of
    three nodes, but in the first model the nodes are fully connected, in the second model there is only one connection
    for each node, and in the third model the nodes are disconnected
    WHEN: I get the degree of one of the nodes of the current model
    THEN: the obtained degree is 2 (the current model is chosen as the first model provided as argument during
    multiplex model initialization)
    '''
    G = nx.Graph()
    G.add_nodes_from(range(3))
    model_3 = SandNet.Model(G)

    G_1 = G.copy()
    G_1.add_edges_from([(0, 1), (1, 2)])
    model_2 = SandNet.Model(G_1)

    G_2 = G_1.copy()
    G_2.add_edges_from([(0, 2)])
    model_1 = SandNet.Model(G_2)

    multiplex = SandNet.Multiplex([model_1, model_2, model_3])
    assert(multiplex.get_model().get_node_degree(1) == 2)


def test_correct_change_of_current_model_by_index():
    '''
    Tests the correct change of the current model stored using the index of the model in the list of stored models

    GIVEN: a sandpile model on a multiplex network, with three models stored. Each model is built on a network of
    three nodes, but in the first model the nodes are fully connected, in the second model there is only one connection
    for each node, and in the third model the nodes are disconnected
    WHEN: I change the current model using the index, choosing the third model, and ask for the degree of one of its nodes
    THEN: the obtained degree is 0
    '''
    G = nx.Graph()
    G.add_nodes_from(range(3))
    model_3 = SandNet.Model(G)

    #need to copy the graph to another variable, otherwise also the network stored in model_3 is modified
    G_1 = G.copy()
    G_1.add_edges_from([(0, 1), (1, 2)])
    model_2 = SandNet.Model(G_1)

    G_2 = G_1.copy()
    G_2.add_edges_from([(0, 2)])
    model_1 = SandNet.Model(G_2)

    multiplex = SandNet.Multiplex([model_1, model_2, model_3])
    multiplex.change_current_model_by_index(2)
    assert(multiplex.get_model().get_node_degree(1) == 0)


def test_correct_change_of_current_model_by_name():
    '''
    Tests the correct change of the current model stored using the name of the model

    GIVEN: a sandpile model on a multiplex network, with three models stored. Each model is built on a network of
    three nodes, but in the first model the nodes are fully connected, in the second model there is only one connection
    for each node, and in the third model the nodes are disconnected. The three models are given three names
    WHEN: I change the current model using the name, choosing the second model, and ask for the degree of one of its nodes
    which is not the central one
    THEN: the obtained degree is 1
    ''' 
    G = nx.Graph()
    G.add_nodes_from(range(3))
    model_3 = SandNet.Model(G)

    #need to copy the graph to another variable, otherwise also the network stored in model_3 is modified
    G_1 = G.copy()
    G_1.add_edges_from([(0, 1), (1, 2)])
    model_2 = SandNet.Model(G_1)

    G_2 = G_1.copy()
    G_2.add_edges_from([(0, 2)])
    model_1 = SandNet.Model(G_2)

    multiplex = SandNet.Multiplex([model_1, model_2, model_3], names = ["a", "b", "c"])
    multiplex.change_current_model_by_name("b")
    assert(multiplex.get_model().get_node_degree(0) == 1)


def test_correct_change_of_index_of_current_model():
    '''
    Tests the correct change of the index of the current model when the current model is changed

    GIVEN: a sandpile model on a multiplex network with three identical layers of 5 x 5 grids
    WHEN: I change the current model to the second element of the list of stored models
    THEN: the index of the current model, which was initially 0, becomes 1
    '''
    model = SandNet.Model()
    multiplex = SandNet.Multiplex([model, model, model])
    assert(multiplex.current_model_index == 0)

    multiplex.change_current_model_by_index(1)
    assert(multiplex.current_model_index == 1)


# Test addition of new model


def test_addition_of_new_model():
    '''
    Tests the correct addition of a new stored sandpile model in an instance of the Multiplex class

    GIVEN: an instance of the Multiplex class with three identical models stored
    WHEN: I add a new model identical to the other models stored
    THEN: the attribute all_models of the multiplex model, which had initially length 3, has now length 4
    '''
    model = SandNet.Model(N = 3)
    multiplex = SandNet.Multiplex([model, model, model])
    assert(len(multiplex.all_models) == 3)

    multiplex.add_model(model)
    assert(len(multiplex.all_models) == 4)


def test_addition_in_last_position():
    '''
    Tests that the added model stored in an instance of the Multiplex class is positioned as the last element of
    the list with the stored models

    GIVEN: an instance of the Multiplex class with three identical models stored (on 3 x 3 grid networks)
    WHEN: I add a new model on a network with 9 disconnected nodes, and set the fourth element of the list of
    stored models as the current model
    THEN: the degree of any one of the nodes of the current model is 0
    '''
    model = SandNet.Model(N = 3)
    multiplex = SandNet.Multiplex([model, model, model])

    G = nx.Graph()
    G.add_nodes_from(range(9))
    new_model = SandNet.Model(G)
    multiplex.add_model(new_model)

    multiplex.change_current_model_by_index(3) #fourth element of the list
    assert(multiplex.get_model().get_node_degree(5) == 0)


def test_name_of_new_added_model():
    '''
    Tests that the name given to the added stored model effectively corresponds to that model

    GIVEN: an instance of the Multiplex class with three identical models stored and without names
    WHEN: I add a new model identical to the other models stored, but I give this one a name
    THEN: the name given to the new model corresponds to index 3 of the list of stored model (which is exactly the new model added)
    '''
    model = SandNet.Model(N = 3)
    multiplex = SandNet.Multiplex([model, model, model])

    multiplex.add_model(model, name = "a")
    assert(multiplex.model_names["a"]  == 3)


def test_addition_of_network_with_different_number_of_nodes():
    ''' 
    Tests the raise of a ValueError when a new model is added to the stored models of an instance of the Multiplex
    class, but the new model is on a network with different number of nodes with respect to the network structures
    of the other models

    GIVEN: an instance of the Multiplex class with three identical models stored
    WHEN: I add a new model on a network with a different number of nodes
    THEN: the code raises a ValueError
    '''
    model = SandNet.Model(N = 3)
    multiplex = SandNet.Multiplex([model, model, model])

    G = nx.Graph()
    G.add_nodes_from(range(8))
    new_model = SandNet.Model(G)
    with pytest.raises(ValueError):
        multiplex.add_model(new_model)