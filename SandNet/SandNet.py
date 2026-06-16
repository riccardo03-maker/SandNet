#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

__all__ =[
    'Model'
]

def _set_threshold(network : nx.Graph, threshold_rule: str, threshold: int) -> nx.Graph:
    '''
    Set thresholds for sandpile model

    If the number of grains on a node exceeds its threshold, the node topples 

    Parameters
    ----------
        network: nx.Graph
            The newtork structure of the sandpile model
        threshold_rule: {'fixed', 'degree'}
            The rule used to set thresholds
                fixed: each node has the same threshold
                degree: the threshold of each node is given by its degree
        threshold: int
            The threshold of each node for the 'fixed' rule. If the threshold rule is not 'fixed', this parameter is ignored

    Returns
    -------
        network: nx.Graph
            The initial network but with thresholds set
    Raises
    -------
        ValueError:
            If the input threshold rule is not an accepted one
    '''
    nodes = list(network.nodes)
    if(threshold_rule == "fixed"):
        for node in nodes:
            network.nodes[node]["threshold"] = threshold
    elif(threshold_rule == "degree"):
        for node in nodes:
            network.nodes[node]["threshold"] = network.degree(node)
    else:
        raise ValueError("Invalid threshold rule")
    return network


def _set_grains(network : nx.Graph) -> nx.Graph:
    '''
    Set the initial number of grains of sand (0 for each node) 

    Parameters
    ----------
        network: nx.Graph
            The newtork structure of the sandpile model

    Returns
    -------
        network: nx.Graph
            The initial network but with initial number of grains set
    '''

    nodes = list(network.nodes)
    for node in nodes:
            network.nodes[node]["grains"] = 0
    return network


def _set_index(network: nx.Graph) -> nx.Graph:
    '''
    Set a progressive index for each node in the network 

    Parameters
    ----------
        network: nx.Graph
            The newtork structure of the sandpile model

    Returns
    -------
        network: nx.Graph
            The initial network but with indexes set
    '''
    nodes = list(network.nodes)
    for i, node in enumerate(nodes):
            network.nodes[node]["index"] = i
    return network


class Model:
    '''
    This class defines a sandpile model on a network structure

    Parameters
    ----------
        network: nx.Graph (default: None)
            The network structure over which the sandpile model is implemented.
            If no value is given as input, the code will build a 2-dimensional square grid, 
            which is the classical sandpile model
        N: int (default: 5)
            The side of the square grid if the classical sandpile model is being implemented. This parameter is ignored if 
            a network structure is provided as argument for the previous parameter
        threshold_rule: {'fixed', 'degree'} (default: 'fixed')
            The rule used to choose the threshold of each node
                fixed: each node has the same threshold
                degree: the threshold of each node is given by its degree
        threshold: int (default: 4)
            The threshold of each node for the 'fixed' rule. If the threshold rule is not 'fixed', 
            this parameter is ignored

    '''

    def __init__(self, network: nx.Graph = None, N = 5, threshold_rule: str = "fixed", threshold: int = 4):
        if network is None:
            #create a 2d square lattice using the right NetworkX function
            self.network = nx.grid_2d_graph(N, N)
        else:
            self.network = network

        self.network = _set_threshold(self.network, threshold_rule, threshold)
        self.network = _set_grains(self.network)

        #set an integer index to each node to identify it (will be useful during the evolution phase)
        self.network = _set_index(self.network)
    

    def select_node_by_index(self, index: int, feature: str):
        '''
        Returns a node feature starting from its index

        Given a sandpile model with a network structure, where each node has an attribute called index
        (choosen during the initialization), this function allows to select a node feature starting from the index
        
        Parameters
        ----------
            index: int
                The index of the node to be selected
            feature: {'threshold', 'grains'}
                The desired output feature of the selected node
        Returns
        -------
            node:
                The selected feature for the node corresponding to the index.
                Note: there is no specific type for the output, since the output type is given by the feature selected
        Raises
        -------
            ValueError:
                If there is no node with the input index
            ValueErrore:
                If the input feature is not defined for that node
        '''
        if(index>=len(self.network.nodes)):
            raise ValueError("No node with the selected index")

        if(feature in ['threshold', 'grains']):
            #returns a list with one dictionary for each node, where the keys are the features
            #and the values are the values of those features
            node_features = list(dict(self.network.nodes(data=True)).values())

            #the next function is used so that 'attribute' is an integer instead of a list of a single integer
            attribute = next(node[feature] for node in node_features if node["index"] == index)
            return attribute
        else:
            raise ValueError("Selected feature does not exist")


    def select_nodes_by_degree(self, degree: int, raises = True):
        '''
        Select all network nodes with a certain degree

        Given a sandpile model with a network structure, this function allows to select a node
        starting from its degree. If more than one node has the same degree, the function will return a list
        of nodes with the selected degree.
        
        Parameters
        ----------
            degree: int
                The degree of the node to be selected
            raises: boolean (default: True)
                Defines the behaviour when no node has the selected degree. If True, it raises a ValueError (see below).
                If False, it just returns an empty list

        Returns
        -------
            node_index: int
                The index (or list of indexes) of the node(s) with the selected degree.
                The returned index can be used as input for the function select_node_by_index to extract a certain feature
        Raises
        -------
            ValueError:
                If there is no node with the selected degree
                Note: if the raises parameter is False, this error is never raised. This is useful since then
                the function can also be used to check if there are no nodes with a certain degree
        '''
        #returns a list with one dictionary for each node, where the keys are the features
        #and the values are the values of those features
        node_features = list(dict(self.network.nodes(data=True)).values())

        degree_list = list(self.network.degree)
        indexes = [node["index"] for i, node in enumerate(node_features) if degree_list[i][1] == degree]

        #transform a list into an integer if it has only one element,
        #otherwise the function would return a list of one element
        if(len(indexes) == 1):
            indexes = indexes[0]

        if raises and not indexes:
            raise ValueError("No node with the selected degree")
        return indexes
                     