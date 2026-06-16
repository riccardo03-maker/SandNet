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
        