#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import sys

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


def _set_grains(network : nx.Graph, initial_grains : str, seed: int) -> nx.Graph:
    '''
    Set the initial number of grains of sand

    Parameters
    ----------
        network: nx.Graph
            The newtork structure of the sandpile model
        initial_grains: {'zero', 'random'}
            Selects how the initial number of grains is chosen:
                zero: all nodes have 0 grains at the beginning
                random: the initial number of grains for each node is a random integer between 0 and the threshold of the node minus one
        seed: int
            The seed for random number generation for the 'random' initial grains mode.
            If initial grains mode is 'zero', this parameter is ignored
    Returns
    -------
        network: nx.Graph
            The initial network but with initial number of grains set
    '''
    nodes = list(network.nodes)
    if(initial_grains == 'zero'):    
        for node in nodes:
            network.nodes[node]["grains"] = 0
    elif(initial_grains == 'random'):
        np.random.seed(seed)
        for node in nodes:
            network.nodes[node]["grains"] = np.random.randint(0, network.nodes[node]["threshold"])
    else:
        raise ValueError("Wrong input for initial_grains")
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
        initial_grains: {'zero', 'random'} (default: 'zero')
            Selects how the initial number of grains for each node is chosen:
                zero: all nodes have 0 grains at the beginning
                random: the initial number of grains for each node is a random integer between 0 and the threshold of the node minus one
        seed: int (default: 42)
            The seed for random number generation. This seed is used for all the methods of the class that generate
            random numbers.
    '''

    def __init__(self, network: nx.Graph = None, N = 5, threshold_rule: str = "fixed", threshold: int = 4, initial_grains: str = 'zero', seed: int = 42):
        if network is None:
            #create a 2d square lattice using the right NetworkX function
            self.network = nx.grid_2d_graph(N, N)
        else:
            self.network = network

        self.seed = seed
        #the seed for random number generation is a parameter of the class, used in all the class methods
        #that generate random numbers

        self.network = _set_threshold(self.network, threshold_rule, threshold)
        self.network = _set_grains(self.network, initial_grains, self.seed)

        #set an integer index to each node to identify it (will be useful during the evolution phase)
        self.network = _set_index(self.network)

        #define now a variable to measure avalanche size during the evolution (see _avalanche method)
        self.avalanche_size = 0
        #and a list with the sizes of all avalanches
        self.avalanche_sizes_collector = []
        

    def select_node_by_index(self, index: int):
        '''
        Returns a network node starting from its index

        Given a sandpile model with a network structure, where each node has an attribute called index
        (choosen during the initialization), this function allows to select a node starting from the index
        
        Parameters
        ----------
            index: int
                The index of the node to be selected
        Returns
        -------
            node:
                The node corresponding to the index.
                Note: there is no specific type for the output, since the nodes of a nx.Graph can be of any type
        Raises
        -------
            ValueError:
                If there is no node with the input index
        '''
        if(index>=len(self.network.nodes)):
            raise ValueError("No node with the selected index")

        #use the next function because we have only one node for each index, so we can stop the search
        #at the first node we find with the correct index
        node = next(node for node in list(self.network.nodes) if self.network.nodes[node]["index"] == index)
        return node


    def select_nodes_by_degree(self, degree: int, raises = True) -> int:
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
                The returned index can be used as input for the function select_node_by_index
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
        
    
    def evolve(self, steps: int, evolve_mode = 'random', position: int = None, lose_probability: float = None):
        '''
        Evolves the sandpile model for a certain number of steps

        Parameters
        ----------
            steps: int
                The number of steps of evolution of the model (corresponds to the total number of grains added)
            evolve_mode: {'fixed', 'random'}
                Sets how the new grains are added
                    fixed: all grains are added in the same position, choosen according to the position parameter (see below)
                    random: grains at each step are added in random positions
            position: int (default: None)
                The index of the node at which new grains are added.
                If the evolve_mode is 'random', this parameter is ignored.
                If the evolve mode is 'fixed' and position is None, one random position is selected
            lose_probability: float (default: None)
                The probability that a grain is lost during a toppling. This is useful if the network has no boundaries
                to avoid the lock of the system into an avalanche of infinite size.
                If this parameter is None, grains will never be lost during a toppling.
                If this parameter is lower than 0 or greater than 1, the code raises a ValueError
        Raises
        -------
            ValueError:
                If the evolve mode is not an accepted one
            ValueError:
                If the position selected does not exist
            ValueError:
                If the probability to lose a grain is not between 0 and 1
        '''
        np.random.seed(self.seed)
        if(lose_probability is not None and (lose_probability < 0. or lose_probability > 1.)):
            raise ValueError("Not accepted value of probability")

        #for both 'fixed' and 'random' evolutions, we create a list of indexes of nodes
        #where the grain will be added at each step.
        #So in the first case we will have a list of the length of the steps with always the same element.
        #This is useful to write the same code in both the evolution modes

        if(evolve_mode == 'fixed'):
            if(position >= len(self.network.nodes)):
                raise ValueError("Position selected does not exist")
            if(position is None):
                position = np.random.randint(0, len(self.network.nodes))
            grain_positions = [position] * steps
        elif(evolve_mode == 'random'):
            grain_positions = np.random.randint(0, len(self.network.nodes), size = steps)
        else:
            raise ValueError("Invalid evolution rule")
        
        #Python has a default recursion limit of 1000. If more than 1000 functions are called at the same time,
        #Python gives a RecursionError. Since we can have very large avalanches for very large networks, we need
        #to increase this limit. So, if we have a network of less than 500 nodes we don't modify this limit, while if
        #we have more nodes we increase the limit to two times the number of nodes. If an avalanche is larger than this value,
        #it is reasonable to believe that the network is trapped into a loop, so then it is correct to raise a RecursionError
        if(len(self.network.nodes) > 500):
            sys.setrecursionlimit(2 * len(self.network.nodes))
        
        for index in grain_positions:
            node = self.select_node_by_index(index)
            self.network.nodes[node]["grains"] += 1
            if(self.network.nodes[node]["grains"] >= self.network.nodes[node]["threshold"]):
                self._avalanche(node, lose_probability)
            self.avalanche_sizes_collector.append(self.avalanche_size) #avalanche size is 0 if no node toppled
            self.avalanche_size = 0
        
        #set the original recursion limit in case the size of the network is changed in successive calls of this function
        sys.setrecursionlimit(1000)
            

    def _avalanche(self, node, lose_probability):
        '''
        Computes an avalanche

        When the grains on a node reach the threshold of that node, the node topples. The node loses grains equal to the threshold
        and these grains are given to the neighbours (one grain for each neighbour). This process can make other nodes topple, generating a cascade process called avalanche
        If the number of neighbours of the toppling node is 0, the code stops.
        If the number of neighbours of the toppling node is less than the threshold, one grain is given to each neighbour and the others are lost.
        If the number of neighbours of the toppling node is higher than the threshold, so that it is not clear to which
        neighbours the grains should be given
        
        Parameters
        ----------
            node:
                The node that is toppling
                Note: there is no specific type for the output, since the nodes of a nx.Graph can be of any type
            lose_probability: float
                The probability that a grain is lost during a toppling. This parameter is directly passed from
                the evolve function

        Raises
        -------
            Exception:
                If the toppling node has no neighbours
            Exception:
                If the toppling node has a number of neighbours higher than its threshold
        Note: this function can also raise a RecursionError when the system remains trapped in a loop (i.e. an avalanche
        of infinite size). However, since the avalanche dynamics is implemented as a recursive function, Python will
        handle this situation automatically giving a RecursionError, so there is no need to implement it.
        '''
        neighbours = list(self.network[node])
        if not neighbours:
            raise Exception("Grains added on a node with no neighbours")
        if(len(neighbours) > self.network.nodes[node]["threshold"]):
            raise Exception("Number of neighbours higher than threshold")
        
        #avalanche_size is not a local variable because _avalanche method is recursive, so it would
        #be resetted to 0 at each call of the method
        self.avalanche_size += 1

        self.network.nodes[node]["grains"] -= self.network.nodes[node]["threshold"]

        if(lose_probability is not None):
            is_grain_passed = list(np.random.uniform(size = len(neighbours)) > lose_probability)
            #if one element of this list is False, the corresponding neighbour of the toppling node will not receive a grain
        else:
            is_grain_passed = [True] * len(neighbours)

        for i, neighbour in enumerate(neighbours):
            if(is_grain_passed[i]):
                self.network.nodes[neighbour]["grains"] += 1
                if(self.network.nodes[neighbour]["grains"] >= self.network.nodes[neighbour]["threshold"]):
                    self._avalanche(neighbour, lose_probability)
            #we can use the recursion because of the Abelian property of sandpile model: avalanche dynamics
            #does not depend on the order of topplings

    
    def get_number_of_nodes(self) -> int:
        '''
        Return the number of nodes in the sandpile model network

        Returns:
        ----------
            count: int
                The number of nodes in the network
        '''
        return len(list(self.network.nodes))
    

    def get_node_degree(self, index: int) -> int:
        '''
        Return the degree of a node in the sandpile model network

        Parameters:
        ----------
            index: int
                The index of the node whose degree we want to know
        Returns:
        ----------
            degree: int
                The degree of the selected node
        '''
        return self.network.degree[self.select_node_by_index(index)] 
    

    def find_boundaries(self) -> list:
        '''
        Finds the boundaries of the sandpile model network

        The boundaries of the network structure of the sandpile model are given by all the nodes with a threshold higher than
        their degree. So, the boundaries are the nodes able to decrease the number of grains in the system after toppling.
        For example, the boundaries for a lattice network are given by the sides of the grid. This function gives all the 
        indexes of the boundary nodes (defined as above) for any kind of network structure.

        Returns
        ----------
            indexes: list of integers
                The indexes of the boundary nodes. If no boundary nodes are present, the function just returns
                an empty list
        '''
        boundaries = []
        for node in list(self.network.nodes):
            if(self.network.nodes[node]["threshold"] > self.network.degree[node]):
                boundaries.append(self.network.nodes[node]["index"])
        
        return boundaries


    def add_boundaries(self, index_list: list = None, n_boundaries: int = None):
        '''
        Adds boundaries to the sandpile model network

        The boundaries of the network structure of the sandpile model are given by all the nodes with a threshold higher than
        their degree. So, the boundaries are the nodes able to decrease the number of grains in the system after toppling.
        This function can turn some nodes into boundaries. These nodes can be either selected by the user (by giving
        the indexes as input) or chosen automatically by the function. In this latter case, the function will rank
        the nodes according to the degree and choose among the nodes with lowest degree those that will become boundaries.

        Note: this function turns nodes into boundaries by just increasing their threshold by 1, so that during a toppling
        starting from that node one grain will be lost. It is like a link with the "external" is added. So the function can
        also act to nodes that are already boundaries, by just adding one more external link.

        Parameters
        ----------
            index_list: list of integers
                The list of indexes of the nodes that will become boundaries
            n_boundaries: int
                The number of nodes to be transformed into boundaries. If index_list is provided, this parameter is ignored
        '''
        if(index_list is not None):
            for index in index_list:
                self.network.nodes[self.select_node_by_index(index)]["threshold"] += 1
        
        elif(n_boundaries is not None):
            indexes = list(range(self.get_number_of_nodes()))
            indexes.sort(key = self.get_node_degree)
            for index in indexes[:n_boundaries]:
                self.network.nodes[self.select_node_by_index(index)]["threshold"] += 1
    

    def change_threshold(self, indexes: list, threshold: int):
        '''
        Changes the threshold of the nodes corresponding to the indexes given as input

        Parameters:
        ----------
            indexes: list of integers
                The indexes of the nodes for which we want to change the threshold
            threshold: int
                The new value of threshold that will be assigned to the selected nodes
        '''
        for index in indexes:
            self.network.nodes[self.select_node_by_index(index)]["threshold"] = threshold
        

if(__name__ == '__main__'):
    #calculate the time required to run the evolution
    from time import time as now
    model = Model(N = 100, initial_grains='random')

    tic = now()
    model.evolve(50000)
    toc = now()

    print("50000 steps of evolution in " + str(toc-tic) + " seconds")