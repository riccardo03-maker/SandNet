#!/usr/bin/python
# -*- coding: utf-8 -*-

from SandNet import Model
import numpy as np
import sys
from scipy.sparse import lil_matrix

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

__all__ =[
    'Multiplex'
]


class Multiplex():
    '''
    This class allows to bind different sandpile models on networks to create a sandpile model on a 
    multiplex network structure

    Parameters
    ----------
        models: list of SandNet.Model
            The sandpile models that have to be bound together in a multiplex network. A Multiplex object can
            bind together as many sandpile models as desired, provided that all network structures have the same
            number of nodes
        names: list of strings (default = None)
            The names that are assigned to each model stored. Useful to recognize and retrieve them more easily.
            If the number of names N provided is lower than the number of models M provided, only the first N models will receive a name
            If the number of names provided is higher than the number of models provided, the code raises a ValueError
        seed: int (default: None)
            The seed for random number generation. If provided, this seed is used for all the methods of the class that generate
            random numbers.
    Raises
    -------
        ValueError:
            If the network structures of the sandpile models provided do not have all the same number of nodes
        ValueError:
            If the number of names provided is higher than the number of models provided
    '''
    def __init__(self, models: list, names: list = None, seed: int = None):
        self.all_models = models

        if names is not None:
            self.model_names = dict(zip(names, list(range(len(names)))))
            if len(names) > len(models):
                raise ValueError("Provided more names than models")
        else:
            self.model_names = {}
            #initialize an empty vocabulary in case models with names are added later with the add_model method

        number_of_nodes = np.array([model.get_number_of_nodes() for model in self.all_models])
        if(len(np.unique(number_of_nodes)) > 1):
            raise ValueError("Provided networks do not have all the same number of nodes")
        
        self.current_model = self.all_models[0]
        #this is the current model we can modify using Model class methods. A method of the Multiplex class allows
        #to switch the current model

        self.current_model_index = 0
        #this is the index of the current model in the list of models stored

        self.seed = seed
    

    def get_model(self) -> Model:
        '''
        Returns the model stored in the current_model attribute

        This method allows to get one of the models stored in a Multiplex instance, and then to modify it using
        the methods of the Model class. The model that is returned (the current model) is always the first one in the list of models
        provided as input during the initialization, unless the current model has been modified using the change_current_model method

        Returns
        ----------
            current_model: SandNet.Model
                The sandpile model stored in the current_model attribute
        '''
        return self.current_model


    def change_current_model_by_index(self, model_position: int):
        '''
        Changes the current model, selecting the new current model through the index it has in the list of stored models

        This method allows to change the model that is returned by the get_model method, choosing from the stored models

        Parameters
        ----------
            model_position: int
                The position of the model in the list of models stored in the current instance of this class.
                This position depends on the order the models were given as arguments during the initialization of the instance
                (the first argument is in position 0, the second in position 1 and so on)
        '''
        self.current_model = self.all_models[model_position]
        self.current_model_index = model_position
    

    def change_current_model_by_name(self, model_name: str):
        '''
        Changes the current model, selecting the new current model through its name

        This method allows to change the model that is returned by the get_model method, choosing from the stored models
        that have a corresponding name

        Parameters
        ----------
            model_name: str
                The name of the new current model
        '''
        model_position = self.model_names[model_name]
        self.current_model = self.all_models[model_position]
        self.current_model_index = model_position
    

    def add_model(self, model: Model, name: str = None):
        '''
        Add a model at the end of the list of models stored in the current instance of this class

        Parameters
        ----------
            model: SandNet.Model
                The sandpile model that has to be added to the list of models. The model is added as the last element
                of this list
            name: string (default: None)
                The name of the added sandpile model
        Raises
        -------
        ValueError:
            If the network structure of the new sandpile model does not have the same number of nodes of the network
            structures of the other models stored
        '''
        if(model.get_number_of_nodes() != self.current_model.get_number_of_nodes()):
            raise ValueError("Provided network has a different number of nodes with respect to the others")
        # just check one of the stored models, the other have the same number of nodes

        self.all_models.append(model)
        if name is not None:
            self.model_names.update({name : len(self.all_models) - 1})
    

    def evolve_together(self, steps: int, together: bool = True, evolve_mode: str = 'random', position: int = None, lose_probability: float = None, 
                        avalanche_matrix: bool = False):
        '''
        Evolves the sandpile model on multiplex networks for a certain number of steps

        This method is analog to the evolve method of the Model class, but it evolves together all the different layers of the multiplex
        network. At each time step, a grain is added to a node of one of the layers of the multiplex network. The layer to which the grain
        is added is selected randomly at each time step.

        Moreover, it is possible to choose whether an avalanche starting in one layer also propagates to the other layers or not

        Parameters
        ----------
            steps: int
                The number of steps of evolution of the model (corresponds to the total number of grains added)
            together: bool (default = True):
                If True, when a node topples in one layer it also topples in the other layers. If False, avalanches in different
                layers do not influence each other
            evolve_mode: {'fixed', 'random'} (default = 'random')
                Sets how the new grains are added
                    fixed: all grains are added in the same position, choosen according to the position parameter (see below)
                    random: grains at each step are added in random positions
                
                    Note: indipendently on the evolve_mode parameter, the layer in which the grain is added is chosen randomly at each time
                    step. So if the evolve mode is fixed, the grain will always be added to the same position, but in different layers at
                    each time step
            position: int (default: None)
                The index of the node at which new grains are added.
                If the evolve_mode is 'random', this parameter is ignored.
                If the evolve mode is 'fixed' and position is None, one random position is selected
            lose_probability: float (default: None)
                The probability that a grain is lost during a toppling. This is useful if the network has no boundaries
                to avoid the lock of the system into an avalanche of infinite size.
                If this parameter is None, grains will never be lost during a toppling.
                If this parameter is lower than 0 or greater than 1, the code raises a ValueError
            avalanche_matrix: bool (default: False)
                If True, calculates also avalanche area and number of topplings per node at each avalanche (avalanche matrix). These measures are
                calculated separately for each layer

                Note: if these quantities are not needed, it is strongly suggested to keep this parameter to its default value
                of False, because in this way the simulation is much faster
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
        
        if(avalanche_matrix):
            #initialize the avalanche matrix for each layer: each row is a step of evolution and each column is a node
            for model in self.all_models:
                model.avalanche_matrix = lil_matrix((steps, model.get_number_of_nodes()), dtype = np.int32)

        #for both 'fixed' and 'random' evolutions, we create a list of indexes of nodes
        #where the grain will be added at each step.
        #So in the first case we will have a list of the length of the steps with always the same element.
        #This is useful to write the same code in both the evolution modes
        if(evolve_mode == 'fixed'):
            if(position >= self.get_model().get_number_of_nodes()):
                raise ValueError("Position selected does not exist")
            if(position is None):
                position = np.random.randint(0, self.get_model().get_number_of_nodes())
            grain_positions = [position] * steps
        elif(evolve_mode == 'random'):
            grain_positions = list(np.random.randint(0, self.get_model().get_number_of_nodes(), size = steps))
        else:
            raise ValueError("Invalid evolution rule")

        #create then a list of layers where the grain will be added at each time step
        #At each time step the grain will be added in the position given by grain_position in the layer given by grain_layer
        grain_layer = list(np.random.randint(0, len(self.all_models), size = steps))
        
        #Python has a default recursion limit of 1000. If more than 1000 functions are called at the same time,
        #Python gives a RecursionError. Since we can have very large avalanches for very large networks, we need
        #to increase this limit. We consider the dimension of the network as the product between the number of nodes and the number
        #of layers. If this dimension is lower than 500, we leave the default recursion limit, otherwise we increase it to two times
        #this dimension. If the method _avalanche_together is called a higher number of times than this limit, it is reasonable to 
        #think that the system is trapped into an avalanche of infinite size, and then it is correct to raise a RecursionError
        multiplex_network_dimension = self.get_model().get_number_of_nodes() * len(self.all_models)
        if(multiplex_network_dimension > 500):
            sys.setrecursionlimit(2 * multiplex_network_dimension)

        for step, index in enumerate(grain_positions):
            self.change_current_model_by_index(grain_layer[step])
            current_model = self.get_model()
            #this assignment in Python is just a reference, not a copy of the variable, so by modifying the current_model variable
            #also the corresponding model stored in the current Multiplex instance is modified 

            node = current_model.select_node_by_index(index)
            current_model.network.nodes[node]["grains"] += 1

            if(current_model.network.nodes[node]["grains"] >= current_model.network.nodes[node]["threshold"]):
                if(together):
                    self._avalanche_together(node, lose_probability, step, avalanche_matrix)
                else:
                    current_model._avalanche(node, lose_probability, step, avalanche_matrix)
                    #if 'together' is False, the avalanche is not propagated to other layers, so the _avalanche method of the Model
                    #class is enough to compute the avalanche

            for model in self.all_models:
                model.avalanche_sizes_collector.append(model.avalanche_size) #avalanche size is 0 if no node toppled
                model.avalanche_size = 0
                model.avalanche_areas_collector.append(model.avalanche_area)
                model.avalanche_area = 0
        
        #set the original recursion limit in case the size of the network is changed in successive calls of this function
        sys.setrecursionlimit(1000)


    def _avalanche_together(self, node, lose_probability: float, step: int, avalanche_matrix: bool):
        '''
        Computes an avalanche, and propagates it to the other layers

        After adding a grain to a node of one of the layers of the multiplex network, if the number of grains on that node is higher or equal
        to its threshold, the node topples. The node loses grains equal to the threshold and these grains are given to the neighbours
        (one grain for each neighbour). This process can make other nodes topple, generating a cascade process called avalanche
        If the number of neighbours of the toppling node is 0, the code stops.
        If the number of neighbours of the toppling node is less than the threshold, one grain is given to each neighbour and the others are lost.
        If the number of neighbours of the toppling node is higher than the threshold, only a number of neighbours
        equal to the threshold chosen randomly will receive a grain.

        The avalanche then propagates to all the other layers of the multiplex network. When a node topples on the initial layer to which
        the grain was added, all the nodes with the same index on the other layers topple as well, even if the number of grains they have
        is lower than their threshold.

        For sake of simplicity, only the topplings in the main layer (the one where the grain was added) can create topplings in the other layers.
        If a toppling in one of the non-main layers causes a toppling of one of the neighbouring nodes, that toppling is not propagated
        to the other layers
        
        Parameters
        ----------
            node:
                The node that is toppling
                Note: there is no specific type for the output, since the nodes of a nx.Graph can be of any type
            lose_probability: float
                The probability that a grain is lost during a toppling. This parameter is directly passed from
                the evolve_together function
            step: int
                The running step of evolution. This parameter is directly passed from the evolve_together function
            avalanche_matrix: bool
                If True, calculates also avalanche area and number of topplings per node at each avalanche (avalanche matrix).
                This parameter is directly passed from the evolve function

        Raises
        -------
            Exception:
                If the toppling node has no neighbours
        Note: this function can also raise a RecursionError when the system remains trapped in a loop (i.e. an avalanche
        of infinite size). However, since the avalanche dynamics is implemented as a recursive function, Python will
        handle this situation automatically giving a RecursionError, so there is no need to implement it.
        '''
        other_models = [model for i, model in enumerate(self.all_models) if i != self.current_model_index]
        current_model = self.get_model()
        #the current model is still the one where the grain was added at the current time step

        #first propagate the toppling to other layers. Starting from the other layers is simpler because they cannot propagate
        #eventual other topplings to the other layers
        toppling_node_index = current_model.network.nodes[node]["index"]
        for model in other_models:
            toppling_node = model.select_node_by_index(toppling_node_index)
            #the node that topples in the other layers is the node with the same index of the toppling node in the main layer

            model._avalanche(toppling_node, lose_probability, step, avalanche_matrix)
            #topplings on non-main layers cannot be propagated, so the _avalanche method of the Model class is enough to compute the avalanche

        neighbours = current_model._select_neighbours(node)
        if not neighbours:
            raise Exception("Grains added on a node with no neighbours")
        
        #avalanche_size and avalanche_area are not local variables because _avalanche method is recursive, so they would
        #be resetted to 0 at each call of the method
        current_model.avalanche_size += 1
        if(avalanche_matrix):
            toppling_node_index = current_model.network.nodes[node]["index"]
            current_model.avalanche_matrix[step, toppling_node_index] += 1
            if(current_model.avalanche_matrix[step, toppling_node_index] == 1):
                current_model.avalanche_area += 1
            #avalanche area must be increased only the first time a node topples in an avalanche, because it only
            #counts the total number of node that have toppled

        current_model.network.nodes[node]["grains"] -= current_model.network.nodes[node]["threshold"]
        if(current_model.network.nodes[node]["grains"] < 0):
            current_model.network.nodes[node]["grains"] = 0
            #avoid negative grains in case toppling occurs with grains lower than threshold

        if(lose_probability is not None):
            is_grain_passed = list(np.random.uniform(size = len(neighbours)) > lose_probability)
            #if one element of this list is False, the corresponding neighbour of the toppling node will not receive a grain
        else:
            is_grain_passed = [True] * len(neighbours)

        for i, neighbour in enumerate(neighbours):
            if(is_grain_passed[i]):
                current_model.network.nodes[neighbour]["grains"] += 1
                if(current_model.network.nodes[neighbour]["grains"] >= current_model.network.nodes[neighbour]["threshold"]):
                    self._avalanche_together(neighbour, lose_probability, step, avalanche_matrix)
            #we can use the recursion because of the Abelian property of sandpile model: avalanche dynamics
            #does not depend on the order of topplings


if(__name__ == '__main__'):
    pass
