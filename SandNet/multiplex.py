#!/usr/bin/python
# -*- coding: utf-8 -*-

from SandNet import Model
import numpy as np

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
        *models: SandNet.Model
            The sandpile models that have to be bound together in a multiplex network. A Multiplex object can
            bind together as many sandpile models as desired, provided that all network structures have the same
            number of nodes
    Raises
    -------
        ValueError:
            If the network structures of the sandpile models provided do not have all the same number of nodes
    '''
    def __init__(self, *models: Model):
        self.all_models = [*models]

        number_of_nodes = np.array([model.get_number_of_nodes() for model in self.all_models])
        if(len(np.unique(number_of_nodes)) > 1):
            raise ValueError("Provided networks do not have all the same number of nodes")
        
        self.current_model = self.all_models[0]
        #this is the current model we can modify using Model class methods. A method of the Multiplex class allows
        #to switch the current model
    

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


    def change_current_model(self, model_position: int):
        '''
        Changes the model stored in the current_model attribute

        This method allows to change the model that is returned by the get_model method, choosing from the models
        provided as input during the initialization

        Parameters
        ----------
            model_position: int
                The position of the model in the list of models stored in the current instance of this class.
                This position depends on the order the models were given as arguments during the initialization of the instance
                (the first argument is in position 0, the second in position 1 and so on)
        '''
        self.current_model = self.all_models[model_position]

    
    def add_model(self, model: Model):
        '''
        Add a model at the end of the list of models stored in the current instance of this class

        Parameters
        ----------
            model: SandNet.Model
                The sandpile model that has to be added to the list of models. The model is added as the last element
                of this list
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
