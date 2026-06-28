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
        models: list of SandNet.Model
            The sandpile models that have to be bound together in a multiplex network. A Multiplex object can
            bind together as many sandpile models as desired, provided that all network structures have the same
            number of nodes
        names: list of strings (default = None)
            The names that are assigned to each model stored. Useful to recognize and retrieve them more easily.
            If the number of names N provided is lower than the number of models M provided, only the first N models will receive a name
            If the number of names provided is higher than the number of models provided, the code raises a ValueError 
    Raises
    -------
        ValueError:
            If the network structures of the sandpile models provided do not have all the same number of nodes
        ValueError:
            If the number of names provided is higher than the number of models provided
    '''
    def __init__(self, models: list, names: list = None):
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
