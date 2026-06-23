#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import SandNet
import numpy as np
from scipy.stats import pareto
import networkx as nx

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

__all__ =[
    'fit_avalanche_size'
]


def _cut_zeros(vector: list) -> list:
    '''
    Removes all the zeros from a list of integers

    Parameters
    ----------
        vector: list of integers
            The input list from which we want to remove zeros

    Returns
    -------
        vector: list of integers
            The input list without zeros. The size of the output list will then be different from the input one.
            If the function does not find any zero, it will just return the input vector
    '''
    no_zero_vector = [i for i in vector if i != 0]
    return no_zero_vector


def fit_avalanche_size(model: SandNet.Model, fit_exponent: float = None, plot: bool = True):
    '''
    Fit and plot avalanche sizes of the sandpile model

    Given a sandpile model object, this function takes the list of all the avalanche sizes and fits them to a 
    power law distribution, using in particular the Pareto distribution from the scipy.stats package.
    After the fit, the function plots avalanche sizes together with the function of best fit in a logarithmic scale.

    Parameters
    ----------
        model: SandNet.Model
            The sandpile model for which we want to fit and plot avalanche sizes

    Returns
    -------
        fit_parameter: float
            The exponent of the power law obtained through fitting

            Note: since in the Pareto distribution used for fitting the exponent is $b+1$ (see scipy.stats.pareto documentation 
            for more details), the parameter obtained through fitting is $b$, and so the parameter returned by this function is not
            the exponent of the power law, but instead the exponent decreased by 1. We prefer not to adjust the returned parameter
            by adding 1 to avoid adding more uncertainty to the parameter due to floating point error
    '''
    avalanche_sizes = _cut_zeros(model.avalanche_sizes_collector)
    fit_parameter, loc, scale = pareto.fit(avalanche_sizes, floc = 0, fscale = 1)
    #fix location and scale parameters for better fit

    if plot:

        #use np.histogram and plt.scatter to have a plot of points instead of a histogram
        y, x = np.histogram(avalanche_sizes, bins = np.logspace(0, np.log10(max(avalanche_sizes))), density=True)
    
        #the last element of y includes the last two bins of x. So x has one more element than y, which we have to cut out
        plt.scatter(x[:-1], y, label = "data points")

    
        y_fit = pareto.pdf(x[:-1], fit_parameter, loc=loc, scale=scale)
        plt.plot(x[:-1], y_fit, label = "power law fit", color = "red", linestyle = "--")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Avalanche size")
        plt.ylabel("Density")
        plt.title("Power law distribution of avalanche sizes")
        plt.legend()
        plt.show()
    
    return fit_parameter


if(__name__ == '__main__'):

    #execute the model on a 100x100 square grid and plot avalanche size and avalanche duration distributions
    model = SandNet.Model(N=100, initial_grains='random')
    model.evolve(30000)

    fit_parameter = fit_avalanche_size(model)
    tau = fit_parameter
    print("Avalanche size exponent: " + str(tau))