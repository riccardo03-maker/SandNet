#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import SandNet
import numpy as np
from scipy.stats import pareto

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

__all__ =[
    'fit_powerlaw',
    'plot_avalanche_size',
    'plot_avalanche_area'
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


def plot_avalanche_area(model: SandNet.Model, figure = None, x_label: str = None, y_label: str = None, 
                        title: str = None, **kwargs):
    '''
    Plots a scatter plot of avalanche area distribution, optionally on an existing figure if provided

    Parameters
    ----------
        model: SandNet.Model
            The sandpile model for which we want to plot avalanche areas
        figure: matplotlib.pyplot figure (default: None)
            The figure on which avalanche area distribution is plotted. If not provided a new one is created
        x_label: str (default: None)
            The label of the x of the plot
        y_label: str (default: None)
            The label of the y of the plot
        title: str (default: None)
            The title of the plot
    Other parameters
    ----------
        **kwargs:
            Options for the plot, passed directly to plt.scatter (see matplotlib.pyplot.scatter for more details
            about these options)
    Returns
    -------
        figure: matplotlib.pyplot figure
            The figure with the avalanche area distribution
    '''
    avalanche_areas = _cut_zeros(model.avalanche_areas_collector)
    if figure is None:
        figure = plt.figure()
    plt.figure(figure)

    #use np.histogram and plt.scatter to have a plot of points instead of a histogram
    y, x = np.histogram(avalanche_areas, bins = np.logspace(0, np.log10(max(avalanche_areas))), density=True)
    
    #the last element of y includes the last two bins of x. So x has one more element than y, which we have to cut out
    plt.scatter(x[:-1], y, **kwargs)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    return figure


def plot_avalanche_size(model: SandNet.Model, figure = None, x_label: str = None, y_label: str = None, 
                        title: str = None, **kwargs):
    '''
    Plots a scatter plot of avalanche size distribution, optionally on an existing figure if provided

    Parameters
    ----------
        model: SandNet.Model
            The sandpile model for which we want to plot avalanche sizes
        figure: matplotlib.pyplot figure (default: None)
            The figure on which avalanche size distribution is plotted. If not provided a new one is created
        x_label: str (default: None)
            The label of the x of the plot
        y_label: str (default: None)
            The label of the y of the plot
        title: str (default: None)
            The title of the plot
    Other parameters
    ----------
        **kwargs:
            Options for the plot, passed directly to plt.scatter (see matplotlib.pyplot.scatter for more details
            about these options)
    Returns
    -------
        figure: matplotlib.pyplot figure
            The figure with the avalanche size distribution
    '''
    avalanche_sizes = _cut_zeros(model.avalanche_sizes_collector)
    if figure is None:
        figure = plt.figure()
    plt.figure(figure)

    #use np.histogram and plt.scatter to have a plot of points instead of a histogram
    y, x = np.histogram(avalanche_sizes, bins = np.logspace(0, np.log10(max(avalanche_sizes))), density=True)
    
    #the last element of y includes the last two bins of x. So x has one more element than y, which we have to cut out
    plt.scatter(x[:-1], y, **kwargs)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    return figure


def fit_powerlaw(model: SandNet.Model, quantity: str = 'size', figure = None, x_label: str = None,
                       y_label: str = None, title: str = None, **kwargs):
    '''
    Fit and plot avalanche sizes of the sandpile model

    Given a sandpile model object, this function takes the list of all the avalanche sizes and fits them to a 
    power law distribution, using in particular the Pareto distribution from the scipy.stats package.
    After the fit, the function can plot the function of best fit in a logarithmic scale.

    Parameters
    ----------
        model: SandNet.Model
            The sandpile model for which we want to fit and plot avalanche sizes
        quantity: {'size', 'area'} (default: 'size')
            The quantity that must be fitted: avalanche size or avalanche area
        figure: matplotlib.pyplot figure (default: None)
            The figure on which fitted function is plotted. If not provided a new one is created.
        x_label: str (default: None)
            The label of the x of the plot.
        y_label: str (default: None)
            The label of the y of the plot.
        title: str (default: None)
            The title of the plot.
    Other parameters
    ----------
        **kwargs:
            Options for the plot, passed directly to plt.scatter (see matplotlib.pyplot.scatter for more details
            about these options).
    Returns
    -------
        fit_parameter: float
            The exponent of the power law obtained through fitting

            Note: since in the Pareto distribution used for fitting the exponent is $b+1$ (see scipy.stats.pareto documentation 
            for more details), the parameter obtained through fitting is $b$, and so the parameter returned by this function is not
            the exponent of the power law, but instead the exponent decreased by 1. We prefer not to adjust the returned parameter
            by adding 1 to avoid adding more uncertainty to the parameter due to floating point error
        figure: matplotlib.pyplot figure
            The figure with the plot of the fitted function
    Raises
    -------
        ValueError:
            If the quantity parameter is not an accepted one (size or area)
    '''
    if(quantity == 'size'):
        data = _cut_zeros(model.avalanche_sizes_collector)
    elif(quantity == 'area'):
        data = _cut_zeros(model.avalanche_areas_collector)
    else:
        raise ValueError("The quantity to be fitted must be either 'size' or 'area'")

    fit_parameter, loc, scale = pareto.fit(data, floc = 0, fscale = 1)
    #fix location and scale parameters for better fit

    if figure is None:
        figure = plt.figure()
    plt.figure(figure)

    #define the points where the fitted function is calculated for the plot. They are the same points
    #of the avalanche size and avalanche area scatter plots
    x = np.logspace(0, np.log10(max(data)))
    
    #the last element of x is not included in the avalanche size and avalanche area plots, so we exclude it also
    #from the fitted function
    y_fit = pareto.pdf(x[:-1], fit_parameter, loc = loc, scale = scale)
    plt.plot(x[:-1], y_fit, **kwargs)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    
    return fit_parameter, figure


if(__name__ == '__main__'):

    #execute the model on a 100x100 square grid and plot avalanche size and avalanche area distributions
    model = SandNet.Model(N=100, initial_grains='random')
    model.evolve(30000, avalanche_matrix=True)
    
    figure = plot_avalanche_size(model, x_label = "Avalanche size/area", y_label = "Density", title = "Avalanche size and area distribution",
                                 label = "Avalanche size")
    
    figure = plot_avalanche_area(model, figure=figure, label = "Avalanche area", color = "orange")
    
    fit_exponent, figure = fit_powerlaw(model, figure = figure, label = "Avalanche size fit", color = "red", linestyle = '--')
    plt.figure(figure)
    plt.show()

    print("Fit exponent avalanche size: " + str(fit_exponent))