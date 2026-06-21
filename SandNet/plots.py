#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import SandNet
import numpy as np
import powerlaw

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

__all__ =[
    'plot_avalanche_size',
    'plot_avalanche_duration'
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


def plot_avalanche_size(model: SandNet.Model):
    '''
    Fit and plot avalanche sizes of the sandpile model

    Given a sandpile model object, this function takes the list of all the avalanche sizes and fits them to a 
    power law distribution. The method used for the fitting is the one developed in Clauset et al. (2009), which
    is specifically developed to fit power law distributions. This method is implemented in the Python package
    powerlaw, which is used in this function. After the fit, the function plots avalanche sizes together with
    the function of best fit in a logarithmic scale.

    Parameters
    ----------
        model: SandNet.Model
            The sandpile model for which we want to fit and plot avalanche sizes

    Returns
    -------
        alpha: float
            The exponent of the power law obtained through fitting
        x_min: int
            The minimum value of avalanche size considered for fitting (see Reference for more details)
        sigma: float
            The error on the power law exponent

            Note: this is only a rough estimation of the error, since we are neglecting the error on the choice
            of x_min, and we are assuming that power law is the correct distribution of our data

    References
    -------
        Clauset A., Shalizi C. R., Newman M. E. J., 2009, *Power-Law Distributions in Empirical Data*, SIAM Review, 51, 661–703
    '''
    avalanche_sizes = _cut_zeros(model.avalanche_sizes_collector)

    #use np.histogram and plt.scatter to have a plot of points instead of a histogram
    y, x = np.histogram(avalanche_sizes, bins = np.logspace(0, np.log10(max(avalanche_sizes))), density=True)

    ax = plt.subplot()
    
    #the last element of y includes the last two bins of x. So x has one more element than y, which we have to cut out
    ax.scatter(x[:-1], y, label = "data points")

    fit = powerlaw.Fit(avalanche_sizes, discrete = True)
    fit.power_law.plot_pdf(ax, linestyle = '--', color = "red", label = "power law fit")
    #the plot with the powerlaw package already sets the log scale for both axes, so we don't have to do it manually

    plt.xlabel("Avalanche size")
    plt.ylabel("Density")
    plt.title("Power law distribution of avalanche sizes")
    plt.legend()
    plt.show()
    
    return fit.power_law.alpha, fit.power_law.xmin, fit.power_law.sigma


def plot_avalanche_duration(model: SandNet.Model):
    '''
    Fit and plot avalanche durations of the sandpile model

    Given a sandpile model object, this function takes the list of all the avalanche durations and fits them to a 
    power law distribution. The method used for the fitting is the one developed in Clauset et al. (2009), which
    is specifically developed to fit power law distributions. This method is implemented in the Python package
    powerlaw, which is used in this function. After the fit, the function plots avalanche durations together with
    the function of best fit in a logarithmic scale.

    Parameters
    ----------
        model: SandNet.Model
            The sandpile model for which we want to fit and plot avalanche durations

    Returns
    -------
        alpha: float
            The exponent of the power law obtained through fitting
        x_min: int
            The minimum value of avalanche duration considered for fitting (see Reference for more details)
        sigma: float
            The error on the power law exponent

            Note: this is only a rough estimation of the error, since we are neglecting the error on the choice
            of x_min, and we are assuming that power law is the correct distribution of our data

    References
    -------
        Clauset A., Shalizi C. R., Newman M. E. J., 2009, *Power-Law Distributions in Empirical Data*, SIAM Review, 51, 661–703
    '''
    avalanche_durations = _cut_zeros(model.avalanche_durations_collector)

    #use np.histogram and plt.scatter to have a plot of points instead of a histogram
    y, x = np.histogram(avalanche_durations, bins = np.logspace(0, np.log10(max(avalanche_durations))), density=True)

    ax = plt.subplot()
    
    #the last element of y includes the last two bins of x. So x has one more element than y, which we have to cut out
    ax.scatter(x[:-1], y, label = "data points")

    fit = powerlaw.Fit(avalanche_durations, discrete = True)
    fit.power_law.plot_pdf(ax, linestyle = '--', color = "red", label = "power law fit")
    #the plot with the powerlaw package already sets the log scale for both axes, so we don't have to do it manually

    plt.xlabel("Avalanche duration")
    plt.ylabel("Density")
    plt.title("Power law distribution of avalanche durations")
    plt.legend()
    plt.show()
    
    return fit.power_law.alpha, fit.power_law.xmin, fit.power_law.sigma


if(__name__ == '__main__'):
    import pickle
    with open("./saves/test", "rb") as fp:
        avalanche_sizes = pickle.load(fp)
    model = SandNet.Model()
    model.avalanche_sizes_collector = avalanche_sizes
    plot_avalanche_size(model)