#!/usr/bin/python
# -*- coding: utf-8 -*-

from SandNet.plots import _cut_zeros
import SandNet
import pytest

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


# Testing _cut_zeros function
def test_cut_zeros():
    '''
    Tests the correct behaviour of the _cut_zeros function

    GIVEN: a list of 10 integers with 4 zeros
    WHEN: I use the _cut_below method to remove all values below 1 (so only zeros)
    THEN: I obtain as output a list of 6 elements
    '''
    vector = [1, 2, 0, 4, 5, 0, 0, 8, 9, 0]
    vector = _cut_zeros(vector)
    assert(len(vector) == 6)


#Testing exception in fit function


def test_wrong_data_input():
    '''
    Tests the raise of an Exception when a data type different from avalanche size or avalanche area is provided as
    input to the fit_powerlaw function

    GIVEN: any kind of sandpile model on a network
    WHEN: I try a wrong input to the 'quantity' parameter of fit_powerlaw
    THEN: the code raises a ValueError
    '''
    with pytest.raises(ValueError):
        SandNet.fit_powerlaw(SandNet.Model(), quantity = 'not_standard_quantity')