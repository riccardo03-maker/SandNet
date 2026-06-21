#!/usr/bin/python
# -*- coding: utf-8 -*-

from SandNet.plots import _cut_zeros

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