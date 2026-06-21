#!/usr/bin/python
# -*- coding: utf-8 -*-

from .__version__ import __version__
from .SandNet import Model
from .plots import plot_avalanche_size, plot_avalanche_duration

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

__all__ = [
    '__version__', 
    'Model',
    'plot_avalanche_size',
    'plot_avalanche_duration'
]

