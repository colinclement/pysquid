"""
edgetest.py

author: Colin Clement
date: 2018-06-13

This script tests whether the edge subtraction works in a very basic way by
calculating the field due to a constant image, which should have no field if we
subtract the edges
"""

import numpy as np

from pysquid.kernels.kernel import Kernel

kern = Kernel((100,100), edges=False, padding=(2,2))
constp = 10*np.ones((104,104))
const = 10*np.ones((100,100))
flux = kern.applyM(constp)
g = kern.applyMt(const)
