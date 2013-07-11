# Authors: Serge Guelton
# License: MIT
'''Compute the derivative of the Rosenbrock function

This functions tests the ability of compiler to fuse numpy operators, use
negative indexing and handle memory view instead of making copies when slicing.

see also http://en.wikipedia.org/wiki/Rosenbrock_function
'''

import numpy as np


def make_env(N=1000000):
    rng = np.random.RandomState(42)
    x = rng.rand(N)
    return (x,), {}
