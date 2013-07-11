# Authors: Yuancheng Peng
# License: MIT
"""Computes the arc distance between a collection of points

This code is challenging because it requires efficient vectorisation of
trigonometric functions that are note natively supported in SSE/AVX. The numpy
version makes use of numpy.tile and transpose, which proves to be challenging
too.

See also http://en.wikipedia.org/wiki/Great-circle_distance
"""

import numpy as np


def make_env(n=100):
    rng = np.random.RandomState(42)
    a = rng.rand(n, 2)
    b = rng.rand(n, 2)
    return (a, b), {}
