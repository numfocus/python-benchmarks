# Authors: Olivier Grisel
# License: MIT

"""Computes the Euclidean distance between each pair of rows in a matrix.

In LaTeX:
    Y[i, j] = sqrt{ \sum_k (A[i, k] - B[j, k])^2 }

This computation is a core routine of many machine learning algorithms that
rely on neighbourhood computations.

"""

import numpy as np


def make_env(shape=(300, 150), seed=0, dtype=np.double):
    rng = np.random.RandomState(seed)
    data = np.asarray(rng.normal(size=shape), dtype=dtype)
    return (data,), {}
