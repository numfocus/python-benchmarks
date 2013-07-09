# Authors: Jake Vanderplas, Olivier Grisel
# License: MIT

import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_cython_for_loops(double[:, ::1] data):
    cdef int n_samples = data.shape[0]
    cdef int n_features = data.shape[1]
    cdef double tmp, d
    cdef double[:, ::1] distances = np.empty((n_samples, n_samples),
                                             dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_samples):
            d = 0.0
            for k in range(n_features):
                tmp = data[i, k] - data[j, k]
                d += tmp * tmp
            distances[i, j] = sqrt(d)
    return np.asarray(distances)


benchmarks = (
    pairwise_cython_for_loops,
)