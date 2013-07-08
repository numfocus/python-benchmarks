# Authors: Jake Vanderplas, Alex Rubinsteyn, Olivier Grisel
# License: MIT

import numpy as np


def pairwise_python_nested_for_loops(data):
    n_samples, n_features = data.shape
    distances = np.empty((n_samples, n_samples), dtype=data.dtype)
    for i in range(n_samples):
        for j in range(n_samples):
            d = 0.0
            for k in range(n_features):
                tmp = data[i, k] - data[j, k]
                d += tmp * tmp
            distances[i, j] = np.sqrt(d)
    return distances


def pairwise_python_inner_broadcasting(data):
    n_samples = data.shape[0]
    result = np.empty((n_samples, n_samples), dtype=data.dtype)
    for i in xrange(n_samples):
        for j in xrange(n_samples):
            result[i, j] = np.sqrt(np.sum((data[i, :] - data[j, :]) ** 2))
    return result


benchmarks = (
    pairwise_python_nested_for_loops,
    pairwise_python_inner_broadcasting,
)
