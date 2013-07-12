# Authors: Jake Vanderplas, Alex Rubinsteyn, Olivier Grisel
# License: MIT

import numpy as np


def pairwise_python_nested_for_loops(data):
    n_samples, n_features = data.shape
    distances = np.empty((n_samples, n_samples), dtype=data.dtype)
    #"omp parallel for private(j, d, k, tmp)"
    for i in range(n_samples):
        for j in range(n_samples):
            d = 0.0
            for k in range(n_features):
                tmp = data[i, k] - data[j, k]
                d += tmp * tmp
            distances[i, j] = np.sqrt(d)
    return distances


def pairwise_python_inner_numpy(data):
    n_samples = data.shape[0]
    result = np.empty((n_samples, n_samples), dtype=data.dtype)
    for i in xrange(n_samples):
        for j in xrange(n_samples):
            result[i, j] = np.sqrt(np.sum((data[i, :] - data[j, :]) ** 2))
    return result


def pairwise_python_broadcast_numpy(data):
    return np.sqrt(((data[:, None, :] - data) ** 2).sum(axis=2))


def pairwise_python_numpy_dot(data):
    X_norm_2 = (data ** 2).sum(axis=1)
    dists = np.sqrt(2 * X_norm_2 - np.dot(data, data.T))
    return dists


benchmarks = (
    pairwise_python_nested_for_loops,
    pairwise_python_inner_numpy,
    pairwise_python_broadcast_numpy,
    pairwise_python_numpy_dot,
)
