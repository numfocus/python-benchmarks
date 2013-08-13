# Authors: James Bergstra
# License: MIT

import numpy as np


def make_env(shape=(256, 256, 256), seed=0, dtype=np.float):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(shape[0], shape[1])), dtype=dtype)
    B = np.asarray(rng.normal(size=(shape[1], shape[2])), dtype=dtype)
    C = np.asarray(rng.normal(size=(shape[0], shape[2])), dtype=dtype)
    alpha = 1.5
    beta = 0.3
    return (alpha, A, B, beta, C), {}
