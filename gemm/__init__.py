# Authors: James Bergstra
# License: MIT

import numpy as np


def make_env(M=512, N=512, K=512, seed=0, dtype=np.float64,
        alpha=1.5,
        beta=0.5):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype)
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    return (alpha, A, B, beta, C), {}

