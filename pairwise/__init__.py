# Authors: Olivier Grisel
# License: MIT

import numpy as np


def make_env(shape=(200, 100), seed=0, dtype=np.float):
    rng = np.random.RandomState(seed)
    data = np.asarray(rng.normal(size=shape), dtype=dtype)
    return (data,), {}
