# Authors: Olivier Grisel
# License: MIT

import numpy as np


def make_env(shape=(300, 150), seed=0, dtype=np.double):
    rng = np.random.RandomState(seed)
    data = np.asarray(rng.normal(size=shape), dtype=dtype)
    return (data,), {}
