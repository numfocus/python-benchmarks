# Authors: Olivier Grisel
# License: MIT

import numpy as np


def make_env(shape=(1000, 100), seed=0, dtype=np.float):
    rng = np.random.RandomState(seed)
    return dict(data=np.asarray(rng.random(shape), dtype=dtype))
