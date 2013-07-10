# Authors: Serge Guelton
# License: MIT
import numpy as np

def make_env(N=50):
    dtype = np.double
    image = np.zeros((N, N, 3), dtype=dtype)
    state = np.zeros((N, N, 2), dtype=dtype)
    state_next = np.empty_like(state)
         
    # colony 1 is strength 1 at position 0,0
    # colony 0 is strength 0 at all other positions
    state[0, 0, 0] = 1
    state[0, 0, 1] = 1

    window_radius = 10

    return (image, state, state_next, window_radius), {}
