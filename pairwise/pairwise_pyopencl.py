# Author: James Bergstra
# License: MIT

import numpy as np
from pybench_pyopencl import pairwise_pyopencl

def pairwise_pyopencl_cpu(data):
    M, K = data.shape
    out = np.zeros((M, M), dtype=data.dtype)
    return pairwise_pyopencl.pairwise_pyopencl_cpu(data, data.T, out)

benchmarks = (
        pairwise_pyopencl_cpu,
)

