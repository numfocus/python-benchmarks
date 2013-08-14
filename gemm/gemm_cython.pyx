
# Authors: Jake Vanderplas, Olivier Grisel
# License: MIT

import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def gemm_cython_for_loops(
        double alpha,
        double[:, ::1] A,
        double[:, ::1] B,
        double beta,
        double[:, ::1] C,
        ):
    cdef int M = C.shape[0]
    cdef int N = C.shape[1]
    cdef int K = A.shape[1]
    cdef double tmp, d
    for i in range(M):
        for j in range(N):
            d = 0.0
            for k in range(K):
                d += A[i, k] * B[k, j]
            C[i, j] = alpha * d + beta * C[i, j]


benchmarks = (
    gemm_cython_for_loops,
)
