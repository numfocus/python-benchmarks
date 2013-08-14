# Authors: James Bergstra
# License: MIT

import numpy as np


# -- too slow to run directly, but good for JIT by other systems
def gemm_python_nested_for_loops(alpha, A, B, beta, C):
    M, N = C.shape
    K = A.shape[1]
    #"omp parallel for private(j, d, k, tmp)"
    for i in range(M):
        for j in range(N):
            d = 0.0
            for k in range(K):
                tmp = A[i, k] * B[j, k]
                d += tmp
            C[i, j] = alpha * d + beta * C[i, j]
    return C


def gemm_python_inner_numpy(alpha, A, B, beta, C):
    M, N = C.shape
    for i in xrange(M):
        for j in xrange(N):
            C[i, j] *= beta
            C[i, j] += alpha * np.dot(A[i, :], B[:, j])
    return C


def gemm_python_broadcast_numpy(alpha, A, B, beta, C):
    return alpha * (A[:, None, :] * B.T[None, :, :]).sum(axis=2) + beta * C


def gemm_python_numpy_dot(alpha, A, B, beta, C):
    C *= beta
    C += alpha * np.dot(A, B)
    return C


benchmarks = (
    gemm_python_inner_numpy,
    gemm_python_broadcast_numpy,
    gemm_python_numpy_dot,
)
