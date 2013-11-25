# Authors: James Bergstra
# License: MIT

"""Computes the GEMM routine from the BLAS standard.

BLAS defines the xGEMM (DGEMM, SGEMM, CGEMM, ZGEMM) operations as
dense matrix multiplication and accumulation as follows:

C <- alpha A x B + beta C

Here (alpha, beta) are scalars and (A, B, C) are matrices.

This operation is one of the most widely-used and most-studied kernels in
high-performance computing, and BLAS implementations such as OpenBLAS, ATLAS,
and MKL provide highly optimized implementations of this operation. GEMM
implementations provide a real-life measure of peak performance on a
particular platform.

Note that the GEMM interface does not actually describe an algorithm, and the
standard does not require particular numerical accuracy.  There are sub-cubic
algorithms (e.g. Strassen), and there are also cubic algorithms that are
"blocked" to be more cache-friendly.  I believe that OpenBLAS and ATLAS use
blocked cubic algorithms, based on the asymptotic GFLOP/s attributed to MKL,
I would guess it uses blocked cubic algorithms too.

My hope with this benchmark is that it can be used to develop fast, readable
GEMM implementations. I'm curious, for example, if a readable, blocked
algorithm in pure Python could be compiled to a reasonable-performing
implementation.

"""

import numpy as np


def make_env(M=512, N=512, K=512, seed=0, dtype=np.float64,
        alpha=1.5,
        beta=0.5):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype)
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    return (alpha, A, B, beta, C), {}

