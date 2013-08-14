# Authors: James Bergstra
# License: MIT
import theano
import theano.tensor as TT


def gemm_theano_tensor_prepare(dtype):
    alpha = TT.scalar(dtype=str(dtype))
    beta = TT.scalar(dtype=str(dtype))
    A = TT.matrix(dtype=str(dtype))
    B = TT.matrix(dtype=str(dtype))
    C = TT.matrix(dtype=str(dtype))
    Z = alpha * TT.sum(A[:, None, :] * B, axis=2) + beta * C
    name = 'gemm_theano_broadcast_' + dtype
    Cin = theano.In(C, mutable=True, borrow=True)
    rval = theano.function([alpha, A, B, beta, Cin],
                           theano.Out(Z, borrow=True),
                           allow_input_downcast=True, name=name)
    rval.__name__ = name
    return rval


def gemm_theano_blas_prepare(dtype):
    alpha = TT.scalar(dtype=str(dtype))
    beta = TT.scalar(dtype=str(dtype))
    A = TT.matrix(dtype=str(dtype))
    B = TT.matrix(dtype=str(dtype))
    C = TT.matrix(dtype=str(dtype))
    Z = alpha * TT.dot(A, B) + beta * C
    Cin = theano.In(C, mutable=True, borrow=True)
    name = 'gemm_theano_blas_' + dtype
    rval = theano.function([alpha, A, B, beta, Cin],
                           theano.Out(Z, borrow=True),
                           allow_input_downcast=True, name=name)
    rval.__name__ = name
    return rval


benchmarks = (
    #gemm_theano_tensor_prepare('float32'),
    gemm_theano_tensor_prepare('float64'),
    #gemm_theano_blas_prepare('float32'),
    gemm_theano_blas_prepare('float64'),
)

