# Authors: James Bergstra
# License: MIT
import theano
import theano.tensor as TT


def pairwise_theano_tensor_prepare(dtype):
    X = TT.matrix(dtype=str(dtype))
    dists = TT.sqrt(
        TT.sum(
            TT.sqr(X[:, None, :] - X),
            axis=2))
    rval = theano.function([X],
                           theano.Out(dists, borrow=True),
                           allow_input_downcast=True)
    rval.__name__ = 'pairwise_theano_broadcast_' + dtype
    return rval


def pairwise_theano_blas_prepare(dtype):
    X = TT.matrix(dtype=str(dtype))
    X_norm_2 = (X ** 2).sum(axis=1)
    dists = TT.sqrt(2 * X_norm_2 - TT.dot(X, X.T))
    rval = theano.function([X],
                           theano.Out(dists, borrow=True),
                           allow_input_downcast=True)
    rval.__name__ = 'pairwise_theano_blas_' + dtype
    return rval


benchmarks = (
    pairwise_theano_tensor_prepare('float32'),
    pairwise_theano_tensor_prepare('float64'),
    pairwise_theano_blas_prepare('float32'),
    pairwise_theano_blas_prepare('float64'),
)
