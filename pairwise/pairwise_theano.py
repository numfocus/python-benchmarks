
import theano
import theano.tensor as TT

def pairwise_theano_prepare(dtype):
    X = TT.matrix(dtype=str(dtype))
    X_norm_2 = (X ** 2).sum(axis=1)
    dists = TT.sqrt(2 * X_norm_2 - TT.dot(X, X.T))
    rval = theano.function([X], dists, allow_input_downcast=True)
    rval.__name__ = 'pairwise_theano_' + dtype
    return rval

benchmarks = (
    pairwise_theano_prepare('float32'),
    pairwise_theano_prepare('float64'),
)


