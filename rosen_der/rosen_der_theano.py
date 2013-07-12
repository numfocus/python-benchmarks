# Authors: Travis E. Oliphant (numpy version), Serge Guelton (python version)
#          James Bergstra (theano version)
# License: BSD
# Source: https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
import theano
from theano import tensor as TT


def rosen_der_theano_prepare(dtype):
    x = TT.vector(dtype=dtype)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = TT.zeros_like(x)
    der = TT.set_subtensor(
        der[1:-1],
        (+ 200 * (xm - xm_m1 ** 2)
                 - 400 * (xm_p1 - xm ** 2) * xm
                 - 2 * (1 - xm)))
    der = TT.set_subtensor(
        der[0],
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]))
    der = TT.set_subtensor(
        der[-1],
        200 * (x[-1] - x[-2] ** 2))
    rval = theano.function([x], der, allow_input_downcast=True)
    rval.__name__ = 'rosen_der_theano_' + dtype
    return rval

benchmarks = (rosen_der_theano_prepare('float32'),
              rosen_der_theano_prepare('float64'))
