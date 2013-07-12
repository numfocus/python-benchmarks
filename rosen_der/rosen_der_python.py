# Authors: Travis E. Oliphant (numpy version), Serge Guelton (python version)
# License: BSD
# Source: https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
import numpy


def rosen_der_numpy(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = numpy.zeros_like(x)
    der[1:-1] = (+ 200 * (xm - xm_m1 ** 2)
                 - 400 * (xm_p1 - xm ** 2) * xm
                 - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_der_python(x):
    n = x.shape[0]
    der = numpy.zeros_like(x)

    for i in range(1, n - 1):
        der[i] = (+ 200 * (x[i] - x[i - 1] ** 2)
                  - 400 * (x[i + 1]
                           - x[i] ** 2) * x[i]
                  - 2 * (1 - x[i]))
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

benchmarks = (rosen_der_numpy, rosen_der_python)
