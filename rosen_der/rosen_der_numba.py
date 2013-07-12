# Authors: Serge Guelton
# License: MIT

from rosen_der import rosen_der_python
from numba import autojit


# segfaults...
#benchmarks = (
#    ("rosen_der_numba",
#     autojit(rosen_der_python.rosen_der_python)),
#)
