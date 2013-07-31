# Authors: Serge Guelton
# License: MIT

from rosen_der import rosen_der_python
from parakeet import jit

benchmarks = (
     ("rosen_der_loops_parakeet", jit(rosen_der_python.rosen_der_python)),
     ("rosen_der_numpy_parakeet", jit(rosen_der_python.rosen_der_numpy))
 )
