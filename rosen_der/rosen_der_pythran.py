# Authors: Serge Guelton
# License: MIT

from rosen_der import rosen_der_python
from pythran import compile_pythrancode
from inspect import getsource
import imp

# grab imports
imports = 'import numpy'
exports = '''
#pythran export rosen_der_numpy(float [])
'''
modname = 'rosen_der_pythran'

# grab the source from the original functions
sources = map(getsource,
              (rosen_der_python.rosen_der_numpy,)
              )
source = '\n'.join(sources)

# patch them

# compile to a native module
native = compile_pythrancode(modname,
                             '\n'.join([imports, exports, source]),
                             cxxflags=['-O2', '-fopenmp'])

# load it
native = imp.load_dynamic(modname, native)

benchmarks = (
    ("rosen_der_pythran",
     native.rosen_der_numpy),
)
