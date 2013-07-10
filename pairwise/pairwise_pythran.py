# Authors: Serge "Sans Paille" Guelton
# License: MIT

from pairwise import pairwise_python
from pythran import compile_pythrancode
from inspect import getsource
import re
import imp

# grab imports
imports = 'import numpy as np'
exports = '#pythran export pairwise_python_nested_for_loops(float[][])'
modname = 'pairwise_pythran'

# grab the source from the original function
source = getsource(pairwise_python.pairwise_python_nested_for_loops)

# a few rewriting rules to prune unsupported features
source = re.sub(r'dtype=data.dtype', 'np.double', source)
source = re.sub(r'#"omp', '"omp', source)

# compile to a native module
native = compile_pythrancode(modname,
                             '\n'.join([imports, exports, source]),
                             cxxflags=['-O2', '-fopenmp']
                             )

# load it
native = imp.load_dynamic(modname, native)

benchmarks = (
    ("pairwise_pythran_nested_for_loops",
     native.pairwise_python_nested_for_loops),
)
