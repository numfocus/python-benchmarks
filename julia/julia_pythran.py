from julia import julia_python
from pythran import compile_pythrancode
from inspect import getsource
import re, imp

# grab imports
imports = 'import numpy as np'
exports = '''
#pythran export julia_python_for_loops(float, float, int, float, float, float)
'''
modname = 'julia_pythran'

# grab the source from the original functions
sources = map(getsource,
        (julia_python.julia_python_for_loops,
            julia_python.kernel,
            )
        )
source = '\n'.join(sources)

# patch them
source = re.sub(r'cutoff=cutoff', 'cutoff', source)
source = re.sub(r'dtype=np.uint32', 'np.uint32', source)
source = re.sub(r'#"omp', '"omp', source)

# compile to a native module
native = compile_pythrancode(modname,
                             '\n'.join([imports, exports, source]),
                             cxxflags=['-O2', '-fopenmp'])

# load it
native = imp.load_dynamic(modname, native)

benchmarks = (
    ("julia_pythran_for_loops",
     native.julia_python_for_loops),
)
