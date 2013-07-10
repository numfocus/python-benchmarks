from growcut import growcut_python
from pythran import compile_pythrancode
from inspect import getsource
import re, imp

# grab imports
imports = 'import math'
exports = '''
#pythran export growcut_python(float[][][], float[][][], float [][][], int)
'''
modname = 'growcut_pythran'

# grab the source from the original functions
sources = map(getsource,
        (growcut_python.window_floor,
            growcut_python.window_ceil,
            growcut_python.growcut_python,
            )
        )
source = '\n'.join(sources)

# compile to a native module
native = compile_pythrancode(modname,
                             '\n'.join([imports, exports, source]))

# load it
native = imp.load_dynamic(modname, native)

benchmarks = (
        ("growcut_pythran",
            native.growcut_python),
)
