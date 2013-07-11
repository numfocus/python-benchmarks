# Authors: Yuancheng Peng
# License: MIT

from arc_distance import arc_distance_python
from pythran import compile_pythrancode
from inspect import getsource
import re
import imp

# grab imports
imports = '''
import numpy as np
from math import *
'''

exports = '''
#pythran export arc_distance_python_nested_for_loops(float [][], float [][])
'''

modname = 'arc_distance_pythran'

# grab the source from original functions
funs = (arc_distance_python.arc_distance_python_nested_for_loops,)
sources = map(getsource, funs)
source = '\n'.join(sources)

# patch
source = re.sub(r'\[a_nrows, b_nrows\]', '(a_nrows, b_nrows)', source)

# compile to native module
source = '\n'.join([exports, imports, source])
native = compile_pythrancode(modname, source)

# load
native = imp.load_dynamic(modname, native)

benchmarks = (("arc_distance_pythran_nested_for_loops",
               native.arc_distance_python_nested_for_loops),
              )
