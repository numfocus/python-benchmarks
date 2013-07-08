import sys
import os
import timeit

setup = lambda module: '''
import numpy as np
from {} import arc_distance
n = 100
a = np.random.rand(n,2)
b = np.random.rand(n,2)
'''.format(module)
stmt = 'arc_distance(a,b)'

for module_path in sys.argv[1:]:
    module, _ = os.path.splitext(module_path)
    timer = timeit.Timer(stmt, setup(module))
    print "{}\t{}".format(module, min(timer.repeat(number=3)))

